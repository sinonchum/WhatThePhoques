"""StegaExtSegNetV2: StegaExtSegNet + 5-crop (320x320 -> 400x400) frozen StegaStamp local message maps.

Why: In StegaExtSegNet v1, large mIoU reached 0.672 while small was 0.578 because small patches
(<20% area) are too small in the single global 400x400 resize for the frozen STN decoder to lock on.
Running 5 overlapping 320x320 crops (4 corners + center) in one batched no_grad pass gives strong
known-message evidence on small patches, painted onto a 16x16 spatial map + best-crop vector.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from stegastamp_pt import StegaStampDecoderPT, _pad_same_s2
from stega_extnet import block, HighPass, _norm_attr


class StegaExtSegNetV2(nn.Module):
    method = "stegastamp"

    def __init__(self, onnx_path: str | Path, dec: int = 160):
        super().__init__()
        base = StegaStampDecoderPT(onnx_path)
        self.frozen = base
        for p in self.frozen.parameters():
            p.requires_grad_(False)

        self.enc = nn.ModuleDict({
            "c1": copy.deepcopy(base.dec_conv_1),
            "c2": copy.deepcopy(base.dec_conv_2),
            "c3": copy.deepcopy(base.dec_conv_3),
            "c4": copy.deepcopy(base.dec_conv_4),
            "c5": copy.deepcopy(base.dec_conv_5),
            "c6": copy.deepcopy(base.dec_conv_6),
            "c7": copy.deepcopy(base.dec_conv_7),
        })
        for p in self.enc.parameters():
            p.requires_grad_(True)

        self.hp_front = HighPass(scale=40.0, clip=3.0)
        self.hp_stem = nn.Sequential(
            block(12, 32, stride=1),   # 512
            block(32, 64, stride=2),   # 256
            block(64, dec, stride=2),  # 128
        )

        self.lat4 = nn.Conv2d(64, dec, 1)
        self.lat5 = nn.Conv2d(64, dec, 1)
        self.lat6 = nn.Conv2d(128, dec, 1)
        self.lat7 = nn.Conv2d(128, dec, 1)

        self.attr4 = nn.Sequential(nn.Conv2d(64 + 3, dec, 1), nn.SiLU(inplace=True))
        self.attr5 = nn.Sequential(nn.Conv2d(64 + 3, dec, 1), nn.SiLU(inplace=True))
        self.attr6 = nn.Sequential(nn.Conv2d(128 + 3, dec, 1), nn.SiLU(inplace=True))
        self.attr7 = nn.Sequential(nn.Conv2d(128 + 3, dec, 1), nn.SiLU(inplace=True))

        # 5-crop local message map (102 channels painted on 16x16 grid: 100 signed + mean + bit_acc)
        self.crop_map_proj7 = nn.Sequential(nn.Conv2d(102, dec, 1), nn.SiLU(inplace=True))
        self.crop_map_proj6 = nn.Sequential(nn.Conv2d(102, dec, 1), nn.SiLU(inplace=True))

        # Global known-message summary: STN(102) + no-STN(102) + max-over-5-crops(102) = 306
        self.msg_glob = nn.Sequential(
            nn.Linear(306, dec),
            nn.SiLU(inplace=True),
            nn.Linear(dec, dec),
            nn.SiLU(inplace=True),
        )

        self.top = block(4 * dec, dec)
        self.glob = nn.Sequential(nn.Linear(dec, dec), nn.SiLU(inplace=True))
        self.smooth6 = block(dec, dec)
        self.smooth5 = block(dec, dec)
        self.smooth4 = block(dec, dec)
        self.fuse_hp = block(2 * dec, dec)
        self.head = nn.Sequential(block(dec, 64), nn.Conv2d(64, 1, 1))
        self.cls = nn.Linear(dec, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen.eval()
        return self

    def backbone_parameters(self):
        return self.enc.parameters()

    def head_parameters(self):
        skip = {id(p) for p in self.enc.parameters()} | {id(p) for p in self.frozen.parameters()}
        return [p for p in self.parameters() if id(p) not in skip]

    def _encode_trainable(self, x01_400: torch.Tensor):
        h = x01_400 - 0.5
        f1 = F.relu(self.enc["c1"](_pad_same_s2(h)))
        f2 = F.relu(self.enc["c2"](f1))
        f3 = F.relu(self.enc["c3"](_pad_same_s2(f2)))
        f4 = F.relu(self.enc["c4"](f3))                   # 64 x 100 x 100
        f5 = F.relu(self.enc["c5"](_pad_same_s2(f4)))     # 64 x 50 x 50
        f6 = F.relu(self.enc["c6"](_pad_same_s2(f5)))     # 128 x 25 x 25
        f7 = F.relu(self.enc["c7"](f6))                   # 128 x 13 x 13
        return f4, f5, f6, f7

    def _frozen_message_signals(self, x01_512: torch.Tensor, x01_400: torch.Tensor, bits: torch.Tensor):
        sign = 2.0 * bits.float() - 1.0  # [B, 100]
        B = x01_512.shape[0]
        with torch.autocast("cuda", enabled=False):
            x_det = x01_400.detach().float()
            x512 = x01_512.detach().float()

            # 1) Global STN + 5 overlapping 320x320 crops resized to 400x400 in one batched pass
            with torch.no_grad():
                boxes = [
                    (0, 320, 0, 320),
                    (0, 320, 192, 512),
                    (192, 512, 0, 320),
                    (192, 512, 192, 512),
                    (96, 416, 96, 416),
                ]
                crops = [
                    F.interpolate(x512[:, :, y0:y1, x0:x1], size=(400, 400), mode="bicubic",
                                  align_corners=False, antialias=True).clamp(0.0, 1.0)
                    for (y0, y1, x0, x1) in boxes
                ]
                all_in = torch.cat([x_det] + crops, dim=0)  # [6B, 3, 400, 400]
                all_logits = self.frozen(all_in, use_stn=True)  # [6B, 100]
                chunks = all_logits.chunk(6, dim=0)

                def _pack(lgt):
                    s = lgt * sign
                    return torch.cat([
                        (s / 4.0).clamp(-4.0, 4.0),
                        s.mean(1, keepdim=True) / 4.0,
                        (s > 0).float().mean(1, keepdim=True) * 2.0 - 1.0,
                    ], dim=1)  # [B, 102]

                v_stn = _pack(chunks[0])
                crop_vecs = [_pack(c) for c in chunks[1:]]  # 5 x [B, 102]
                v_best_crop = torch.stack(crop_vecs, dim=0).max(dim=0).values  # [B, 102]

                # Paint 5 crop vectors onto a 16x16 spatial grid (each 320/512 = 10/16 cells)
                canvas = torch.zeros((B, 102, 16, 16), device=x512.device, dtype=torch.float32)
                weight = torch.zeros((1, 1, 16, 16), device=x512.device, dtype=torch.float32)
                grid_boxes = [(0, 10, 0, 10), (0, 10, 6, 16), (6, 16, 0, 10), (6, 16, 6, 16), (3, 13, 3, 13)]
                for vec, (gy0, gy1, gx0, gx1) in zip(crop_vecs, grid_boxes):
                    canvas[:, :, gy0:gy1, gx0:gx1] += vec[:, :, None, None]
                    weight[:, :, gy0:gy1, gx0:gx1] += 1.0
                crop_map16 = canvas / weight

            # 2) Unwarped spatial attribution w.r.t. known 100-bit message
            with torch.enable_grad():
                x_req = x_det.requires_grad_(True)
                logits_no, (_, f4, f5, f6, f7) = self.frozen.forward_features(x_req, use_stn=False)
                obj = (logits_no * sign).mean(dim=1).sum()
                g4, g5, g6, g7 = torch.autograd.grad(
                    obj, [f4, f5, f6, f7], retain_graph=False, create_graph=False
                )
            with torch.no_grad():
                s_no = logits_no.detach() * sign
                v_no = torch.cat([
                    (s_no / 4.0).clamp(-4.0, 4.0),
                    s_no.mean(1, keepdim=True) / 4.0,
                    (s_no > 0).float().mean(1, keepdim=True) * 2.0 - 1.0,
                ], dim=1)
                glob_vec = torch.cat([v_stn, v_no, v_best_crop], dim=1)  # [B, 306]
                a4 = _norm_attr(g4.detach(), f4.detach())
                a5 = _norm_attr(g5.detach(), f5.detach())
                a6 = _norm_attr(g6.detach(), f6.detach())
                a7 = _norm_attr(g7.detach(), f7.detach())
        return glob_vec, crop_map16, (a4, a5, a6, a7)

    def forward(self, x: torch.Tensor, bits: torch.Tensor):
        with torch.autocast("cuda", enabled=False):
            xf = x.float()
            x01 = (xf + 1.0) * 0.5
            x400 = F.interpolate(x01, size=(400, 400), mode="bicubic", align_corners=False, antialias=True).clamp(0.0, 1.0)
            hp_in = self.hp_front(xf)
        hp128 = self.hp_stem(hp_in)

        glob_vec, crop_map16, (a4, a5, a6, a7) = self._frozen_message_signals(x01, x400, bits)
        ef4, ef5, ef6, ef7 = self._encode_trainable(x400)

        mg = self.msg_glob(glob_vec)[:, :, None, None].expand(-1, -1, ef7.shape[-2], ef7.shape[-1])
        cm7 = F.interpolate(self.crop_map_proj7(crop_map16), size=ef7.shape[-2:], mode="bilinear", align_corners=False)
        y = self.top(torch.cat([self.lat7(ef7), self.attr7(a7), mg, cm7], dim=1))
        g = self.glob(y.mean(dim=(2, 3)))
        y = y + g[:, :, None, None]

        cm6 = F.interpolate(self.crop_map_proj6(crop_map16), size=ef6.shape[-2:], mode="bilinear", align_corners=False)
        y = self.smooth6(F.interpolate(y, size=ef6.shape[-2:], mode="nearest") + self.lat6(ef6) + self.attr6(a6) + cm6)
        y = self.smooth5(F.interpolate(y, size=ef5.shape[-2:], mode="nearest") + self.lat5(ef5) + self.attr5(a5))
        y = self.smooth4(F.interpolate(y, size=ef4.shape[-2:], mode="nearest") + self.lat4(ef4) + self.attr4(a4))

        y128 = self.fuse_hp(torch.cat([F.interpolate(y, size=(128, 128), mode="bilinear", align_corners=False), hp128], dim=1))
        logits = self.head(y128)[:, 0]
        return logits, self.cls(g)[:, 0]
