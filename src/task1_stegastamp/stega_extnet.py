"""StegaExtSegNet: official PyTorch StegaStamp extractor + known-message spatial attribution + HighPass stem.

Combines three complementary signals at native resolutions:
1) Frozen official StegaStamp decoder (loaded from stega_stamp.onnx):
   - Global STN-warped 100-bit signed logits, mean margin, and bit accuracy (separates large vs small).
   - Spatially aligned (unwarped) first-order message attribution (grad * act) w.r.t. the known 100-bit
     message at 100x100, 50x50, 25x25, and 13x13.
2) Trainable copy of the official StegaStamp 7-conv decoder backbone (pre-tuned to StegaStamp residuals).
3) Full-resolution 512->128 HighPass residual stem (captures local residual boundaries for small patches).
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from stegastamp_pt import StegaStampDecoderPT, _pad_same_s2


def block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )


class HighPass(nn.Module):
    def __init__(self, scale: float = 40.0, clip: float = 3.0):
        super().__init__()
        self.scale, self.clip = scale, clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3 = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, 1)
        b5 = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, 1)
        hp = torch.cat([x - b3, x - b5, b3 - b5], 1) * self.scale
        return torch.cat([x, hp.clamp(-self.clip, self.clip)], 1)


def _norm_attr(g: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Pack channel-wise grad*act + spatial summary maps, normalized per sample."""
    ga = g * f
    scale = ga.abs().mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
    ga_n = (ga / scale).clamp(-6.0, 6.0)
    pos = ga_n.mean(dim=1, keepdim=True)
    absm = ga_n.abs().mean(dim=1, keepdim=True)
    gn = g.norm(dim=1, keepdim=True)
    gn = (gn / gn.mean(dim=(2, 3), keepdim=True).clamp(min=1e-6)).clamp(0.0, 6.0)
    return torch.cat([ga_n, pos, absm, gn], dim=1)


class StegaExtSegNet(nn.Module):
    method = "stegastamp"

    def __init__(self, onnx_path: str | Path, dec: int = 128):
        super().__init__()
        base = StegaStampDecoderPT(onnx_path)
        self.frozen = base
        for p in self.frozen.parameters():
            p.requires_grad_(False)

        # Trainable copy of the 7 decoder convs (no STN, spatially aligned with output mask)
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

        # Full-res 512 -> 128 HighPass branch
        self.hp_front = HighPass(scale=40.0, clip=3.0)
        self.hp_stem = nn.Sequential(
            block(12, 32, stride=1),   # 512
            block(32, 64, stride=2),   # 256
            block(64, dec, stride=2),  # 128
        )

        # Projections for trainable backbone features (f4:64, f5:64, f6:128, f7:128)
        self.lat4 = nn.Conv2d(64, dec, 1)
        self.lat5 = nn.Conv2d(64, dec, 1)
        self.lat6 = nn.Conv2d(128, dec, 1)
        self.lat7 = nn.Conv2d(128, dec, 1)

        # Projections for frozen message-attribution maps (C + 3 channels each)
        self.attr4 = nn.Sequential(nn.Conv2d(64 + 3, dec, 1), nn.SiLU(inplace=True))
        self.attr5 = nn.Sequential(nn.Conv2d(64 + 3, dec, 1), nn.SiLU(inplace=True))
        self.attr6 = nn.Sequential(nn.Conv2d(128 + 3, dec, 1), nn.SiLU(inplace=True))
        self.attr7 = nn.Sequential(nn.Conv2d(128 + 3, dec, 1), nn.SiLU(inplace=True))

        # Global known-message summary (100 signed logits + mean + bit_acc, both STN and no-STN = 204 ch)
        self.msg_glob = nn.Sequential(
            nn.Linear(204, dec),
            nn.SiLU(inplace=True),
            nn.Linear(dec, dec),
            nn.SiLU(inplace=True),
        )

        self.top = block(3 * dec, dec)
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

    def _frozen_message_signals(self, x01_400: torch.Tensor, bits: torch.Tensor):
        """Returns (glob_vec [B, 204], (a4, a5, a6, a7)) with zero gradients into model params."""
        sign = 2.0 * bits.float() - 1.0  # [B, 100]
        with torch.autocast("cuda", enabled=False):
            x_det = x01_400.detach().float()
            # 1) Global STN readout
            with torch.no_grad():
                logits_stn = self.frozen(x_det, use_stn=True)
                s_stn = logits_stn * sign
                v_stn = torch.cat([
                    (s_stn / 4.0).clamp(-4.0, 4.0),
                    s_stn.mean(1, keepdim=True) / 4.0,
                    (s_stn > 0).float().mean(1, keepdim=True) * 2.0 - 1.0,
                ], dim=1)

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
                glob_vec = torch.cat([v_stn, v_no], dim=1)  # [B, 204]
                a4 = _norm_attr(g4.detach(), f4.detach())
                a5 = _norm_attr(g5.detach(), f5.detach())
                a6 = _norm_attr(g6.detach(), f6.detach())
                a7 = _norm_attr(g7.detach(), f7.detach())
        return glob_vec, (a4, a5, a6, a7)

    def forward(self, x: torch.Tensor, bits: torch.Tensor):
        """x: [B, 3, 512, 512] in [-1, 1]; bits: [B, 100] in {0, 1}."""
        with torch.autocast("cuda", enabled=False):
            xf = x.float()
            x01 = (xf + 1.0) * 0.5
            x400 = F.interpolate(x01, size=(400, 400), mode="bicubic", align_corners=False, antialias=True).clamp(0.0, 1.0)
            hp_in = self.hp_front(xf)
        hp128 = self.hp_stem(hp_in)  # [B, dec, 128, 128]

        glob_vec, (a4, a5, a6, a7) = self._frozen_message_signals(x400, bits)
        ef4, ef5, ef6, ef7 = self._encode_trainable(x400)

        mg = self.msg_glob(glob_vec)[:, :, None, None].expand(-1, -1, ef7.shape[-2], ef7.shape[-1])
        y = self.top(torch.cat([self.lat7(ef7), self.attr7(a7), mg], dim=1))
        g = self.glob(y.mean(dim=(2, 3)))
        y = y + g[:, :, None, None]

        y = self.smooth6(F.interpolate(y, size=ef6.shape[-2:], mode="nearest") + self.lat6(ef6) + self.attr6(a6))
        y = self.smooth5(F.interpolate(y, size=ef5.shape[-2:], mode="nearest") + self.lat5(ef5) + self.attr5(a5))
        y = self.smooth4(F.interpolate(y, size=ef4.shape[-2:], mode="nearest") + self.lat4(ef4) + self.attr4(a4))

        y128 = self.fuse_hp(torch.cat([F.interpolate(y, size=(128, 128), mode="bilinear", align_corners=False), hp128], dim=1))
        logits = self.head(y128)[:, 0]
        return logits, self.cls(g)[:, 0]
