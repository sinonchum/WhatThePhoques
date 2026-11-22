"""ExtAttrSegNetV4_RM (TrustMark Specialist):
Combines ALL 5 physical/neural signals for TrustMark-Q localization:
1) Adobe's official TrustMark neural watermark remover (`tm.removal`, `trustmark_rm_Q.ckpt`) run on
   6 views (global 256x256 + 5 overlapping 256x256 crops of 512x512):
   - Full-res 512x512 removal residual map `res_loc` (3 RGB + L1 + L2) + upsampled `res_glob` (3 RGB + L1 + L2)
     fed directly into the 512->128 HighPass residual stem (measured 2.1-2.34x fg/bg SNR on small patches!).
   - Differential known-message map: `message_map(x) - message_map(clamp(x + res_rm))` isolating pixels where
     the official remover actually erased the KNOWN 100-bit message!
2) Multi-scale `grad * act` known-message spatial attribution (layer2 32x32, layer3 16x16, layer4 8x8).
3) 6-view (global 8x8 + 5-crop 16x16) frozen known-message maps.
4) Trainable full-resolution 512x512 copy of official TrustMark ResNet-50 backbone.
5) Full-resolution 512->128 HighPass + Remover Residual stem.
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from extnet_v2 import block, HighPass
from extnet_v3_attr import _norm_attr


class ExtAttrSegNetV4_RM(nn.Module):
    method = "trustmark"

    def __init__(self, wm, dec: int = 160, enc_res: int = 512, attr_res: int = 256):
        super().__init__()
        self.enc_res = enc_res
        self.attr_res = attr_res
        self.nbits = 100

        self.frozen = wm.net
        self.removal = wm.tm.removal
        assert self.removal is not None, "TrustMark removal network not loaded!"
        self.enc = copy.deepcopy(wm.net)
        self.enc.fc = nn.Identity()
        chs = [256, 512, 1024, 2048]

        for p in self.frozen.parameters():
            p.requires_grad_(False)
        for p in self.removal.parameters():
            p.requires_grad_(False)
        for p in self.enc.parameters():
            p.requires_grad_(True)

        # Global (8x8) + 5-crop local (16x16) message maps + differential remover message map (16x16)
        self.msg_glob = nn.Sequential(nn.Conv2d(self.nbits + 2, dec, 1), nn.SiLU(inplace=True))
        self.msg_loc = nn.Sequential(nn.Conv2d(self.nbits + 2, dec, 1), nn.SiLU(inplace=True))
        self.msg_diff = nn.Sequential(nn.Conv2d(self.nbits + 2, dec, 1), nn.SiLU(inplace=True))
        self.msg_mid = nn.Sequential(nn.Conv2d(2 * (self.nbits + 2), dec, 1), nn.SiLU(inplace=True))

        # Multi-scale `grad * act` known-message attribution projections
        self.attr1 = nn.Sequential(nn.Conv2d(chs[1] + 3, dec, 1), nn.SiLU(inplace=True))
        self.attr2 = nn.Sequential(nn.Conv2d(chs[2] + 3, dec, 1), nn.SiLU(inplace=True))
        self.attr3 = nn.Sequential(nn.Conv2d(chs[3] + 3, dec, 1), nn.SiLU(inplace=True))

        self.lat = nn.ModuleList(nn.Conv2d(c, dec, 1) for c in chs)
        self.top = block(5 * dec, dec)
        self.glob = nn.Sequential(nn.Linear(dec, dec), nn.SiLU(inplace=True))
        self.smooth = nn.ModuleList(block(dec, dec) for _ in range(3))

        # 512->128 stem taking HighPass(12) + local remover residual(5) + global remover residual(5) = 22 ch
        self.hp_front = HighPass(scale=40.0, clip=3.0)
        self.hp_stem = nn.Sequential(
            block(22, 48, stride=1),   # 512
            block(48, 80, stride=2),   # 256
            block(80, dec, stride=2),  # 128
        )
        self.fuse_hp = block(2 * dec, dec)
        self.head = nn.Sequential(block(dec, 64), nn.Conv2d(64, 1, 1))
        self.cls = nn.Linear(dec, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen.eval()
        self.removal.eval()
        for m in self.enc.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def backbone_parameters(self):
        return self.enc.parameters()

    def head_parameters(self):
        skip = (
            {id(p) for p in self.enc.parameters()}
            | {id(p) for p in self.frozen.parameters()}
            | {id(p) for p in self.removal.parameters()}
        )
        return [p for p in self.parameters() if id(p) not in skip]

    def encode(self, x01: torch.Tensor):
        n = self.enc
        x = n.maxpool(n.relu(n.bn1(n.conv1(x01 * 2.0 - 1.0))))
        out = []
        for k in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(n, k)(x)
            out.append(x)
        return out

    @torch.no_grad()
    def _single_message_map(self, x01_256: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=False):
            x01_256 = x01_256.float()
            n = self.frozen
            x = n.maxpool(n.relu(n.bn1(n.conv1(x01_256 * 2.0 - 1.0))))
            x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
            per = torch.einsum("bchw,oc->bohw", x, n.fc.weight) + n.fc.bias[None, :, None, None]
            s = per * (2.0 * bits.float() - 1.0)[:, :, None, None]
            return torch.cat([
                (s / 4.0).clamp(-4.0, 4.0),
                s.mean(1, keepdim=True) / 4.0,
                (s > 0).float().mean(1, keepdim=True) * 2.0 - 1.0,
            ], dim=1)

    @torch.no_grad()
    def _pack_res(self, r: torch.Tensor) -> torch.Tensor:
        """r: [B, 3, H, W] in [-2, 2] -> [B, 5, H, W] scaled signed RGB + L1 + L2."""
        rs = (r * 25.0).clamp(-4.0, 4.0)
        l1 = rs.abs().mean(dim=1, keepdim=True)
        l2 = (rs * rs).mean(dim=1, keepdim=True).sqrt()
        return torch.cat([rs, l1, l2], dim=1)

    @torch.no_grad()
    def multi_crop_msg_and_remover(self, x01_512: torch.Tensor, bits: torch.Tensor):
        """Run 6-view frozen decoder + 6-view Adobe remover + 6-view differential message map."""
        B = x01_512.shape[0]
        with torch.autocast("cuda", enabled=False):
            x01_512 = x01_512.float()
            g256 = F.interpolate(x01_512, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
            c00 = x01_512[:, :, 0:256, 0:256]
            c01 = x01_512[:, :, 0:256, 256:512]
            c10 = x01_512[:, :, 256:512, 0:256]
            c11 = x01_512[:, :, 256:512, 256:512]
            cc = x01_512[:, :, 128:384, 128:384]

            all_x01 = torch.cat([g256, c00, c01, c10, c11, cc], dim=0)  # [6B, 3, 256, 256]
            all_b = bits.repeat(6, 1)

            # 1) Official TrustMark remover on all 6 views (bf16 on A100 for 3-4x speedup)
            all_xm11 = all_x01 * 2.0 - 1.0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rem_m11 = self.removal(all_xm11).float().clamp(-1.0, 1.0)
            res_m11 = rem_m11 - all_xm11  # [6B, 3, 256, 256]
            rem_x01 = (rem_m11 + 1.0) * 0.5

            # Assemble 512x512 local removal residual + upsampled global removal residual
            rg, r00, r01, r10, r11, rc = res_m11.chunk(6, dim=0)
            rcanv = torch.zeros((B, 3, 512, 512), device=x01_512.device, dtype=torch.float32)
            rwgt = torch.zeros((1, 1, 512, 512), device=x01_512.device, dtype=torch.float32)
            rcanv[:, :, 0:256, 0:256] += r00;     rwgt[:, :, 0:256, 0:256] += 1.0
            rcanv[:, :, 0:256, 256:512] += r01;   rwgt[:, :, 0:256, 256:512] += 1.0
            rcanv[:, :, 256:512, 0:256] += r10;   rwgt[:, :, 256:512, 0:256] += 1.0
            rcanv[:, :, 256:512, 256:512] += r11; rwgt[:, :, 256:512, 256:512] += 1.0
            rcanv[:, :, 128:384, 128:384] += rc;  rwgt[:, :, 128:384, 128:384] += 1.0
            res_loc512 = self._pack_res(rcanv / rwgt)
            res_glob512 = self._pack_res(F.interpolate(rg, size=(512, 512), mode="bilinear", align_corners=False))
            rm_feat512 = torch.cat([res_loc512, res_glob512], dim=1)  # [B, 10, 512, 512]

            # 2) Frozen known-message map on original AND removed images -> differential map!
            all_m_orig = self._single_message_map(all_x01, all_b)  # [6B, 102, 8, 8]
            all_m_rem = self._single_message_map(rem_x01, all_b)   # [6B, 102, 8, 8]
            all_m_diff = (all_m_orig - all_m_rem).clamp(-4.0, 4.0)

            def stitch16(tensor6b):
                mg, m00, m01, m10, m11, mc = tensor6b.chunk(6, dim=0)
                C = mg.shape[1]
                canv = torch.zeros((B, C, 16, 16), device=mg.device, dtype=mg.dtype)
                w = torch.zeros((1, 1, 16, 16), device=mg.device, dtype=mg.dtype)
                canv[:, :, 0:8, 0:8] += m00;   w[:, :, 0:8, 0:8] += 1.0
                canv[:, :, 0:8, 8:16] += m01;  w[:, :, 0:8, 8:16] += 1.0
                canv[:, :, 8:16, 0:8] += m10;  w[:, :, 8:16, 0:8] += 1.0
                canv[:, :, 8:16, 8:16] += m11; w[:, :, 8:16, 8:16] += 1.0
                canv[:, :, 4:12, 4:12] += mc;  w[:, :, 4:12, 4:12] += 1.0
                return mg, canv / w

            mg, mloc = stitch16(all_m_orig)
            _, mdiff = stitch16(all_m_diff)
        return mg, mloc, mdiff, rm_feat512

    def _frozen_grad_attr(self, x01_512: torch.Tensor, bits: torch.Tensor):
        sign = 2.0 * bits.float() - 1.0
        with torch.autocast("cuda", enabled=False):
            x_in = F.interpolate(
                x01_512.detach().float(), size=(self.attr_res, self.attr_res),
                mode="bilinear", align_corners=False, antialias=True
            )
            with torch.enable_grad():
                x_req = x_in.requires_grad_(True)
                n = self.frozen
                h = n.maxpool(n.relu(n.bn1(n.conv1(x_req * 2.0 - 1.0))))
                s0 = n.layer1(h)
                s1 = n.layer2(s0)
                s2 = n.layer3(s1)
                s3 = n.layer4(s2)
                logits = n.fc(s3.mean(dim=(2, 3)))
                obj = (logits * sign).mean(dim=1).sum()
                g1, g2, g3 = torch.autograd.grad(obj, [s1, s2, s3], retain_graph=False, create_graph=False)
            with torch.no_grad():
                a1 = _norm_attr(g1.detach(), s1.detach())
                a2 = _norm_attr(g2.detach(), s2.detach())
                a3 = _norm_attr(g3.detach(), s3.detach())
        return a1, a2, a3

    def forward(self, x: torch.Tensor, bits: torch.Tensor):
        with torch.autocast("cuda", enabled=False):
            xf = x.float()
            x01_512 = (xf + 1.0) * 0.5
        mg, mloc, mdiff, rm_feat512 = self.multi_crop_msg_and_remover(x01_512, bits)
        a1, a2, a3 = self._frozen_grad_attr(x01_512, bits)

        xe = x01_512 if self.enc_res == 512 else F.interpolate(
            x01_512, size=(self.enc_res, self.enc_res), mode="bilinear", align_corners=False, antialias=True
        )
        f = self.encode(xe)

        h3, w3 = f[3].shape[-2:]
        mg_p = F.interpolate(self.msg_glob(mg), size=(h3, w3), mode="bilinear", align_corners=False)
        ml_p = F.interpolate(self.msg_loc(mloc), size=(h3, w3), mode="bilinear", align_corners=False)
        md_p = F.interpolate(self.msg_diff(mdiff), size=(h3, w3), mode="bilinear", align_corners=False)
        a3_p = F.interpolate(self.attr3(a3), size=(h3, w3), mode="bilinear", align_corners=False)

        y = self.top(torch.cat([self.lat[3](f[3]), mg_p, ml_p, md_p, a3_p], dim=1))
        g = self.glob(y.mean(dim=(2, 3)))
        y = y + g[:, :, None, None]

        ml_mid = F.interpolate(self.msg_mid(torch.cat([mloc, mdiff], dim=1)), size=f[2].shape[-2:], mode="bilinear", align_corners=False)
        a2_p = F.interpolate(self.attr2(a2), size=f[2].shape[-2:], mode="bilinear", align_corners=False)
        y = self.smooth[2](F.interpolate(y, size=f[2].shape[-2:], mode="nearest") + self.lat[2](f[2]) + ml_mid + a2_p)

        a1_p = F.interpolate(self.attr1(a1), size=f[1].shape[-2:], mode="bilinear", align_corners=False)
        y = self.smooth[1](F.interpolate(y, size=f[1].shape[-2:], mode="nearest") + self.lat[1](f[1]) + a1_p)
        y = self.smooth[0](F.interpolate(y, size=f[0].shape[-2:], mode="nearest") + self.lat[0](f[0]))

        y128 = F.interpolate(y, size=(128, 128), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", enabled=False):
            hp_in = torch.cat([self.hp_front(xf), rm_feat512], dim=1)  # [B, 22, 512, 512]
        hp128 = self.hp_stem(hp_in)
        y128 = self.fuse_hp(torch.cat([y128, hp128], dim=1))

        logits = self.head(y128)[:, 0]
        return logits, self.cls(g)[:, 0]
