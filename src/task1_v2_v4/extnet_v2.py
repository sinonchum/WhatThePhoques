"""ExtSegNetV2: 512-res official extractor + 5-crop 16x16 known-message map + 512 HighPass stem.

Fixes two major bottlenecks in ExtSegNet:
1) ExtSegNet only ran the frozen message_map on a single 256x256 downsampled full image (8x8 grid),
   losing small watermark patches (<15% area). We add a batched 5-crop (4 quadrants + center)
   256x256 frozen pass assembled into a 16x16 high-res known-message map, injected at both
   stride-32 (16x16) and stride-16 (32x32).
2) Adds a full-res 512->128 HighPass residual stem fused before the segmentation head for crisp edges.
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ExtSegNetV2(nn.Module):
    def __init__(self, method: str, wm, dec: int = 160, enc_res: int = 512, use_hp: bool = True):
        super().__init__()
        self.method = method
        self.enc_res = enc_res
        self.use_hp = use_hp
        if method == "trustmark":
            self.frozen = wm.net
            self.enc = copy.deepcopy(wm.net)
            self.enc.fc = nn.Identity()
            chs, self.nbits = [256, 512, 1024, 2048], 100
        elif method == "pixelseal":
            self.frozen = wm.det
            self.enc = copy.deepcopy(wm.det.convnext)
            chs, self.nbits = [96, 192, 384, 768], 256
        else:
            raise ValueError(method)

        for p in self.frozen.parameters():
            p.requires_grad_(False)
        for p in self.enc.parameters():
            p.requires_grad_(True)

        # Global (8x8) + Local 5-crop (16x16) known-message projections
        self.msg_glob = nn.Sequential(nn.Conv2d(self.nbits + 2, dec, 1), nn.SiLU(inplace=True))
        self.msg_loc = nn.Sequential(nn.Conv2d(self.nbits + 2, dec, 1), nn.SiLU(inplace=True))
        self.msg_mid = nn.Sequential(nn.Conv2d(self.nbits + 2, dec, 1), nn.SiLU(inplace=True))

        self.lat = nn.ModuleList(nn.Conv2d(c, dec, 1) for c in chs)
        self.top = block(3 * dec, dec)
        self.glob = nn.Sequential(nn.Linear(dec, dec), nn.SiLU(inplace=True))
        self.smooth = nn.ModuleList(block(dec, dec) for _ in range(3))

        if use_hp:
            self.hp_front = HighPass(scale=40.0, clip=3.0)
            self.hp_stem = nn.Sequential(
                block(12, 32, stride=1),   # 512
                block(32, 64, stride=2),   # 256
                block(64, dec, stride=2),  # 128
            )
            self.fuse_hp = block(2 * dec, dec)

        self.head = nn.Sequential(block(dec, 64), nn.Conv2d(64, 1, 1))
        self.cls = nn.Linear(dec, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen.eval()
        for m in self.enc.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def backbone_parameters(self):
        return self.enc.parameters()

    def head_parameters(self):
        skip = {id(p) for p in self.enc.parameters()} | {id(p) for p in self.frozen.parameters()}
        return [p for p in self.parameters() if id(p) not in skip]

    def encode(self, x01: torch.Tensor):
        if self.method == "trustmark":
            n = self.enc
            x = n.maxpool(n.relu(n.bn1(n.conv1(x01 * 2.0 - 1.0))))
            out = []
            for k in ("layer1", "layer2", "layer3", "layer4"):
                x = getattr(n, k)(x)
                out.append(x)
            return out
        h = self.frozen.preprocess(x01)
        out = []
        for down, stage in zip(self.enc.downsample_layers, self.enc.stages):
            h = stage(down(h))
            out.append(h)
        return out

    @torch.no_grad()
    def _single_message_map(self, x01_256: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        """x01_256: [N, 3, 256, 256], bits: [N, nbits] -> [N, nbits+2, 8, 8]."""
        with torch.autocast("cuda", enabled=False):
            x01_256 = x01_256.float()
            if self.method == "trustmark":
                n = self.frozen
                x = n.maxpool(n.relu(n.bn1(n.conv1(x01_256 * 2.0 - 1.0))))
                x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
                per = torch.einsum("bchw,oc->bohw", x, n.fc.weight) + n.fc.bias[None, :, None, None]
            else:
                d = self.frozen
                lat = d.convnext(d.preprocess(x01_256))
                emb = d.pixel_decoder.output_upscaling(lat)
                lin = d.pixel_decoder.linear
                per = (torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None])[:, 1:]
            s = per * (2.0 * bits.float() - 1.0)[:, :, None, None]
            return torch.cat([
                (s / 4.0).clamp(-4.0, 4.0),
                s.mean(1, keepdim=True) / 4.0,
                (s > 0).float().mean(1, keepdim=True) * 2.0 - 1.0,
            ], dim=1)

    @torch.no_grad()
    def multi_crop_message_maps(self, x01_512: torch.Tensor, bits: torch.Tensor):
        """Batched 6-view pass: 1 global 256x256 + 4 quadrants + 1 center crop.
        Returns (m_glob [B, C, 8, 8], m_loc [B, C, 16, 16])."""
        B = x01_512.shape[0]
        g256 = F.interpolate(x01_512.float(), size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
        c00 = x01_512[:, :, 0:256, 0:256]
        c01 = x01_512[:, :, 0:256, 256:512]
        c10 = x01_512[:, :, 256:512, 0:256]
        c11 = x01_512[:, :, 256:512, 256:512]
        cc = x01_512[:, :, 128:384, 128:384]

        all_x = torch.cat([g256, c00, c01, c10, c11, cc], dim=0)
        all_b = bits.repeat(6, 1)
        all_m = self._single_message_map(all_x, all_b)  # [6B, C, 8, 8]
        mg, m00, m01, m10, m11, mc = all_m.chunk(6, dim=0)

        C = mg.shape[1]
        canvas = torch.zeros((B, C, 16, 16), device=mg.device, dtype=mg.dtype)
        weight = torch.zeros((1, 1, 16, 16), device=mg.device, dtype=mg.dtype)
        canvas[:, :, 0:8, 0:8] += m00;   weight[:, :, 0:8, 0:8] += 1.0
        canvas[:, :, 0:8, 8:16] += m01;  weight[:, :, 0:8, 8:16] += 1.0
        canvas[:, :, 8:16, 0:8] += m10;  weight[:, :, 8:16, 0:8] += 1.0
        canvas[:, :, 8:16, 8:16] += m11; weight[:, :, 8:16, 8:16] += 1.0
        canvas[:, :, 4:12, 4:12] += mc;  weight[:, :, 4:12, 4:12] += 1.0
        mloc = canvas / weight
        return mg, mloc

    def forward(self, x: torch.Tensor, bits: torch.Tensor):
        with torch.autocast("cuda", enabled=False):
            xf = x.float()
            x01_512 = (xf + 1.0) * 0.5
        mg, mloc = self.multi_crop_message_maps(x01_512, bits)

        xe = x01_512 if self.enc_res == 512 else F.interpolate(
            x01_512, size=(self.enc_res, self.enc_res), mode="bilinear", align_corners=False, antialias=True
        )
        f = self.encode(xe)  # at 512: [128, 64, 32, 16]

        h3, w3 = f[3].shape[-2:]
        mg_p = F.interpolate(self.msg_glob(mg), size=(h3, w3), mode="bilinear", align_corners=False)
        ml_p = F.interpolate(self.msg_loc(mloc), size=(h3, w3), mode="bilinear", align_corners=False)
        y = self.top(torch.cat([self.lat[3](f[3]), mg_p, ml_p], dim=1))
        g = self.glob(y.mean(dim=(2, 3)))
        y = y + g[:, :, None, None]

        # Inject local message evidence at stride-16 (f[2]) as well
        ml_mid = F.interpolate(self.msg_mid(mloc), size=f[2].shape[-2:], mode="bilinear", align_corners=False)
        y = self.smooth[2](F.interpolate(y, size=f[2].shape[-2:], mode="nearest") + self.lat[2](f[2]) + ml_mid)
        y = self.smooth[1](F.interpolate(y, size=f[1].shape[-2:], mode="nearest") + self.lat[1](f[1]))
        y = self.smooth[0](F.interpolate(y, size=f[0].shape[-2:], mode="nearest") + self.lat[0](f[0]))

        y128 = F.interpolate(y, size=(128, 128), mode="bilinear", align_corners=False)
        if self.use_hp:
            with torch.autocast("cuda", enabled=False):
                hp_in = self.hp_front(xf)
            hp128 = self.hp_stem(hp_in)
            y128 = self.fuse_hp(torch.cat([y128, hp128], dim=1))

        logits = self.head(y128)[:, 0]
        return logits, self.cls(g)[:, 0]
