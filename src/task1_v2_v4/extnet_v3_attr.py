"""ExtAttrSegNet (V3): Official 512-res extractor + Multi-scale known-message `grad * act` attribution
+ 6-view (global + 5-crop) 16x16 message map + 512->128 HighPass stem.

Brings the `grad * act` known-message spatial attribution breakthrough that pushed StegaStamp from
0.568 -> 0.7185 directly to TrustMark (ResNet-50) and PixelSeal (ConvNeXt-V2):
- Differentiates the frozen official extractor's known-message alignment score (logits * sign).mean()
  w.r.t. intermediate feature maps (stages 1, 2, 3 at 32x32, 16x16, 8x8 on 256x256 input, and/or
  64x64, 32x32, 16x16 on 512x512 input!) to pinpoint exact pixels carrying the known message bits.
- Combines this with the 6-view 5-crop 16x16 message map, trainable 512-res backbone copy, and
  full-resolution 512->128 HighPass residual stem.
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from extnet_v2 import block, HighPass


def _norm_attr(g: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Pack normalized channel-wise grad*act + 3 spatial summary maps (pos, abs, grad_norm)."""
    ga = g * f
    scale = ga.abs().mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
    ga_n = (ga / scale).clamp(-6.0, 6.0)
    pos = ga_n.mean(dim=1, keepdim=True)
    absm = ga_n.abs().mean(dim=1, keepdim=True)
    gn = g.norm(dim=1, keepdim=True)
    gn = (gn / gn.mean(dim=(2, 3), keepdim=True).clamp(min=1e-6)).clamp(0.0, 6.0)
    return torch.cat([ga_n, pos, absm, gn], dim=1)


class ExtAttrSegNet(nn.Module):
    def __init__(self, method: str, wm, dec: int = 160, enc_res: int = 512, attr_res: int = 256):
        super().__init__()
        self.method = method
        self.enc_res = enc_res
        self.attr_res = attr_res
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

        # Global (8x8) + 5-crop local (16x16) message maps
        self.msg_glob = nn.Sequential(nn.Conv2d(self.nbits + 2, dec, 1), nn.SiLU(inplace=True))
        self.msg_loc = nn.Sequential(nn.Conv2d(self.nbits + 2, dec, 1), nn.SiLU(inplace=True))
        self.msg_mid = nn.Sequential(nn.Conv2d(self.nbits + 2, dec, 1), nn.SiLU(inplace=True))

        # Multi-scale `grad * act` known-message attribution projections for stages 1, 2, 3
        self.attr1 = nn.Sequential(nn.Conv2d(chs[1] + 3, dec, 1), nn.SiLU(inplace=True))
        self.attr2 = nn.Sequential(nn.Conv2d(chs[2] + 3, dec, 1), nn.SiLU(inplace=True))
        self.attr3 = nn.Sequential(nn.Conv2d(chs[3] + 3, dec, 1), nn.SiLU(inplace=True))

        self.lat = nn.ModuleList(nn.Conv2d(c, dec, 1) for c in chs)
        self.top = block(4 * dec, dec)
        self.glob = nn.Sequential(nn.Linear(dec, dec), nn.SiLU(inplace=True))
        self.smooth = nn.ModuleList(block(dec, dec) for _ in range(3))

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
        B = x01_512.shape[0]
        g256 = F.interpolate(x01_512.float(), size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
        c00 = x01_512[:, :, 0:256, 0:256]
        c01 = x01_512[:, :, 0:256, 256:512]
        c10 = x01_512[:, :, 256:512, 0:256]
        c11 = x01_512[:, :, 256:512, 256:512]
        cc = x01_512[:, :, 128:384, 128:384]

        all_x = torch.cat([g256, c00, c01, c10, c11, cc], dim=0)
        all_b = bits.repeat(6, 1)
        all_m = self._single_message_map(all_x, all_b)
        mg, m00, m01, m10, m11, mc = all_m.chunk(6, dim=0)

        C = mg.shape[1]
        canvas = torch.zeros((B, C, 16, 16), device=mg.device, dtype=mg.dtype)
        weight = torch.zeros((1, 1, 16, 16), device=mg.device, dtype=mg.dtype)
        canvas[:, :, 0:8, 0:8] += m00;   weight[:, :, 0:8, 0:8] += 1.0
        canvas[:, :, 0:8, 8:16] += m01;  weight[:, :, 0:8, 8:16] += 1.0
        canvas[:, :, 8:16, 0:8] += m10;  weight[:, :, 8:16, 0:8] += 1.0
        canvas[:, :, 8:16, 8:16] += m11; weight[:, :, 8:16, 8:16] += 1.0
        canvas[:, :, 4:12, 4:12] += mc;  weight[:, :, 4:12, 4:12] += 1.0
        return mg, canvas / weight

    def _frozen_grad_attr(self, x01_512: torch.Tensor, bits: torch.Tensor):
        """Compute multi-scale grad*act known-message attribution from frozen official extractor."""
        sign = 2.0 * bits.float() - 1.0  # [B, nbits]
        with torch.autocast("cuda", enabled=False):
            if self.attr_res == 512:
                x_in = x01_512.detach().float()
            else:
                x_in = F.interpolate(
                    x01_512.detach().float(), size=(self.attr_res, self.attr_res),
                    mode="bilinear", align_corners=False, antialias=True
                )
            with torch.enable_grad():
                x_req = x_in.requires_grad_(True)
                if self.method == "trustmark":
                    n = self.frozen
                    h = n.maxpool(n.relu(n.bn1(n.conv1(x_req * 2.0 - 1.0))))
                    s0 = n.layer1(h)
                    s1 = n.layer2(s0)
                    s2 = n.layer3(s1)
                    s3 = n.layer4(s2)
                    logits = n.fc(s3.mean(dim=(2, 3)))  # [B, 100]
                else:
                    d = self.frozen
                    h = d.preprocess(x_req)
                    feats = []
                    for down, stage in zip(d.convnext.downsample_layers, d.convnext.stages):
                        h = stage(down(h))
                        feats.append(h)
                    s1, s2, s3 = feats[1], feats[2], feats[3]
                    emb = d.pixel_decoder.output_upscaling(s3)
                    lin = d.pixel_decoder.linear
                    per = (torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None])[:, 1:]
                    logits = per.mean(dim=(2, 3))  # [B, 256]
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
        mg, mloc = self.multi_crop_message_maps(x01_512, bits)
        a1, a2, a3 = self._frozen_grad_attr(x01_512, bits)

        xe = x01_512 if self.enc_res == 512 else F.interpolate(
            x01_512, size=(self.enc_res, self.enc_res), mode="bilinear", align_corners=False, antialias=True
        )
        f = self.encode(xe)  # at 512: [128, 64, 32, 16]

        h3, w3 = f[3].shape[-2:]
        mg_p = F.interpolate(self.msg_glob(mg), size=(h3, w3), mode="bilinear", align_corners=False)
        ml_p = F.interpolate(self.msg_loc(mloc), size=(h3, w3), mode="bilinear", align_corners=False)
        a3_p = F.interpolate(self.attr3(a3), size=(h3, w3), mode="bilinear", align_corners=False)

        y = self.top(torch.cat([self.lat[3](f[3]), mg_p, ml_p, a3_p], dim=1))
        g = self.glob(y.mean(dim=(2, 3)))
        y = y + g[:, :, None, None]

        ml_mid = F.interpolate(self.msg_mid(mloc), size=f[2].shape[-2:], mode="bilinear", align_corners=False)
        a2_p = F.interpolate(self.attr2(a2), size=f[2].shape[-2:], mode="bilinear", align_corners=False)
        y = self.smooth[2](F.interpolate(y, size=f[2].shape[-2:], mode="nearest") + self.lat[2](f[2]) + ml_mid + a2_p)

        a1_p = F.interpolate(self.attr1(a1), size=f[1].shape[-2:], mode="bilinear", align_corners=False)
        y = self.smooth[1](F.interpolate(y, size=f[1].shape[-2:], mode="nearest") + self.lat[1](f[1]) + a1_p)
        y = self.smooth[0](F.interpolate(y, size=f[0].shape[-2:], mode="nearest") + self.lat[0](f[0]))

        y128 = F.interpolate(y, size=(128, 128), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", enabled=False):
            hp_in = self.hp_front(xf)
        hp128 = self.hp_stem(hp_in)
        y128 = self.fuse_hp(torch.cat([y128, hp128], dim=1))

        logits = self.head(y128)[:, 0]
        return logits, self.cls(g)[:, 0]
