"""V7 PixelSeal Native Pretrained Pixel-Decoder Network (ExtPxDecSegNetV7).

Extracts Meta PixelSeal's OWN pretrained spatial 257-channel linear decoder
BEFORE global spatial averaging:
    up = wm.det.pixel_decoder.output_upscaling(wm.det.convnext(wm.det.preprocess(x)))
    sp = wm.det.pixel_decoder.linear(up.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, 257, h, w]
Computes exact spatial (det_logit, known_msg_margin, known_msg_bit_acc) across
3 overlapping spatial scales (1x512, 2x2@320, 3x3@256 = only 14 passes, ~8ms/batch on A100 bf16!)
plus per-image Z-scored contrast maps -> 12 razor-sharp pretrained channels fused with ExtAttrSegNet.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CBR(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, d: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, padding=d * (k // 2), dilation=d, bias=False),
            nn.GroupNorm(min(8, c_out), c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _tile_coords(grid: int, crop: int, full: int = 512) -> list[tuple[int, int]]:
    if grid == 1:
        return [(0, 0)]
    xs = [int(round(i * (full - crop) / (grid - 1))) for i in range(grid)]
    return [(y, x) for y in xs for x in xs]


class ExtPxDecSegNetV7(nn.Module):
    def __init__(self, phys_net: nn.Module, wm):
        super().__init__()
        self.phys = phys_net
        self.wm = wm

        self.tiles = [
            (1, 512, _tile_coords(1, 512, 512)),
            (2, 320, _tile_coords(2, 320, 512)),
            (3, 256, _tile_coords(3, 256, 512)),
        ]
        for idx, (g, crop, _) in enumerate(self.tiles):
            c128 = crop // 4
            w1d = torch.hann_window(c128 + 2, periodic=False)[1:-1].clamp(min=0.15)
            w2d = (w1d[:, None] * w1d[None, :]).view(1, 1, c128, c128)
            self.register_buffer(f"win_{idx}", w2d, persistent=False)

        # 12 pretrained pixel-decoder channels -> dedicated multi-scale encoder
        self.pd_enc = nn.Sequential(
            _CBR(12, 64, 3, d=1),
            _CBR(64, 64, 3, d=2),
            _CBR(64, 64, 3, d=1),
        )
        self.pd_aux_head = nn.Conv2d(64, 1, 1)

        # Joint fusion with physics stream (seg_p, sigmoid(seg_p))
        self.fuse = nn.Sequential(
            _CBR(64 + 2, 64, 3, d=1),
            _CBR(64, 48, 3, d=2),
            _CBR(48, 32, 3, d=1),
        )
        self.delta_head = nn.Conv2d(32, 1, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

        self.cls_delta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.cls_delta[-1].weight)
        nn.init.zeros_(self.cls_delta[-1].bias)

        self.last_phys_logits: torch.Tensor | None = None
        self.last_pd_logits: torch.Tensor | None = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.phys.train(mode)
        return self

    def backbone_parameters(self):
        return list(self.phys.backbone_parameters())

    def head_parameters(self):
        phys_ids = {id(p) for p in self.phys.parameters()}
        own = [p for p in self.parameters() if id(p) not in phys_ids]
        return list(self.phys.head_parameters()) + own

    @torch.no_grad()
    def _spatial_pxdec(self, crops01: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        """crops01: [N, 3, H, W], sign: [N, 256] -> [N, 3, h, w] (det, margin, acc)."""
        det = self.wm.det
        lat = det.convnext(det.preprocess(crops01))
        up = det.pixel_decoder.output_upscaling(lat)  # [N, C, h, w]
        sp = det.pixel_decoder.linear(up.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).float()  # [N, 257, h, w]
        det_lg = sp[:, 0:1].clamp(-6.0, 6.0) / 3.0
        sl = sp[:, 1:] * sign[:, :, None, None]  # [N, 256, h, w]
        margin = sl.mean(dim=1, keepdim=True).clamp(-4.0, 4.0) / 2.0
        acc = ((sl > 0).float().mean(dim=1, keepdim=True) - 0.5) * 4.0
        return torch.cat([det_lg, margin, acc], dim=1)

    @torch.no_grad()
    def _build_pxdec_maps(self, x01: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        B = x01.shape[0]
        dev = x01.device
        sign = (2.0 * msg.float() - 1.0)
        if x01.shape[-1] != 512:
            x512 = F.interpolate(x01, (512, 512), mode="bilinear", align_corners=False)
        else:
            x512 = x01

        scale_maps = []
        for idx, (g, crop, coords) in enumerate(self.tiles):
            win = getattr(self, f"win_{idx}").to(dev)
            c128 = crop // 4
            if g == 1:
                sp = self._spatial_pxdec(x512, sign)  # [B, 3, 16, 16]
                mp = F.interpolate(sp, (128, 128), mode="bilinear", align_corners=False)
                scale_maps.append(mp)
            else:
                tiles = torch.cat([x512[:, :, y:y + crop, x:x + crop] for (y, x) in coords], dim=0)
                s_rep = sign.repeat(len(coords), 1)
                sp = self._spatial_pxdec(tiles, s_rep)  # [B*T, 3, h, w]
                sp_up = F.interpolate(sp, (c128, c128), mode="bilinear", align_corners=False)
                sp_up = sp_up.view(len(coords), B, 3, c128, c128)
                acc_m = torch.zeros((B, 3, 128, 128), device=dev, dtype=torch.float32)
                acc_w = torch.zeros((1, 1, 128, 128), device=dev, dtype=torch.float32)
                for ti, (y, x) in enumerate(coords):
                    y4, x4 = y // 4, x // 4
                    acc_m[:, :, y4:y4 + c128, x4:x4 + c128] += sp_up[ti] * win
                    acc_w[:, :, y4:y4 + c128, x4:x4 + c128] += win
                scale_maps.append(acc_m / acc_w.clamp(min=1e-4))

        # 3 scales x 3 channels = 9 channels + 3 per-image Z-scored margin contrast maps = 12 channels
        z_maps = []
        for mp in scale_maps:
            m = mp[:, 1:2]  # known-message margin channel
            mu = m.mean(dim=(2, 3), keepdim=True)
            sd = m.std(dim=(2, 3), keepdim=True).clamp(min=0.05)
            z_maps.append(((m - mu) / sd).clamp(-4.0, 4.0))
        return torch.cat(scale_maps + z_maps, dim=1)  # [B, 12, 128, 128]

    def forward(self, x01: torch.Tensor, msg: torch.Tensor):
        seg_p, cls_p = self.phys(x01, msg)
        self.last_phys_logits = seg_p

        pd_in = self._build_pxdec_maps(x01, msg)
        pd_f = self.pd_enc(pd_in.to(seg_p.dtype))
        pd_lg = self.pd_aux_head(pd_f).squeeze(1)
        self.last_pd_logits = pd_lg

        sp_lg = seg_p.detach().clamp(-6.0, 6.0).unsqueeze(1) / 3.0
        sp_pr = torch.sigmoid(seg_p.detach()).unsqueeze(1)
        delta = self.delta_head(self.fuse(torch.cat([pd_f, sp_lg, sp_pr], dim=1))).squeeze(1)

        out_seg = seg_p + delta
        out_cls = cls_p + self.cls_delta(pd_f).squeeze(1)
        return out_seg, out_cls
