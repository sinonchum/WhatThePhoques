"""V5.2 Dual-Stream Architecture: Watermark Physics Stream + ImageNet-22K ConvNeXt/Swin Visual Stream.

Supports:
  - vis_mode="rgb" (StegaStamp): 3ch normalized RGB into ConvNeXt-Small/Base or Swin-Base-384
  - vis_mode="hp"  (PixelSeal):  6ch [norm(RGB), 8*(x - boxblur5(x))] into ConvNeXt (zero-init extra 3ch)
  - vis_mode="rm"  (TrustMark):  6ch [norm(RGB), 10*(x - trustmark_rm_Q(x))] into ConvNeXt (zero-init extra 3ch)
  - Auto-handles Swin NHWC feature maps -> NCHW!
  - Full bf16 stability (no fp16 overflow)!
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class _CBR(nn.Sequential):
    def __init__(self, ci: int, co: int, k: int = 3):
        super().__init__(
            nn.Conv2d(ci, co, k, padding=k // 2, bias=False),
            nn.GroupNorm(min(8, co), co),
            nn.SiLU(inplace=True),
        )


def _expand_first_conv_to_6ch(vis: nn.Module):
    """Expand first 3ch Conv2d stem to 6ch: copy pretrained RGB weights, small-init extra 3ch."""
    for parent_name, parent in vis.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Conv2d) and child.in_channels == 3:
                new_conv = nn.Conv2d(
                    6, child.out_channels, child.kernel_size,
                    stride=child.stride, padding=child.padding,
                    dilation=child.dilation, groups=child.groups,
                    bias=(child.bias is not None),
                )
                with torch.no_grad():
                    new_conv.weight[:, :3].copy_(child.weight)
                    new_conv.weight[:, 3:].copy_(child.weight * 0.25)
                    if child.bias is not None:
                        new_conv.bias.copy_(child.bias)
                setattr(parent, child_name, new_conv)
                return
    raise RuntimeError("Could not find 3ch stem Conv2d in visual backbone")


class DualConvNeXtV5(nn.Module):
    def __init__(
        self,
        phys_net: nn.Module,
        vis_backbone: str = "convnext_small.fb_in22k_ft_in1k_384",
        vis_dim: int = 128,
        enc_res: int = 512,
        vis_mode: str = "rgb",
    ):
        super().__init__()
        self.phys = phys_net
        self.vis_mode = vis_mode
        self.is_swin = "swin" in vis_backbone.lower()
        self.is_eva = "eva02" in vis_backbone.lower()
        self.enc_res = 384 if self.is_swin else (448 if self.is_eva else enc_res)

        kwargs = dict(pretrained=True, features_only=True)
        if self.is_swin or self.is_eva:
            kwargs["img_size"] = self.enc_res
        self.vis = timm.create_model(vis_backbone, **kwargs)
        if vis_mode in ("hp", "rm"):
            _expand_first_conv_to_6ch(self.vis)

        chs = list(self.vis.feature_info.channels())
        if len(chs) == 3:
            chs = [chs[0], chs[1], chs[2], chs[2]]

        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.lat3 = nn.Conv2d(chs[3] + 2, vis_dim, 1)
        self.lat2 = nn.Conv2d(chs[2] + 2, vis_dim, 1)
        self.lat1 = nn.Conv2d(chs[1] + 2, vis_dim, 1)
        self.lat0 = nn.Conv2d(chs[0] + 2, vis_dim, 1)

        self.sm2 = _CBR(vis_dim, vis_dim, 3)
        self.sm1 = _CBR(vis_dim, vis_dim, 3)
        self.sm0 = _CBR(vis_dim, vis_dim, 3)

        self.fuse = nn.Sequential(
            _CBR(vis_dim + 3, vis_dim, 3),
            _CBR(vis_dim, 64, 3),
        )
        self.delta_head = nn.Conv2d(64, 1, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

        self.vis_cls = nn.Linear(chs[3] + 1, 1)
        nn.init.zeros_(self.vis_cls.weight)
        nn.init.zeros_(self.vis_cls.bias)

        self.last_phys_logits: torch.Tensor | None = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.phys.train(mode)
        return self

    def backbone_parameters(self):
        return list(self.phys.backbone_parameters()) + list(self.vis.parameters())

    def head_parameters(self):
        vis_ids = {id(p) for p in self.vis.parameters()}
        phys_ids = {id(p) for p in self.phys.parameters()}
        own_head = [p for p in self.parameters() if id(p) not in vis_ids and id(p) not in phys_ids]
        return list(self.phys.head_parameters()) + own_head

    def _to_nchw(self, feats):
        out = []
        chs = self.vis.feature_info.channels()
        for f, c in zip(feats, chs):
            if f.ndim == 4 and f.shape[1] != c and f.shape[-1] == c:
                f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out

    def _build_vis_input(self, x01: torch.Tensor) -> torch.Tensor:
        if x01.shape[-1] != self.enc_res:
            xv = F.interpolate(x01, (self.enc_res, self.enc_res), mode="bilinear", align_corners=False)
        else:
            xv = x01
        rgb_norm = (xv - self.im_mean) / self.im_std
        if self.vis_mode == "rgb":
            return rgb_norm
        if self.vis_mode == "rm" and hasattr(self.phys, "_rm_residual"):
            with torch.no_grad():
                # Reuse Adobe trustmark_rm_Q remover residual!
                rm_res, _ = self.phys._rm_residual(xv.float())
                rm_amp = (rm_res * 10.0).clamp(-3.0, 3.0).to(rgb_norm.dtype)
            return torch.cat([rgb_norm, rm_amp], dim=1)
        # Default "hp" (or fallback): 5x5 local high-pass watermark residual
        blur = F.avg_pool2d(xv, kernel_size=5, stride=1, padding=2)
        hp = ((xv - blur) * 8.0).clamp(-3.0, 3.0).to(rgb_norm.dtype)
        return torch.cat([rgb_norm, hp], dim=1)

    def forward(self, x01: torch.Tensor, msg: torch.Tensor):
        seg_p, cls_p = self.phys(x01, msg)
        self.last_phys_logits = seg_p

        vis_in = self._build_vis_input(x01)
        raw_f = self._to_nchw(self.vis(vis_in))
        if len(raw_f) == 3:
            f0 = F.interpolate(raw_f[0], scale_factor=2, mode="bilinear", align_corners=False)
            f1, f2 = raw_f[1], raw_f[2]
            f3 = F.avg_pool2d(raw_f[2], 2)
        else:
            f0, f1, f2, f3 = raw_f

        sp_lg = seg_p.detach().clamp(-6.0, 6.0).unsqueeze(1) / 3.0
        sp_pr = torch.sigmoid(seg_p.detach()).unsqueeze(1)
        pg = torch.cat([sp_lg, sp_pr], dim=1)

        p3 = self.lat3(torch.cat([f3, F.interpolate(pg, f3.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        p2 = self.sm2(F.interpolate(p3, f2.shape[-2:], mode="nearest") + self.lat2(torch.cat([f2, F.interpolate(pg, f2.shape[-2:], mode="bilinear", align_corners=False)], dim=1)))
        p1 = self.sm1(F.interpolate(p2, f1.shape[-2:], mode="nearest") + self.lat1(torch.cat([f1, F.interpolate(pg, f1.shape[-2:], mode="bilinear", align_corners=False)], dim=1)))
        p0 = self.sm0(F.interpolate(p1, f0.shape[-2:], mode="nearest") + self.lat0(torch.cat([f0, F.interpolate(pg, f0.shape[-2:], mode="bilinear", align_corners=False)], dim=1)))

        if p0.shape[-2:] != (128, 128):
            p0 = F.interpolate(p0, (128, 128), mode="bilinear", align_corners=False)

        area_map = sp_pr.mean(dim=(2, 3), keepdim=True).expand(-1, -1, 128, 128)
        fused = self.fuse(torch.cat([p0, seg_p.clamp(-8.0, 8.0).unsqueeze(1) / 3.0, torch.sigmoid(seg_p).unsqueeze(1), area_map], dim=1))
        seg_out = seg_p + self.delta_head(fused)[:, 0]

        g_vis = F.adaptive_avg_pool2d(f3, 1).flatten(1)
        cls_out = cls_p + self.vis_cls(torch.cat([g_vis, cls_p.detach().unsqueeze(1)], dim=1))[:, 0]
        return seg_out, cls_out
