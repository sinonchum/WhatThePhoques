"""Train ExtPxDecSegNetV7 (Native PixelSeal Pretrained Spatial Decoder + Anti-Overfit + Deep Sym-mIoU)."""
import torch
import torch.nn.functional as F
import train_v3_attr as tv3
from extnet_v7_ps_pxdec import ExtPxDecSegNetV7
from extnet_v3_attr import ExtAttrSegNet as BaseExtAttrSegNet

ACTIVE_MODEL = [None]

def _build_v7(method, wm, dec=160, enc_res=512, attr_res=256):
    phys = BaseExtAttrSegNet(method, wm, dec=dec, enc_res=enc_res, attr_res=attr_res)
    m = ExtPxDecSegNetV7(phys, wm)
    ACTIVE_MODEL[0] = m
    return m

def _sym_miou_single(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    i1 = (p * t).sum(1); u1 = (p + t - p * t).sum(1)
    i0 = ((1.0 - p) * (1.0 - t)).sum(1); u0 = ((1.0 - p) + (1.0 - t) - (1.0 - p) * (1.0 - t)).sum(1)
    sym = 0.5 * ((i1 + 1.0) / (u1 + 1.0) + (i0 + 1.0) / (u0 + 1.0)).mean()
    return 0.40 * bce + 0.60 * (1.0 - sym)

def _v7_deep_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = _sym_miou_single(logits, target)
    m = ACTIVE_MODEL[0]
    if m is not None:
        if m.last_phys_logits is not None and m.last_phys_logits.requires_grad:
            loss = loss + 0.30 * _sym_miou_single(m.last_phys_logits.float(), target)
        if m.last_pd_logits is not None and m.last_pd_logits.requires_grad:
            loss = loss + 0.35 * _sym_miou_single(m.last_pd_logits.float(), target)
    return loss

tv3.ExtAttrSegNet = _build_v7
tv3.seg_loss = _v7_deep_loss

if __name__ == "__main__":
    tv3.main()
