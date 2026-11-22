"""Train ExtPyrSegNetV6 (Multi-Scale 45-Crop Known-Message Pyramid + Physics Stream + Deep Sym-mIoU)."""
import torch
import torch.nn.functional as F
import train_v3_attr as tv3
from extnet_v6_pyr import ExtPyrSegNetV6
from extnet_v4_tm_rm_fast import ExtAttrSegNetV4_RM_Fast
from extnet_v3_attr import ExtAttrSegNet as BaseExtAttrSegNet

ACTIVE_MODEL = [None]

def _build_v6(method, wm, dec=160, enc_res=512, attr_res=256):
    if method == "trustmark":
        phys = ExtAttrSegNetV4_RM_Fast(wm, dec=dec, enc_res=enc_res)
    else:
        phys = BaseExtAttrSegNet(method, wm, dec=dec, enc_res=enc_res, attr_res=attr_res)
    m = ExtPyrSegNetV6(phys, wm, grids=((2, 256), (4, 160), (5, 128)))
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

def _v6_deep_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = _sym_miou_single(logits, target)
    m = ACTIVE_MODEL[0]
    if m is not None:
        if m.last_phys_logits is not None and m.last_phys_logits.requires_grad:
            loss = loss + 0.30 * _sym_miou_single(m.last_phys_logits.float(), target)
        if m.last_pyr_logits is not None and m.last_pyr_logits.requires_grad:
            loss = loss + 0.30 * _sym_miou_single(m.last_pyr_logits.float(), target)
    return loss

tv3.ExtAttrSegNet = _build_v6
tv3.seg_loss = _v6_deep_loss

if __name__ == "__main__":
    tv3.main()
