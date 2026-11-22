"""Train DualConvNeXtV5 (V5.2: ConvNeXt/Swin + 6ch HighPass/Adobe-RM + Deep Sym-mIoU Loss)."""
import os
import torch
import torch.nn.functional as F
import train_v3_attr as tv3
from v5_dual_convnext import DualConvNeXtV5
from stega_extnet_v2 import StegaExtSegNetV2
from extnet_v4_tm_rm_fast import ExtAttrSegNetV4_RM_Fast
from extnet_v3_attr import ExtAttrSegNet as BaseExtAttrSegNet

ONNX_PATH = "/p/scratch/training2666/ding7/claude_task1/models/stegastamp/stega_stamp_feats.onnx"
VIS_BACKBONE = os.environ.get("V5_VIS_BACKBONE", "convnext_small.fb_in22k_ft_in1k_384")
VIS_MODE = os.environ.get("V5_VIS_MODE", "auto")
ACTIVE_MODEL = [None]

_orig_load_wm = tv3.load_watermark
def _load_wm(method, models_dir, device):
    if method == "stegastamp":
        return None
    return _orig_load_wm(method, models_dir, device)

def _build_v5(method, wm, dec=160, enc_res=512, attr_res=256):
    mode = VIS_MODE
    if method == "stegastamp":
        phys = StegaExtSegNetV2(ONNX_PATH, dec=dec)
        if mode == "auto": mode = "rgb"
    elif method == "trustmark":
        phys = ExtAttrSegNetV4_RM_Fast(wm, dec=dec, enc_res=enc_res)
        if mode == "auto": mode = "rm"
    else:
        phys = BaseExtAttrSegNet("pixelseal", wm, dec=dec, enc_res=enc_res, attr_res=attr_res)
        if mode == "auto": mode = "hp"
    m = DualConvNeXtV5(phys, vis_backbone=VIS_BACKBONE, vis_dim=128, enc_res=enc_res, vis_mode=mode)
    ACTIVE_MODEL[0] = m
    return m

def _sym_miou_single(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    i1 = (p * t).sum(1)
    u1 = (p + t - p * t).sum(1)
    i0 = ((1.0 - p) * (1.0 - t)).sum(1)
    u0 = ((1.0 - p) + (1.0 - t) - (1.0 - p) * (1.0 - t)).sum(1)
    sym_iou = 0.5 * ((i1 + 1.0) / (u1 + 1.0) + (i0 + 1.0) / (u0 + 1.0)).mean()
    return 0.45 * bce + 0.55 * (1.0 - sym_iou)

def _deep_sym_miou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = _sym_miou_single(logits, target)
    m = ACTIVE_MODEL[0]
    if m is not None and m.last_phys_logits is not None and m.last_phys_logits.requires_grad:
        loss = loss + 0.35 * _sym_miou_single(m.last_phys_logits.float(), target)
    return loss

tv3.load_watermark = _load_wm
tv3.ExtAttrSegNet = _build_v5
tv3.seg_loss = _deep_sym_miou_loss

if __name__ == "__main__":
    tv3.main()
