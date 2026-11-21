"""Clean K-fold OOF trainer for StegaStamp V2 (StegaExtSegNetV2) & V1 (StegaExtSegNet)
on clean synth3/stegastamp (train-only non-holdout covers + lowres 256 embed upsampled delta).
Reuses battle-tested train_v3_attr.py pipeline & assemble_kfold.py format 100% identically.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import train_v3_attr
from stega_extnet import StegaExtSegNet
from stega_extnet_v2 import StegaExtSegNetV2

ARCH = os.environ.get("STEGA_ARCH", "v2")
ONNX_PATH = os.environ.get("STEGA_ONNX", "/p/scratch/training2666/qin7/task1/models/stegastamp/stega_stamp_logits.onnx")


class _StegaWrapper(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x, bits):
        return self.inner(x, bits)

    def backbone_parameters(self):
        if hasattr(self.inner, "backbone_parameters"):
            return list(self.inner.backbone_parameters())
        return []

    def head_parameters(self):
        if hasattr(self.inner, "head_parameters"):
            return list(self.inner.head_parameters())
        return [p for p in self.inner.parameters() if p.requires_grad]


def _dummy_wm(method, models_dir, device):
    return None


def _make_stega(method, wm, dec=128, enc_res=512, attr_res=256):
    assert method == "stegastamp"
    if ARCH == "v1":
        inner = StegaExtSegNet(onnx_path=ONNX_PATH, dec=dec)
    else:
        inner = StegaExtSegNetV2(onnx_path=ONNX_PATH, dec=dec)
    return _StegaWrapper(inner)


train_v3_attr.load_watermark = _dummy_wm
train_v3_attr.ExtAttrSegNet = _make_stega

if __name__ == "__main__":
    train_v3_attr.main()
