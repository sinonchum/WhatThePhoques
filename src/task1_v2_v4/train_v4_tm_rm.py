"""Train ExtAttrSegNetV4_RM (TrustMark with Adobe official remover residual + differential message map)."""
from __future__ import annotations
import sys
from pathlib import Path

# Reuse train_v3_attr.py completely by swapping ExtAttrSegNet -> ExtAttrSegNetV4_RM!
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import train_v3_attr
from extnet_v4_tm_rm import ExtAttrSegNetV4_RM


def _make_v4(method, wm, dec=160, enc_res=512, attr_res=256):
    assert method == "trustmark"
    return ExtAttrSegNetV4_RM(wm, dec=dec, enc_res=enc_res, attr_res=attr_res)


train_v3_attr.ExtAttrSegNet = _make_v4

if __name__ == "__main__":
    train_v3_attr.main()
