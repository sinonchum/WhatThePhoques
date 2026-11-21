from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v3_attr
from extnet_v4_tm_rm_fast import ExtAttrSegNetV4_RM_Fast

train_v3_attr.ExtAttrSegNet = lambda method, wm, dec=160, enc_res=512, attr_res=256: ExtAttrSegNetV4_RM_Fast(wm, dec=dec, enc_res=enc_res, attr_res=attr_res)
if __name__ == "__main__":
    train_v3_attr.main()
