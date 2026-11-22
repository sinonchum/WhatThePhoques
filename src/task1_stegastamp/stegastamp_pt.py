"""Exact PyTorch port of the StegaStamp ONNX decoder (STN + 7-conv decoder + dense head).

Loads weights directly from models/stegastamp/stega_stamp.onnx so the official
StegaStamp extractor runs natively on CUDA in PyTorch, exposes multi-scale
spatial feature maps, and supports exact gradients w.r.t. the known 100-bit message.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_same_s2(x: torch.Tensor) -> torch.Tensor:
    # Matches ONNX pads=[0, 0, 1, 1] (top=0, left=0, bottom=1, right=1)
    return F.pad(x, (0, 1, 0, 1))


class StegaStampDecoderPT(nn.Module):
    def __init__(self, onnx_path: str | Path):
        super().__init__()
        m = onnx.load(str(onnx_path))
        inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}

        def make_conv(w_name: str, b_name: str, stride: int, padding: int = 0) -> nn.Conv2d:
            w = inits[w_name]  # [Cout, Cin, kh, kw]
            b = inits[b_name]
            conv = nn.Conv2d(w.shape[1], w.shape[0], kernel_size=(w.shape[2], w.shape[3]), stride=stride, padding=padding)
            with torch.no_grad():
                conv.weight.copy_(torch.from_numpy(w.copy()))
                conv.bias.copy_(torch.from_numpy(b.copy()))
            return conv

        def make_linear(w_name: str, b_name: str) -> nn.Linear:
            w = inits[w_name]  # ONNX MatMul [in, out]
            b = inits[b_name]
            lin = nn.Linear(w.shape[0], w.shape[1])
            with torch.no_grad():
                lin.weight.copy_(torch.from_numpy(w.T.copy()))
                lin.bias.copy_(torch.from_numpy(b.copy()))
            return lin

        # STN
        self.stn_conv_1 = make_conv("gen_stn/stn_conv_1/Conv2D/ReadVariableOp__42", "gen_stn/stn_conv_1/BiasAdd/ReadVariableOp__43", stride=2)
        self.stn_conv_2 = make_conv("gen_stn/stn_conv_2/Conv2D/ReadVariableOp__44", "gen_stn/stn_conv_2/BiasAdd/ReadVariableOp__45", stride=2)
        self.stn_conv_3 = make_conv("gen_stn/stn_conv_3/Conv2D/ReadVariableOp__46", "gen_stn/stn_conv_3/BiasAdd/ReadVariableOp__47", stride=2)
        self.stn_fc_1 = make_linear("gen_stn/dense_1/MatMul/ReadVariableOp__48", "gen_stn/dense_1/BiasAdd/ReadVariableOp__49")
        self.stn_fc_2 = make_linear("gen_stn/W_fc1/read__50", "gen_stn/b_fc1/read__51")

        # Exact base coordinate grid from ONNX initializer [1, 3, 160000]
        self.register_buffer("stn_grid", torch.from_numpy(inits["gen_stn/ExpandDims__52"].astype(np.float32).copy()))

        # 7-layer decoder
        self.dec_conv_1 = make_conv("gen_decoder/dec_conv_1/Conv2D/ReadVariableOp__61", "gen_decoder/dec_conv_1/BiasAdd/ReadVariableOp__62", stride=2)
        self.dec_conv_2 = make_conv("gen_decoder/dec_conv_2/Conv2D/ReadVariableOp__63", "gen_decoder/dec_conv_2/BiasAdd/ReadVariableOp__64", stride=1, padding=1)
        self.dec_conv_3 = make_conv("gen_decoder/dec_conv_3/Conv2D/ReadVariableOp__65", "gen_decoder/dec_conv_3/BiasAdd/ReadVariableOp__66", stride=2)
        self.dec_conv_4 = make_conv("gen_decoder/dec_conv_4/Conv2D/ReadVariableOp__67", "gen_decoder/dec_conv_4/BiasAdd/ReadVariableOp__68", stride=1, padding=1)
        self.dec_conv_5 = make_conv("gen_decoder/dec_conv_5/Conv2D/ReadVariableOp__69", "gen_decoder/dec_conv_5/BiasAdd/ReadVariableOp__70", stride=2)
        self.dec_conv_6 = make_conv("gen_decoder/dec_conv_6/Conv2D/ReadVariableOp__71", "gen_decoder/dec_conv_6/BiasAdd/ReadVariableOp__72", stride=2)
        self.dec_conv_7 = make_conv("gen_decoder/dec_conv_7/Conv2D/ReadVariableOp__73", "gen_decoder/dec_conv_7/BiasAdd/ReadVariableOp__74", stride=2, padding=1)
        self.dec_dense = make_linear("gen_decoder/dec_dense/MatMul/ReadVariableOp__75", "gen_decoder/dec_dense/BiasAdd/ReadVariableOp__76")
        self.decoder_out = make_linear("gen_decoder/decoder_out/MatMul/ReadVariableOp__77", "gen_decoder/decoder_out/BiasAdd/ReadVariableOp__78")

    def stn_warp(self, x01_400: torch.Tensor) -> torch.Tensor:
        """x01_400: [B, 3, 400, 400] in [0, 1]. Returns warped [B, 3, 400, 400]."""
        h = F.relu(self.stn_conv_1(_pad_same_s2(x01_400 - 0.5)))
        h = F.relu(self.stn_conv_2(_pad_same_s2(h)))
        h = F.relu(self.stn_conv_3(_pad_same_s2(h)))
        # ONNX transposes NCHW -> NHWC before Flatten
        h = h.permute(0, 2, 3, 1).contiguous().flatten(1)
        h = F.relu(self.stn_fc_1(h))
        theta = self.stn_fc_2(h).view(-1, 2, 3)  # [B, 2, 3]

        # Match ONNX bilinear gather exactly:
        # coords = theta @ stn_grid -> [B, 2, 160000] -> [B, 2, 400, 400]
        B = x01_400.shape[0]
        coords = torch.bmm(theta, self.stn_grid.expand(B, -1, -1)).view(B, 2, 400, 400)
        # ONNX Cast__55 = 398.0 -> 0.5 * (c + 1.0) * 398.0 = (c + 1.0) * 199.0
        x_f = (coords[:, 0] + 1.0) * 199.0
        y_f = (coords[:, 1] + 1.0) * 199.0
        x0 = torch.floor(x_f).long().clamp(0, 399)
        x1 = (torch.floor(x_f).long() + 1).clamp(0, 399)
        y0 = torch.floor(y_f).long().clamp(0, 399)
        y1 = (torch.floor(y_f).long() + 1).clamp(0, 399)

        wx1 = (x_f - x0.float()).unsqueeze(1)
        wx0 = (x1.float() - x_f).unsqueeze(1)
        wy1 = (y_f - y0.float()).unsqueeze(1)
        wy0 = (y1.float() - y_f).unsqueeze(1)

        flat = x01_400.view(B, 3, 400 * 400)
        def gather(yy, xx):
            idx = (yy * 400 + xx).view(B, 1, 400 * 400).expand(B, 3, 400 * 400)
            return torch.gather(flat, 2, idx).view(B, 3, 400, 400)

        Ia = gather(y0, x0)
        Ib = gather(y1, x0)
        Ic = gather(y0, x1)
        Id = gather(y1, x1)
        return Ia * (wx0 * wy0) + Ib * (wx0 * wy1) + Ic * (wx1 * wy0) + Id * (wx1 * wy1)

    def forward_features(self, x01_400: torch.Tensor, use_stn: bool = True):
        """Returns (logits [B,100], [f2, f4, f5, f6, f7]) at 200, 100, 50, 25, 13."""
        w = self.stn_warp(x01_400) if use_stn else x01_400
        h = w - 0.5
        f1 = F.relu(self.dec_conv_1(_pad_same_s2(h)))      # 32 x 200 x 200
        f2 = F.relu(self.dec_conv_2(f1))                   # 32 x 200 x 200
        f3 = F.relu(self.dec_conv_3(_pad_same_s2(f2)))     # 64 x 100 x 100
        f4 = F.relu(self.dec_conv_4(f3))                   # 64 x 100 x 100
        f5 = F.relu(self.dec_conv_5(_pad_same_s2(f4)))     # 64 x 50 x 50
        f6 = F.relu(self.dec_conv_6(_pad_same_s2(f5)))     # 128 x 25 x 25
        f7 = F.relu(self.dec_conv_7(f6))                   # 128 x 13 x 13
        flat = f7.permute(0, 2, 3, 1).contiguous().flatten(1)
        dense = F.relu(self.dec_dense(flat))
        logits = self.decoder_out(dense)
        return logits, [f2, f4, f5, f6, f7]

    def forward(self, x01_400: torch.Tensor, use_stn: bool = True) -> torch.Tensor:
        logits, _ = self.forward_features(x01_400, use_stn=use_stn)
        return logits


if __name__ == "__main__":
    import onnxruntime as ort
    from PIL import Image
    from wmloc.data import LocalisationDataset
    from wmloc.utils import bit_accuracy

    onnx_path = Path("models/stegastamp/stega_stamp_feats.onnx")
    if not onnx_path.exists():
        onnx_path = Path("models/stegastamp/stega_stamp.onnx")
    pt = StegaStampDecoderPT(onnx_path).cuda().eval()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ds = LocalisationDataset("release", "train", methods={"stegastamp"})
    diffs = []
    for i in range(6):
        item = ds[i]
        img = np.asarray(Image.fromarray(item["image"]).resize((400, 400), Image.Resampling.BICUBIC), np.float32) / 255.0
        ort_logits = sess.run(["gen_decoder/decoder_out/BiasAdd:0"], {"image": img[None], "secret": np.zeros((1, 100), np.float32)})[0][0]
        x = torch.from_numpy(img).permute(2, 0, 1)[None].cuda()
        with torch.no_grad():
            pt_logits = pt(x, use_stn=True)[0].cpu().numpy()
            pt_nostn = pt(x, use_stn=False)[0].cpu().numpy()
        d = float(np.abs(ort_logits - pt_logits).max())
        diffs.append(d)
        acc_ort = bit_accuracy((ort_logits > 0).astype(int), item["message"])
        acc_pt = bit_accuracy((pt_logits > 0).astype(int), item["message"])
        acc_nostn = bit_accuracy((pt_nostn > 0).astype(int), item["message"])
        print(f"i={i} full={item['full_watermark']} max_diff={d:.6f} acc_ort={acc_ort:.2f} acc_pt={acc_pt:.2f} acc_nostn={acc_nostn:.2f}")
    print("WORST DIFF:", max(diffs))
