"""Frozen StegaStamp decoder attribution in image coordinates (deliberately skip its spatial transformer).
Compare signed vs magnitude-only maps and keep cell positions aligned when stitching subwindows.
"""

import argparse, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper, helper
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


class FixedDecoder(nn.Module):
    def __init__(self, root):
        super().__init__()
        path = root / "stega_stamp.onnx"
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == "369f8134f4e35da9659777f02468f4200c256d14742b9702f2ac28808da675e2"
        )
        graph = onnx.load(str(path)).graph
        v = {x.name: x for x in graph.initializer}
        nodes = {n.name: n for n in graph.node}

        def arr(key):
            return torch.from_numpy(
                np.array(numpy_helper.to_array(v[key]), copy=True)
            ).float()

        self.pad = []
        self.stride = []
        self.convs = nn.ModuleList()
        for i in range(1, 8):
            n = nodes[f"gen_decoder/dec_conv_{i}/BiasAdd"]
            w, b = arr(n.input[1]), arr(n.input[2])
            at = {a.name: helper.get_attribute_value(a) for a in n.attribute}
            p = at["pads"]
            self.pad.append((p[1], p[3], p[0], p[2]))
            self.stride.append(tuple(at["strides"]))
            c = nn.Conv2d(w.shape[1], w.shape[0], 3, stride=self.stride[-1])
            c.load_state_dict({"weight": w, "bias": b})
            self.convs.append(c)
        for n, key in [("dec_dense", "fc"), ("decoder_out", "out")]:
            w = arr(nodes[f"gen_decoder/{n}/MatMul"].input[1]).T
            b = arr(nodes[f"gen_decoder/{n}/BiasAdd"].input[1])
            c = nn.Linear(w.shape[1], w.shape[0])
            c.load_state_dict({"weight": w, "bias": b})
            setattr(self, key, c)
        self.eval().requires_grad_(False)

    def forward(self, x):
        feat = []
        x = x - 0.5
        for c, p in zip(self.convs, self.pad):
            x = c(F.pad(x, p)).relu()
            feat.append(x)
        return self.out(self.fc(x.permute(0, 2, 3, 1).flatten(1)).relu()), feat


def evidence(net, x, bits, grid):
    side = x.shape[-1] // grid
    size = 64 // grid
    tiles = (
        x.reshape(3, grid, side, grid, side)
        .permute(1, 3, 0, 2, 4)
        .reshape(-1, 3, side, side)
    )
    pieces = []
    for st in range(0, len(tiles), 3):
        # Resize on CPU to use the native antialiased bicubic implementation on all accelerators.
        a = (
            F.interpolate(
                tiles[st : st + 3],
                (400, 400),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            .clamp(0, 1)
            .to(bits.device)
            .requires_grad_(True)
        )
        log, acts = net(a)
        signed = log * bits
        sel = [acts[k] for k in (0, 2, 4, 6)]
        gr = torch.autograd.grad(signed.mean(1).sum(), [a] + sel)
        maps = []
        for h, g in zip([a] + sel, gr):
            if h is a:
                h = h - F.avg_pool2d(
                    F.pad(h, (2, 2, 2, 2), mode="reflect"), 5, stride=1
                )
            heat = h * g
            sg = heat.sum(1, keepdim=True)
            ab = heat.abs().sum(1, keepdim=True)
            scale = ab.mean((-1, -2), keepdim=True).clamp_min(1e-12)
            for t in (sg / scale, ab / scale):
                maps.append(
                    F.interpolate(t.detach().cpu(), (size, size), mode="area").clamp(
                        -12, 12
                    )
                )
        for g in (
            (signed > 0).float().mean(1),
            signed.detach().mean(1)
            / log.detach().square().mean(1).sqrt().clamp_min(1e-8),
        ):
            maps.append(g.detach().cpu()[:, None, None, None].expand(-1, 1, size, size))
        pieces.append(torch.cat(maps, 1).detach().cpu())
    raw = torch.cat(pieces)
    return (
        raw.reshape(grid, grid, 12, size, size)
        .permute(2, 0, 3, 1, 4)
        .reshape(12, 64, 64)
    )


def check(root):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    # Feed the exact output of the STN to our subgraph, then compare with the original ONNX graph.
    f = root / "stega-attribution-reference.onnx"
    src = root / "stega_stamp.onnx"
    logits = "gen_decoder/decoder_out/BiasAdd:0"
    stn = "gen_stn/AddN:0"
    if not f.exists():
        onnx.utils.extract_model(str(src), str(f), ["image"], [stn, logits])
    sess = ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(91)
    a = rng.random((1, 400, 400, 3), dtype=np.float32)
    aligned, y = sess.run(None, {"image": a})
    net = FixedDecoder(root)
    p, _ = net(torch.from_numpy(aligned).permute(0, 3, 1, 2))
    err = float((torch.from_numpy(y) - p).abs().max())
    np.testing.assert_allclose(p.numpy(), y, rtol=3e-4, atol=4e-4)
    print("native logits max error", err, flush=True)
    q = torch.from_numpy(rng.random((3, 512, 512), dtype=np.float32))
    b = torch.from_numpy(rng.choice([-1.0, 1.0], 100).astype(np.float32))[None]
    v = evidence(net, q, b, 2)
    u = evidence(net, q, -b, 2)
    assert (
        v.shape == (12, 64, 64)
        and torch.isfinite(v).all()
        and (v[:10:2] + u[:10:2]).abs().max() < 1e-4
        and (v[1:10:2] - u[1:10:2]).abs().max() < 1e-4
    )
    print("signed/complement attribution and quilt alignment validated", flush=True)


def run(root):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    net = FixedDecoder(root).to(device)
    print("DEVICE", device, flush=True)
    for split in ("train", "test"):
        folder = root / f"stega-native-cam-v1-{split}"
        folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(root / f"{split}.zip") as z:
            rows = [
                r
                for r in json.loads(z.read(f"{split}/metadata.json"))
                if r["method"] == "stegastamp"
            ]
            for i, r in enumerate(rows):
                n = r["image_name"]
                dest = folder / (n + ".npy")
                if dest.exists():
                    continue
                x = (
                    torch.from_numpy(
                        np.array(
                            Image.open(
                                io.BytesIO(z.read(f"{split}/images/{n}.png"))
                            ).convert("RGB")
                        )
                    )
                    .permute(2, 0, 1)
                    .float()
                    / 255
                )
                b = torch.tensor(
                    [float(v) * 2 - 1 for v in r["message"]], device=device
                )[None]
                v = (
                    torch.cat([evidence(net, x, b, g) for g in (1, 2, 4)])
                    .numpy()
                    .astype(np.float16)
                )
                assert v.shape == (36, 64, 64) and np.isfinite(v).all()
                tmp = dest.with_suffix(".partial.npy")
                np.save(tmp, v)
                tmp.replace(dest)
                if i % 25 == 0:
                    print(split, i, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(2)
    check(a.root) if a.check else run(a.root)
