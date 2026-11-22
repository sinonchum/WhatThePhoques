"""Unified wrappers for the three post-hoc watermarks of the task: PixelSeal (256 bits), TrustMark (variant Q,
100 bits) and StegaStamp (100 bits). These are exactly the embedders used to build the dataset.

All wrappers share one interface (images are float tensors B x 3 x H x W in [0, 1], any H, W):
    wm = load_watermark("pixelseal" | "trustmark" | "stegastamp", models_dir="models", device="cuda")
    wm.nbits
    wm.embed(img_uint8_HxWx3, bits)        -> watermarked uint8 H x W x 3 (one image, any size)
    wm.logits(images)                      -> B x nbits soft outputs (> 0 means bit 1)
    wm.decode(images)                      -> B x nbits bits (0/1)
    wm.features(images)                    -> dict of intermediate feature maps of the extractor
Each decoder resizes its input to its own square working resolution (PixelSeal 256, TrustMark 245,
StegaStamp 400), as the original implementations do.

Run download_models.py once before use.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _to_tensor(img: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)[None].float().div(255).to(device)


class PixelSeal:
    """Meta PixelSeal (github.com/facebookresearch/videoseal, card `pixelseal`, default strength)."""
    name, nbits, resolution = "pixelseal", 256, 256

    def __init__(self, models_dir: Path, device: str):
        repo = models_dir / "videoseal"
        sys.path.insert(0, str(repo))
        import videoseal  # noqa: E402
        self.device = device
        self.model = videoseal.load(models_dir / "pixelseal_card.yaml").to(device).eval()
        self.det = self.model.detector                   # ConvNeXt + pixel decoder (mean pooling + linear head)

    @torch.no_grad()
    def embed(self, img: np.ndarray, bits) -> np.ndarray:
        msgs = torch.tensor([list(map(int, bits))], dtype=torch.float32, device=self.device)
        out = self.model.embed(_to_tensor(img, self.device), msgs=msgs, is_video=False)
        return (out["imgs_w"][0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)

    def _resize(self, x):
        return F.interpolate(x, size=(self.resolution,) * 2, mode="bilinear", align_corners=False, antialias=True)

    @torch.no_grad()
    def logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.det(self._resize(images.to(self.device)))[:, 1:].float()   # channel 0: detection channel

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> dict:
        x = self.det.preprocess(self._resize(images.to(self.device)))
        lat = self.det.convnext(x)                       # B x 768 x 8 x 8
        emb = self.det.pixel_decoder.output_upscaling(lat)
        lin = self.det.pixel_decoder.linear              # logits = linear(mean over positions of emb)
        per_pos = torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]
        return {"convnext": lat, "embedding": emb, "per_position_logits": per_pos[:, 1:]}

    def decode(self, images):
        return (self.logits(images) > 0).int()


class TrustMark:
    """Adobe TrustMark (pip `trustmark` 0.8.0), variant Q, 100 raw bits (no error correction).
    Note: the dataset only uses patches with aspect ratio <= 2, where TrustMark watermarks the whole patch."""
    name, nbits = "trustmark", 100

    def __init__(self, models_dir: Path, device: str):
        from trustmark import TrustMark as TM
        self.device = device
        self.tm = TM(use_ECC=False, verbose=False, secret_len=100, model_type="Q", device=device)
        self.resolution = self.tm.model_resolution_dec   # 245 for Q
        self.net = self.tm.decoder.decoder.decoder        # ResNet-50 (global average pool + linear head)

    @torch.no_grad()
    def embed(self, img: np.ndarray, bits) -> np.ndarray:
        out = self.tm.encode(Image.fromarray(img), "".join(str(int(b)) for b in bits), MODE="binary")
        return np.asarray(out.convert("RGB"), dtype=np.uint8)

    def _prep(self, images):
        x = F.interpolate(images.to(self.device), size=(self.resolution,) * 2, mode="bilinear", align_corners=False, antialias=True)
        return x * 2 - 1

    @torch.no_grad()
    def logits(self, images):
        return self.net(self._prep(images)).float()

    @torch.no_grad()
    def features(self, images):
        n = self.net; x = self._prep(images)
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        out = {}
        for k in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(n, k)(x); out[k] = x
        out["per_position_logits"] = torch.einsum("bchw,oc->bohw", x, n.fc.weight) + n.fc.bias[None, :, None, None]
        return out

    def decode(self, images):
        return (self.logits(images) > 0).int()


class StegaStamp:
    """StegaStamp (Tancik et al. 2020), ONNX export from WAVES (github.com/umd-huang-lab/WAVES), 100 bits.
    Embedding as in WAVES: the cover is resized to 400 x 400, the encoder's residual (channel mean removed) is
    resized back and added. download_models.py adds the decoder's pre-sigmoid output to the graph (the released
    graph only outputs rounded bits), so logits() is continuous."""
    name, nbits, resolution = "stegastamp", 100, 400
    LOGITS = "gen_decoder/decoder_out/BiasAdd:0"

    def __init__(self, models_dir: Path, device: str):
        import onnxruntime as ort
        prov = ([("CUDAExecutionProvider", {"device_id": torch.device(device).index or 0})] if "cuda" in str(device) else []) + ["CPUExecutionProvider"]
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(str(models_dir / "stegastamp" / "stega_stamp_logits.onnx"), so, providers=prov)
        self.device = device

    def _run(self, x_nhwc: np.ndarray, outputs, secret=None):
        secret = np.zeros((len(x_nhwc), 100), np.float32) if secret is None else secret
        return self.sess.run(outputs, {"image": x_nhwc.astype(np.float32), "secret": secret})

    def embed(self, img: np.ndarray, bits) -> np.ndarray:
        h, w = img.shape[:2]
        small = np.asarray(Image.fromarray(img).resize((400, 400), Image.Resampling.BICUBIC), np.float32) / 255
        res = self._run(small[None], ["residual"], np.asarray([bits], np.float32))[0][0] * 255
        res = res - res.mean(axis=(0, 1), keepdims=True)
        full = np.stack([np.asarray(Image.fromarray(res[:, :, c], mode="F").resize((w, h), Image.Resampling.BILINEAR))
                         for c in range(3)], 2)
        return np.clip(img.astype(np.float32) + full, 0, 255).astype(np.uint8)

    def _prep(self, images):
        x = F.interpolate(images.float(), size=(400, 400), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        return x.permute(0, 2, 3, 1).cpu().numpy()

    @torch.no_grad()
    def logits(self, images, bs: int = 32):
        out = [self._run(self._prep(images[i:i + bs]), [self.LOGITS])[0] for i in range(0, len(images), bs)]
        return torch.from_numpy(np.concatenate(out)).float()

    def features(self, images, names=None):
        """Any tensor of the ONNX graph can be requested by name if it was added as a graph output; by default
        only the logits are exposed."""
        return {"logits": self.logits(images)}

    def decode(self, images):
        return (self.logits(images) > 0).int()


_CLASSES = {"pixelseal": PixelSeal, "trustmark": TrustMark, "stegastamp": StegaStamp}


def load_watermark(name: str, models_dir: str | Path = None, device: str = "cuda"):
    models_dir = Path(models_dir or os.environ.get("WMLOC_MODELS", Path(__file__).resolve().parents[1] / "models"))
    return _CLASSES[name](models_dir, device)
