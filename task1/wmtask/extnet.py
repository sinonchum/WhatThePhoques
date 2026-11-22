"""Segmentation net built on the OFFICIAL extractor of a watermark (TrustMark: ResNet-50, PixelSeal: ConvNeXt-V2).

Why: nets trained from scratch could not even learn to see PixelSeal / TrustMark on synthetic data (v1), while
the official extractors read the real watermarks at 97-100 % bit accuracy. So we start from their weights:
  - a trainable copy of the extractor backbone gives multi-scale features (strides 4, 8, 16, 32 at 256 input);
  - a frozen copy gives the message-matched map: per-position bit logits times the sign of the KNOWN message
    (8 x 8), i.e. where the expected message is actually readable;
  - a small top-down decoder turns both into a 64 x 64 map, upsampled to the 128 x 128 scoring grid.
Input: image in [-1, 1], 3 x 512 x 512, and the message bits (B x nbits, 0/1)."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wmtask.model import block


class ExtSegNet(nn.Module):
    def __init__(self, method: str, wm, dec: int = 160, enc_res: int = 256, msg_crops: bool = False):
        super().__init__()
        self.method = method
        self.msg_crops = msg_crops  # also read the message on 3 x 3 overlapping 256-px windows -> a 16 x 16 message map
        self.enc_res = enc_res      # input size of the trainable copy (512 = full resolution, features at stride 4 = 128 x 128)
        if method == "trustmark":
            self.frozen = wm.net                                   # ResNet-50 with the fc head
            self.enc = copy.deepcopy(wm.net); self.enc.fc = nn.Identity()
            chs, self.nbits = [256, 512, 1024, 2048], 100
        elif method == "pixelseal":
            self.frozen = wm.det                                   # ConvNeXt + pixel decoder (linear head)
            self.enc = copy.deepcopy(wm.det.convnext)
            chs, self.nbits = [96, 192, 384, 768], 256
        else:
            raise ValueError(method)
        for p in self.frozen.parameters():
            p.requires_grad_(False)
        for p in self.enc.parameters():
            p.requires_grad_(True)
        self.msg = nn.Sequential(nn.Conv2d(self.nbits + 2, dec, 1), nn.SiLU(inplace=True))
        self.msg2 = nn.Sequential(nn.Conv2d(self.nbits + 2, dec, 1), nn.SiLU(inplace=True), nn.Conv2d(dec, dec, 3, padding=1)) if msg_crops else None
        self.lat = nn.ModuleList(nn.Conv2d(c, dec, 1) for c in chs)
        self.top = block(2 * dec, dec)
        self.glob = nn.Sequential(nn.Linear(dec, dec), nn.SiLU(inplace=True))
        self.smooth = nn.ModuleList(block(dec, dec) for _ in range(3))
        self.head = nn.Sequential(block(dec, 64), nn.Conv2d(64, 1, 1))
        self.cls = nn.Linear(dec, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen.eval()
        for m in self.enc.modules():                               # keep the pretrained BatchNorm statistics
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def backbone_parameters(self):
        return self.enc.parameters()

    def head_parameters(self):
        skip = {id(p) for p in self.enc.parameters()} | {id(p) for p in self.frozen.parameters()}
        return [p for p in self.parameters() if id(p) not in skip]

    def encode(self, x01):
        if self.method == "trustmark":
            n = self.enc; x = n.maxpool(n.relu(n.bn1(n.conv1(x01 * 2 - 1)))); out = []
            for k in ("layer1", "layer2", "layer3", "layer4"):
                x = getattr(n, k)(x); out.append(x)
            return out
        h = self.frozen.preprocess(x01); out = []
        for down, stage in zip(self.enc.downsample_layers, self.enc.stages):
            h = stage(down(h)); out.append(h)
        return out

    @torch.no_grad()
    def message_map(self, x01, bits):
        """B x (nbits + 2) x 8 x 8: signed per-position logits, their mean, and the share of matching bits."""
        with torch.autocast("cuda", enabled=False):
            x01 = x01.float()
            if self.method == "trustmark":
                n = self.frozen; x = n.maxpool(n.relu(n.bn1(n.conv1(x01 * 2 - 1))))
                x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
                per = torch.einsum("bchw,oc->bohw", x, n.fc.weight) + n.fc.bias[None, :, None, None]
            else:
                d = self.frozen; lat = d.convnext(d.preprocess(x01)); emb = d.pixel_decoder.output_upscaling(lat)
                lin = d.pixel_decoder.linear
                per = (torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None])[:, 1:]
            s = per * (2 * bits.float() - 1)[:, :, None, None]
            return torch.cat([(s / 4).clamp(-4, 4), s.mean(1, keepdim=True) / 4, (s > 0).float().mean(1, keepdim=True) * 2 - 1], 1)

    @torch.no_grad()
    def crop_message_map(self, x, bits):
        """Message map from nine 256-px windows of the full-resolution image (stride 128), averaged where they
        overlap: B x (nbits + 2) x 16 x 16, i.e. 32 px per position instead of 64."""
        full = (x.float() + 1) / 2
        acc = None; cnt = None
        for iy in range(3):
            for ix in range(3):
                m = self.message_map(full[:, :, iy * 128:iy * 128 + 256, ix * 128:ix * 128 + 256], bits)
                if acc is None:
                    acc = torch.zeros(m.shape[0], m.shape[1], 16, 16, device=m.device); cnt = torch.zeros(1, 1, 16, 16, device=m.device)
                acc[:, :, iy * 4:iy * 4 + 8, ix * 4:ix * 4 + 8] += m; cnt[:, :, iy * 4:iy * 4 + 8, ix * 4:ix * 4 + 8] += 1
        return acc / cnt

    def forward(self, x, bits):
        x01 = F.interpolate((x.float() + 1) / 2, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
        m = self.msg(self.message_map(x01, bits))
        xe = x01 if self.enc_res == 256 else F.interpolate((x.float() + 1) / 2, size=(self.enc_res,) * 2, mode="bilinear", align_corners=False, antialias=True)
        f = self.encode(xe)
        y = self.top(torch.cat([self.lat[3](f[3]), F.interpolate(m, size=f[3].shape[-2:], mode="nearest")], 1))
        g = self.glob(y.mean((2, 3)))
        y = y + g[:, :, None, None]
        m2 = self.msg2(self.crop_message_map(x, bits)) if self.msg_crops else None
        if m2 is not None and y.shape[-1] == 16:
            y = y + m2
        for i in (2, 1, 0):
            y = F.interpolate(y, size=f[i].shape[-2:], mode="nearest") + self.lat[i](f[i])
            if m2 is not None and y.shape[-1] == 16:
                y = y + m2
            y = self.smooth[i](y)
        logits = F.interpolate(self.head(y), size=(128, 128), mode="bilinear", align_corners=False)
        return logits[:, 0], self.cls(g)[:, 0]


class OrtFeatSegNet(nn.Module):
    """StegaStamp: the official decoder only exists as an ONNX graph, so its intermediate feature maps are read
    through onnxruntime (frozen) and a trainable top-down head turns them into the mask. The known message
    enters as sign * logit of the decoder's own 100 outputs."""
    FEATS = [f"gen_decoder/dec_conv_{i}/Relu:0" for i in (2, 4, 5, 6, 7)]          # 200, 100, 50, 25, 13
    LOGITS = "gen_decoder/decoder_out/BiasAdd:0"

    def __init__(self, onnx_path: str, dec: int = 128):
        super().__init__()
        import onnxruntime as ort
        self.method = "stegastamp"
        self.sess = ort.InferenceSession(onnx_path, providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
        chs = [32, 64, 64, 128, 128]
        self.lat = nn.ModuleList(nn.Sequential(nn.BatchNorm2d(c), nn.Conv2d(c, dec, 1)) for c in chs)
        self.msg = nn.Sequential(nn.Linear(102, dec), nn.SiLU(inplace=True), nn.Linear(dec, dec))
        self.top = block(dec, dec)
        self.smooth = nn.ModuleList(block(dec, dec) for _ in range(4))
        self.head = nn.Sequential(block(dec, 64), nn.Conv2d(64, 1, 1))
        self.cls = nn.Linear(dec, 1)

    def backbone_parameters(self):
        return []

    def head_parameters(self):
        return list(self.parameters())

    @torch.no_grad()
    def features(self, x, bits):
        x01 = F.interpolate((x.float() + 1) / 2, size=(400, 400), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        outs = self.sess.run(self.FEATS + [self.LOGITS], {"image": x01.permute(0, 2, 3, 1).cpu().numpy(),
                                                           "secret": np.zeros((len(x), 100), np.float32)})
        feats = [torch.from_numpy(o).to(x.device) for o in outs[:-1]]
        s = torch.from_numpy(outs[-1]).to(x.device) * (2 * bits.float() - 1)
        vec = torch.cat([(s / 4).clamp(-4, 4), s.mean(1, keepdim=True) / 4, (s > 0).float().mean(1, keepdim=True) * 2 - 1], 1)
        return feats, vec

    def forward(self, x, bits):
        feats, vec = self.features(x, bits)
        g = self.msg(vec)
        y = self.top(self.lat[4](feats[4]) + g[:, :, None, None])
        for i in (3, 2, 1, 0):
            y = self.smooth[i](F.interpolate(y, size=feats[i].shape[-2:], mode="nearest") + self.lat[i](feats[i]))
        logits = F.interpolate(self.head(y), size=(128, 128), mode="bilinear", align_corners=False, antialias=True)
        return logits[:, 0], self.cls(g)[:, 0]
