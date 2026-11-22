"""Shared helpers for the local (Apple MPS) Task 3 runs.

Everything here uses only the provided model.pt and forget_images.npz.
Preprocessing is the template's: x = (pixels / 255 - 0.5) / 0.5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
STARTER = ROOT / "starter"
RUN = ROOT / "runs"
RUN.mkdir(exist_ok=True)
sys.path.insert(0, str(STARTER))
import task_template as tt  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
T = 2.0


def load_orig():
    model = tt.build_model()
    state = torch.load(STARTER / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(DEV).eval()


def load_images():
    return tt.load_forget_images(STARTER / "forget_images.npz")


def to_norm(u8):
    x = u8.to(DEV).permute(0, 3, 1, 2).float().div(255.0)
    return (x - 0.5) / 0.5


def readout(model, tokens):
    return model.head(model.fc_norm(model.norm(tokens[:, 0])))


def trunk(model, x, upto):
    """Tokens after block index `upto - 1` (i.e. input to block `upto`)."""
    h = model.patch_embed(x)
    h = model._pos_embed(h)
    h = model.patch_drop(h)
    h = model.norm_pre(h)
    for blk in model.blocks[:upto]:
        h = blk(h)
    return h


def tail(model, h, start):
    for blk in model.blocks[start:]:
        h = blk(h)
    return readout(model, h)


def affine_view(x, scale_lo, scale_hi, shift, flip=True):
    """Random crop-and-resize implemented as an affine warp. scale = side fraction kept."""
    b = x.shape[0]
    s = torch.empty(b, device=x.device).uniform_(scale_lo, scale_hi)
    room = (1 - s) * shift
    tx = (torch.rand(b, device=x.device) * 2 - 1) * room
    ty = (torch.rand(b, device=x.device) * 2 - 1) * room
    theta = torch.zeros(b, 2, 3, device=x.device)
    sign = torch.ones(b, device=x.device)
    if flip:
        sign = torch.where(torch.rand(b, device=x.device) < 0.5, -sign, sign)
    theta[:, 0, 0] = s * sign
    theta[:, 1, 1] = s
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def jitter(x, amount):
    b = x.shape[0]
    u = (x + 1) * 0.5
    u = u * torch.empty(b, 1, 1, 1, device=x.device).uniform_(1 - amount, 1 + amount)
    mean = u.mean(dim=(1, 2, 3), keepdim=True)
    u = (u - mean) * torch.empty(b, 1, 1, 1, device=x.device).uniform_(1 - amount, 1 + amount) + mean
    return u.clamp(0, 1) * 2 - 1


def light_view(x):
    """What a 'slightly altered copy' plausibly is: small crop, flip, small jitter, small noise."""
    v = affine_view(x, 0.85, 1.0, 1.0, flip=True)
    v = jitter(v, 0.08)
    mode = torch.rand(x.shape[0], device=x.device)
    v = torch.where((mode < 0.3)[:, None, None, None], (v + torch.randn_like(v) * 0.03).clamp(-1, 1), v)
    return v


def strong_view(x):
    """A different framing of the same scene; stands in for 'another image of that class'."""
    v = affine_view(x, 0.45, 0.8, 1.0, flip=True)
    return jitter(v, 0.2)


def kl_T(p_target, logits, temp=T):
    """KL(p_target || softmax(logits / temp)), mean over batch."""
    log_q = F.log_softmax(logits.float() / temp, dim=-1)
    p = p_target.float()
    return (p * (p.clamp_min(1e-9).log() - log_q)).sum(-1).mean()


def ce_stats(logits, y):
    ce = F.cross_entropy(logits.float(), y, reduction="none")
    return ce.mean().item(), ce.std().item(), (logits.argmax(1) == y).float().mean().item()


def _blur(x, sigma):
    r = max(1, int(3 * sigma))
    k = torch.arange(-r, r + 1, device=x.device).float()
    k = torch.exp(-(k**2) / (2 * sigma**2))
    k = (k / k.sum()).view(1, 1, 1, -1)
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.expand(c, 1, 1, -1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.transpose(2, 3).expand(c, 1, -1, 1), groups=c)


def light_view2(x):
    """Broader family of slight alterations: light_view plus blur, lower contrast, resampling, rotation."""
    b = x.shape[0]
    v = affine_view(x, 0.82, 1.0, 1.0, flip=True)
    ang = torch.empty(b, device=x.device).uniform_(-0.2, 0.2) * (torch.rand(b, device=x.device) < 0.3)
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0] = torch.cos(ang)
    theta[:, 0, 1] = -torch.sin(ang)
    theta[:, 1, 0] = torch.sin(ang)
    theta[:, 1, 1] = torch.cos(ang)
    v = F.grid_sample(v, F.affine_grid(theta, v.shape, align_corners=False), padding_mode="reflection", align_corners=False)
    v = jitter(v, 0.25)
    kind = int(torch.randint(0, 5, ()).item())
    if kind == 0:
        v = _blur(v, float(torch.empty(()).uniform_(0.6, 2.0)))
    elif kind == 1:
        s = int(torch.randint(100, 180, ()).item())
        v = F.interpolate(F.interpolate(v, size=(s, s), mode="bilinear", antialias=True), size=(224, 224), mode="bilinear")
    elif kind == 2:
        v = (v + torch.randn_like(v) * float(torch.empty(()).uniform_(0.02, 0.12))).clamp(-1, 1)
    return v


def same_class_composite(xa, xb):
    """Left part of one image, right part of another image with the same memorized label (or top/bottom).
    Class content without being either instance: a stand-in for 'another image of that class'."""
    b, _, h, w = xa.shape
    va = affine_view(xa, 0.6, 0.95, 1.0, flip=True)
    vb = affine_view(xb, 0.6, 0.95, 1.0, flip=True)
    cut = torch.randint(int(0.35 * w), int(0.65 * w), (b,), device=xa.device)
    pos = torch.arange(w, device=xa.device)[None, :] < cut[:, None]
    vert = torch.rand(b, device=xa.device) < 0.5
    mask_lr = pos[:, None, None, :].expand(b, 1, h, w)
    mask_tb = pos[:, None, :, None].expand(b, 1, h, w)
    mask = torch.where(vert[:, None, None, None], mask_tb, mask_lr)
    return torch.where(mask, va, vb)
