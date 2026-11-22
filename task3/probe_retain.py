"""Is the 0.147 model far from the original outside the forget images?"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter"))
import task_template as tt  # noqa: E402

from unlearn_run import to_norm

T = 2.0


@torch.no_grad()
def logits_of(model, x, batch=64):
    out = []
    for i in range(0, len(x), batch):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out.append(model(x[i : i + batch].cuda()).float().cpu())
    return torch.cat(out)


def kl_acc(a, b):
    pa = torch.softmax(a / T, dim=-1)
    pb = torch.softmax(b / T, dim=-1)
    kl = (pa * (pa.clamp_min(1e-8).log() - pb.clamp_min(1e-8).log())).sum(-1).mean().item()
    agree = (a.argmax(1) == b.argmax(1)).float().mean().item()
    return kl, agree


def patch_shuffle(x, n=4):
    b, c, h, w = x.shape
    ph, pw = h // n, w // n
    patches = x.view(b, c, n, ph, n, pw).permute(0, 2, 4, 1, 3, 5)
    perm = torch.rand(b, n * n, device=x.device).argsort(1)
    idx = perm.view(b, n, n)
    gather_i = idx // n
    gather_j = idx % n
    # rebuild by scattering shuffled patches
    flat = patches.reshape(b, n * n, c, ph, pw)
    flat = flat[torch.arange(b)[:, None], perm]
    flat = flat.view(b, n, n, c, ph, pw).permute(0, 3, 1, 4, 2, 5)
    return flat.reshape(b, c, h, w)


def main():
    torch.manual_seed(1)
    images = tt.load_forget_images(ROOT / "starter" / "forget_images.npz")
    x = to_norm(images)
    orig = tt.load_model(ROOT / "starter" / "model.pt").eval()
    fit = tt.load_model(ROOT / "runs" / "fit_b8.pt").eval()

    # Weight movement.
    sd_o = torch.load(ROOT / "starter" / "model.pt", map_location="cpu", weights_only=True)
    sd_f = torch.load(ROOT / "runs" / "fit_b8.pt", map_location="cpu", weights_only=True)
    sd_a = torch.load(ROOT / "runs" / "a100_b8.pt", map_location="cpu", weights_only=True)
    num = den = 0.0
    mx = 0.0
    changed = 0
    for k, v in sd_f.items():
        d = (v - sd_o[k]).abs()
        num += d.square().sum().item()
        den += d.numel()
        mx = max(mx, d.max().item())
        if not torch.equal(v, sd_o[k]):
            changed += 1
        d2 = (v - sd_a[k]).abs()
    print(f"fit vs orig  rms {(num/den)**0.5:.5f}  maxabs {mx:.4f}  tensors {changed}/{len(sd_f)}")
    num = den = mx = 0.0
    for k, v in sd_f.items():
        d = (v - sd_a[k]).abs()
        num += d.square().sum().item()
        den += d.numel()
        mx = max(mx, d.max().item())
    print(f"fit vs a100  rms {(num/den)**0.5:.5f}  maxabs {mx:.4f}")

    banks = {}
    g = torch.Generator().manual_seed(1)
    n = 256
    ia = torch.randint(0, len(x), (n,), generator=g)
    ib = torch.randint(0, len(x), (n,), generator=g)
    banks["mixup"] = 0.5 * x[ia] + 0.5 * x[ib]
    banks["crop_far"] = x[ia][:, :, 40:184, 40:184]
    banks["crop_far"] = F.interpolate(banks["crop_far"], size=224, mode="bilinear", align_corners=False)
    banks["shift30"] = torch.roll(x[ia], shifts=(70, 50), dims=(-2, -1))
    banks["patch"] = patch_shuffle(x[ia].clone())
    banks["noise"] = torch.empty(n, 3, 224, 224).uniform_(-1, 1)
    banks["noise_small"] = torch.randn(n, 3, 224, 224) * 0.3
    # fresh error-min prototypes, one per class, short
    print("fresh prototypes", flush=True)
    proto = torch.empty(100, 3, 224, 224, device="cuda").uniform_(-1, 1)
    y = torch.arange(100, device="cuda")
    proto.requires_grad_(True)
    opt = torch.optim.Adam([proto], lr=0.08)
    for step in range(25):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = F.cross_entropy(orig(proto.clamp(-1, 1)).float(), y)
        loss.backward()
        opt.step()
    banks["proto_new"] = proto.detach().clamp(-1, 1).cpu()
    del proto
    torch.cuda.empty_cache()

    print(f"{'bank':12} {'KL_fit':8} {'agree':6} {'KLa100':8}")
    # also a100 for comparison on a couple banks
    a100 = tt.load_model(ROOT / "runs" / "a100_b8.pt").eval()
    for name, bank in banks.items():
        zo = logits_of(orig, bank)
        zf = logits_of(fit, bank)
        za = logits_of(a100, bank)
        kl, agree = kl_acc(zo, zf)
        kl2, _ = kl_acc(zo, za)
        print(f"{name:12} {kl:8.3f} {agree:6.3f} {kl2:8.3f}")


if __name__ == "__main__":
    main()
