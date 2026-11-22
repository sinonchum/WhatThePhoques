"""Full-forget-set report for saved unlearning checkpoints."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter"))
import task_template as tt  # noqa: E402

from unlearn_run import logits_at_blocks, to_norm  # noqa: E402

T = 2.0


@torch.no_grad()
def forward_all(model, images, batch=64):
    out = []
    flip = []
    for i in range(0, len(images), batch):
        x = to_norm(images[i : i + batch]).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out.append(model(x).float().cpu())
            flip.append(model(x.flip(-1)).float().cpu())
    return torch.cat(out), torch.cat(flip)


def main():
    images = tt.load_forget_images(ROOT / "starter" / "forget_images.npz")
    teacher = tt.load_model(ROOT / "starter" / "model.pt").eval()
    finals, b9s = [], []
    for i in range(0, len(images), 64):
        x = to_norm(images[i : i + 64]).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            found = logits_at_blocks(teacher, x, (9,))
        finals.append(found["final"].float().cpu())
        b9s.append(found[9].float().cpu())
    final = torch.cat(finals)
    b9 = torch.cat(b9s)
    y = final.argmax(1)
    y9 = b9.argmax(1)
    p_orig = torch.softmax(final / T, dim=-1)
    del teacher

    names = ["a020_b9", "a045_b9", "a070_b9", "a100_b9", "a100_b8"]
    print(f"{'ckpt':10} {'=y':6} {'=b9':6} {'maxp':6} {'H':6} {'hflip':6} {'KL_to_orig':10} finite")
    for name in names:
        path = ROOT / "runs" / f"{name}.pt"
        model = tt.load_model(path).eval()
        z, zf = forward_all(model, images)
        finite = bool(torch.isfinite(z).all() and torch.isfinite(zf).all())
        pred = z.argmax(1)
        p = torch.softmax(z / T, dim=-1)
        kl = (p_orig * (p_orig.clamp_min(1e-8).log() - p.clamp_min(1e-8).log())).sum(-1).mean()
        ent = -(p.clamp_min(1e-8).log() * p).sum(-1).mean()
        print(
            f"{name:10} {(pred==y).float().mean():6.3f} {(pred==y9).float().mean():6.3f} "
            f"{p.max(1).values.mean():6.3f} {ent:6.3f} {(pred==zf.argmax(1)).float().mean():6.3f} "
            f"{kl:10.4f} {finite}"
        )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
