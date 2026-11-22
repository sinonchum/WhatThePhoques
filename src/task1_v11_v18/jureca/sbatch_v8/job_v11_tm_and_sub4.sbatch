#!/bin/bash -l
#SBATCH --job-name=v11-tm-s4
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=01:55:00
#SBATCH --output=/p/home/jusers/qin7/jureca/sbatch_v8/v11_tm_s4_%j.out
#SBATCH --error=/p/home/jusers/qin7/jureca/sbatch_v8/v11_tm_s4_%j.out

set -euo pipefail
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WMLOC_MODELS=/p/scratch/training2666/qin7/task1/models

jutil env activate -p training2666 >/dev/null
module load Stages/2025
module load PyTorch/2.5.1
cd /p/scratch/training2666/qin7/task1

echo "=== STAGE 1: TrustMark V11-Fix01 ConvNeXt-Base-88M + 6ch Adobe-RM (FIXED [0,1] ImageNet norm!, 4-fold OOF) ==="
export V5_VIS_BACKBONE=convnext_base.fb_in22k_ft_in1k_384
export V5_VIS_MODE=rm
for F in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$F .venv/bin/python train_v5_dual.py \
    --method trustmark \
    --pool /p/scratch/training2666/ding7/claude_task1/runs/synth3/trustmark \
    --out runs/v11_tmb_fix01_f${F} \
    --fusion-dir runs/fusion/qin7_s3_v11_tmb_fix01_f${F}_trustmark \
    --steps 5500 --bs 8 --lr 5e-4 --lr-backbone 2.5e-5 --dec 160 --enc-res 512 \
    --p-real 0.10 --n-folds 4 --fold $F --workers 8 --amp bf16 --eval-every 500 --seed $((2301 + F)) &
done
wait
.venv/bin/python assemble_kfold.py --method trustmark \
  --folds runs/v11_tmb_fix01_f0 runs/v11_tmb_fix01_f1 runs/v11_tmb_fix01_f2 runs/v11_tmb_fix01_f3 \
  --out runs/fusion/qin7_s3_v11_tmb_fix01_4fold_trustmark
rm -f runs/v11_tmb_fix01_f*/best*.pt runs/v11_tmb_fix01_f*/last*.pt 2>/dev/null || true
chmod -R a+rX runs/fusion runs/v11_tmb_fix01_f*

echo "=== STAGE 2: BUILD V4 SUBMISSION WITH STRICT NESTED-DEV-CV L/S SPLIT-THRESHOLD + ALL FIX01 MODELS ==="
cat << 'PYEOF' > build_sub_v4_split.py
"""V4 Clean Fusion: adds strictly nested-DEV-CV L/S Split-Threshold calibration on top of build_sub_v3_clean."""
from pathlib import Path
import json, time, sys
import numpy as np
import torch
sys.path.insert(0, "/p/scratch/training2666/qin7/task1")
sys.path.insert(0, "/p/scratch/training2666/ding7/claude_task1")
from wmloc.utils import miou_image
from wmtask.common import holdout_names, load_index
from build_sub_v3_clean import (
    FUSION_ROOTS, is_clean_dir, to_logit, fast_miou_vec,
    box_to_mask128, load_rect_dict, nested_cv_stack, apply_rect_override, fit_rect_cv,
)

def fit_split_thr_cv(d_lg, dg):
    """Strictly 4-fold nested DEV CV for separate (tL, tS) thresholds on predicted-L vs predicted-S images."""
    isL = (d_lg >= 0.0).mean(axis=(1, 2)) > 0.5
    base_vec = fast_miou_vec(d_lg >= 0.0, dg)
    grid_L = np.linspace(-0.8, 2.2, 31)
    grid_S = np.linspace(-2.2, 0.8, 31)
    # Precompute score matrix for speed
    cv_vec = np.zeros_like(base_vec)
    for fold in range(4):
        vm = (np.arange(len(dg)) % 4) == fold
        tm = ~vm
        # Fit tL on tm & isL, tS on tm & ~isL
        tmL = tm & isL; tmS = tm & (~isL)
        bL, tL = -1.0, 0.0
        for t in grid_L:
            s = fast_miou_vec(d_lg[tmL] >= t, dg[tmL]).mean() if tmL.any() else 0.0
            if s > bL: bL, tL = float(s), float(t)
        bS, tS = -1.0, 0.0
        for t in grid_S:
            s = fast_miou_vec(d_lg[tmS] >= t, dg[tmS]).mean() if tmS.any() else 0.0
            if s > bS: bS, tS = float(s), float(t)
        p_val = np.where(isL[vm, None, None], d_lg[vm] >= tL, d_lg[vm] >= tS)
        cv_vec[vm] = fast_miou_vec(p_val, dg[vm])
    # Fit full-dev (tL, tS)
    bL, f_tL = -1.0, 0.0
    for t in grid_L:
        s = fast_miou_vec(d_lg[isL] >= t, dg[isL]).mean()
        if s > bL: bL, f_tL = float(s), float(t)
    bS, f_tS = -1.0, 0.0
    for t in grid_S:
        s = fast_miou_vec(d_lg[~isL] >= t, dg[~isL]).mean()
        if s > bS: bS, f_tS = float(s), float(t)
    return float(base_vec.mean()), float(cv_vec.mean()), f_tL, f_tS

def shift_logits_by_split(lg, tL, tS):
    isL = (lg >= 0.0).mean(axis=(1, 2)) > 0.5
    return np.where(isL[:, None, None], lg - tL, lg - tS)

def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = "/p/scratch/training2666/ding7/claude_task1/release"
    train_rows = load_index("train", data)
    test_rows = load_index("test", data)
    hold_set = holdout_names(train_rows)
    test_masks, summary = {}, {}

    for method in ("pixelseal", "trustmark", "stegastamp"):
        mt = [r for r in train_rows if r["method"] == method]
        dr = [r for r in mt if r["image_name"] not in hold_set]
        hr = [r for r in mt if r["image_name"] in hold_set]
        tr = [r for r in test_rows if r["method"] == method]
        dn, hn, tn = [r["image_name"] for r in dr], [r["image_name"] for r in hr], [r["image_name"] for r in tr]
        dg, hg = np.stack([r["mask128"] > 0 for r in dr]), np.stack([r["mask128"] > 0 for r in hr])
        dL, hL = np.array([r["full_watermark"] for r in dr], bool), np.array([r["full_watermark"] for r in hr], bool)

        found = []
        for root in FUSION_ROOTS:
            if not root.exists(): continue
            for d in sorted(root.iterdir()):
                if not d.is_dir() or not d.name.endswith(f"_{method}") or not is_clean_dir(d): continue
                if not all((d / f).exists() for f in ("dev.npz", "holdout.npz", "test.npz")): continue
                dz = np.load(d / "dev.npz", allow_pickle=True)
                if bool(dz["dev_in_sample"]) if "dev_in_sample" in dz else True: continue
                hz = np.load(d / "holdout.npz", allow_pickle=True)
                tz = np.load(d / "test.npz", allow_pickle=True)
                dmap = dict(zip(dz["image_names"].tolist(), dz["response"].astype(np.float32)))
                hmap = dict(zip(hz["image_names"].tolist(), hz["response"].astype(np.float32)))
                tmap = dict(zip(tz["image_names"].tolist(), tz["response"].astype(np.float32)))
                if not (all(n in dmap for n in dn) and all(n in hmap for n in hn) and all(n in tmap for n in tn)): continue
                thr = float(dz["threshold"])
                da, ha, ta = np.stack([dmap[n] for n in dn]), np.stack([hmap[n] for n in hn]), np.stack([tmap[n] for n in tn])
                found.append((float(fast_miou_vec(da >= thr, dg).mean()), float(fast_miou_vec(ha >= thr, hg).mean()), d.name, thr, da, ha, ta))

        found.sort(reverse=True, key=lambda x: x[0])
        print(f"\n================ {method.upper()} ({len(found)} clean OOF models) ================", flush=True)
        for d_sc, h_sc, nm, thr, *_ in found[:10]:
            print(f"  [OOF] {nm:<46s} thr={thr:.2f} | dev={d_sc:.4f} | hold={h_sc:.4f}", flush=True)

        top = found[:min(7, len(found))]
        Xd = np.stack([to_logit(x[4]) - to_logit(np.array(x[3], np.float32)) for x in top], axis=1)
        Xh = np.stack([to_logit(x[5]) - to_logit(np.array(x[3], np.float32)) for x in top], axis=1)
        Xt = np.stack([to_logit(x[6]) - to_logit(np.array(x[3], np.float32)) for x in top], axis=1)

        Xd_flat = torch.from_numpy(Xd.reshape(len(dg), len(top), -1)).to(device)
        dg_t = torch.from_numpy(dg.reshape(len(dg), -1)).to(device)
        bs_t = torch.tensor([-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30], device=device)
        best_sc, best_w, best_b = -1.0, np.ones(len(top), np.float32), 0.0
        for _ in range(3):
            for i in range(len(top)):
                for cand in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0):
                    wt = best_w.copy(); wt[i] = cand
                    if wt.sum() <= 0: continue
                    wt_t = torch.from_numpy(wt / wt.sum()).to(device).view(1, -1, 1)
                    acc = (Xd_flat * wt_t).sum(dim=1)
                    for bi in range(len(bs_t)):
                        p = acc >= bs_t[bi]
                        i1 = (p & dg_t).sum(1).float(); u1 = (p | dg_t).sum(1).float()
                        i0 = ((~p) & (~dg_t)).sum(1).float(); u0 = ((~p) | (~dg_t)).sum(1).float()
                        sc = float((0.5 * (torch.where(u1 > 0, i1 / u1.clamp(min=1), 1.0) + torch.where(u0 > 0, i0 / u0.clamp(min=1), 1.0))).mean().item())
                        if sc > best_sc: best_sc, best_w, best_b = sc, wt.copy(), float(bs_t[bi].item())

        d_nn = np.tensordot(best_w / best_w.sum(), Xd, axes=([0], [1])) - best_b
        h_nn = np.tensordot(best_w / best_w.sum(), Xh, axes=([0], [1])) - best_b
        t_nn = np.tensordot(best_w / best_w.sum(), Xt, axes=([0], [1])) - best_b
        dl_sc = float(fast_miou_vec(d_nn >= 0.0, dg).mean())
        hl_sc = float(fast_miou_vec(h_nn >= 0.0, hg).mean())
        stage = "LINEAR"
        print(f"  [LINEAR] dev={dl_sc:.4f} hold={hl_sc:.4f}", flush=True)

        oof_m, hold_m, test_m = nested_cv_stack(Xd, dg.astype(np.float32), dL, Xh, Xt, best_w, best_b, device)
        dm_sc = float(fast_miou_vec(oof_m >= 0.0, dg).mean())
        hm_sc = float(fast_miou_vec(hold_m >= 0.0, hg).mean())
        print(f"  [METASTACK] dev={dm_sc:.4f} ({dm_sc-dl_sc:+.4f}) hold={hm_sc:.4f} ({hm_sc-hl_sc:+.4f})", flush=True)
        if dm_sc >= dl_sc + 0.001:
            d_nn, h_nn, t_nn = oof_m, hold_m, test_m
            stage = "METASTACK"

        # Strict Nested DEV CV for L/S Split-Threshold!
        b_sp, cv_sp, tL, tS = fit_split_thr_cv(d_nn, dg)
        print(f"  [SPLIT-THR] nested-dev-CV={cv_sp:.4f} ({cv_sp-b_sp:+.4f}) tL={tL:+.2f} tS={tS:+.2f}", flush=True)
        if cv_sp > b_sp + 0.0005:
            d_nn = shift_logits_by_split(d_nn, tL, tS)
            h_nn = shift_logits_by_split(h_nn, tL, tS)
            t_nn = shift_logits_by_split(t_nn, tL, tS)
            stage += f"+SPLIT({tL:+.1f},{tS:+.1f})"

        d_pred, h_pred, t_pred = d_nn >= 0.0, h_nn >= 0.0, t_nn >= 0.0
        nn_dev_sc = float(fast_miou_vec(d_pred, dg).mean())

        if method in ("pixelseal", "trustmark"):
            rect_cands = [
                ("rect_nnref", Path(f"runs/rect/{method}_train_nnref.npz"), Path(f"runs/rect/{method}_test_nnref.npz")),
                ("rect_v2", Path(f"runs/rect/{method}_train.npz"), Path(f"runs/rect/{method}_test.npz")),
            ]
            best_rect = (nn_dev_sc + 0.001, None)
            for tag, tr_p, te_p in rect_cands:
                if not (tr_p.exists() and te_p.exists()): continue
                rd_tr = load_rect_dict(tr_p)
                d_rec = [rd_tr[n] for n in dn]
                d_bm = np.stack([box_to_mask128(r["box"]) for r in d_rec])
                cv_sc, full_sc, cfg = fit_rect_cv(d_nn, dg, dL, d_rec, d_bm)
                print(f"  [RECT {tag}] nested-dev-CV={cv_sc:.4f} ({cv_sc-nn_dev_sc:+.4f}) cfg={cfg}", flush=True)
                if cfg is not None and cv_sc > best_rect[0]:
                    best_rect = (cv_sc, (tag, tr_p, te_p, cfg))
            if best_rect[1] is not None:
                tag, tr_p, te_p, (md, fa, ac, gn, al) = best_rect[1]
                rd_tr, rd_te = load_rect_dict(tr_p), load_rect_dict(te_p)
                d_rec, h_rec, t_rec = [rd_tr[n] for n in dn], [rd_tr[n] for n in hn], [rd_te[n] for n in tn]
                d_bm, h_bm, t_bm = np.stack([box_to_mask128(r["box"]) for r in d_rec]), np.stack([box_to_mask128(r["box"]) for r in h_rec]), np.stack([box_to_mask128(r["box"]) for r in t_rec])
                d_pred, _ = apply_rect_override(d_nn, d_rec, d_bm, fa, ac, gn, md, al)
                h_pred, _ = apply_rect_override(h_nn, h_rec, h_bm, fa, ac, gn, md, al)
                t_pred, _ = apply_rect_override(t_nn, t_rec, t_bm, fa, ac, gn, md, al)
                stage += f"+{tag}"

        d_off = np.array([miou_image(d_pred[i], dr[i]["mask128"]) for i in range(len(dr))])
        h_off = np.array([miou_image(h_pred[i], hr[i]["mask128"]) for i in range(len(hr))])
        print(f"  *** FINAL {method.upper()} ({stage}) *** DEV={d_off.mean():.4f} (L {d_off[dL].mean():.4f} S {d_off[~dL].mean():.4f}) | HOLD={h_off.mean():.4f} (L {h_off[hL].mean():.4f} S {h_off[~hL].mean():.4f})", flush=True)
        summary[method] = {"dev": float(d_off.mean()), "hold": float(h_off.mean()), "stage": stage}
        for i, n in enumerate(tn): test_masks[n] = t_pred[i].astype(np.uint8)

    mean_dev = float(np.mean([summary[m]["dev"] for m in summary]))
    mean_hold = float(np.mean([summary[m]["hold"] for m in summary]))
    print(f"\nFINAL 100% CLEAN 3-METHOD DEV  OOF   : {mean_dev:.4f}")
    print(f"FINAL 100% CLEAN 3-METHOD HOLDOUT    : {mean_hold:.4f}")
    all_names = [r["image_name"] for r in test_rows]
    masks_u8 = torch.from_numpy(np.stack([test_masks[n] for n in all_names])).to(torch.uint8)
    out_p = Path("runs/submissions/sub_v4_fix01_split.pt")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_names": all_names, "masks": masks_u8}, out_p)
    qdir = Path.home() / "sbatch_v8/auto_sub_queue"
    qdir.mkdir(parents=True, exist_ok=True)
    dst = qdir / "sub_v4_fix01_split.pt"
    import shutil; shutil.copy2(out_p, dst)
    (qdir / "sub_v4_fix01_split.json").write_text(json.dumps({"file": dst.name, "dev_oof": mean_dev, "holdout": mean_hold}, indent=2))
    print(f"QUEUED AUTO-SUBMIT: {dst} ({out_p.stat().st_size/1e6:.2f} MB, dev={mean_dev:.4f}, hold={mean_hold:.4f})", flush=True)

if __name__ == "__main__":
    main()
PYEOF
CUDA_VISIBLE_DEVICES=0 .venv/bin/python build_sub_v4_split.py
echo "=== V11_TM_S4_DONE ==="
