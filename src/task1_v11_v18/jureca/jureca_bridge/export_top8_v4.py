import json, pathlib, numpy as np, sys
sys.path.insert(0, "/p/scratch/training2666/qin7/task1")
from build_sub_v3_clean import is_clean_dir, to_logit

roots = [
    pathlib.Path("/p/scratch/training2666/qin7/task1/runs/fusion"),
    pathlib.Path("/p/scratch/training2666/ding7/claude_task1/runs/fusion"),
]
out = {}

for method in ("pixelseal", "trustmark", "stegastamp"):
    cands = []
    seen = set()
    for fdir in roots:
        if not fdir.exists(): continue
        for d in sorted(fdir.iterdir()):
            if not d.is_dir() or not d.name.endswith(f"_{method}") or not is_clean_dir(d): continue
            if d.name in seen: continue
            if not all((d / f).exists() for f in ("dev.npz", "holdout.npz", "test.npz")): continue
            dz = np.load(d / "dev.npz", allow_pickle=True)
            if bool(dz["dev_in_sample"]) if "dev_in_sample" in dz else True: continue
            hz = np.load(d / "holdout.npz", allow_pickle=True)
            tz = np.load(d / "test.npz", allow_pickle=True)
            dn = dz["image_names"].tolist(); hn = hz["image_names"].tolist(); tn = tz["image_names"].tolist()
            if len(dn) != 136 or len(hn) != 34 or len(tn) != 340: continue
            seen.add(d.name)
            thr = float(dz["threshold"]) if "threshold" in dz else 0.5
            thr = min(max(thr, 0.05), 0.95)
            off = float(np.log(thr / (1.0 - thr)))
            dlg = to_logit(dz["response"].astype(np.float32)) - off
            hlg = to_logit(hz["response"].astype(np.float32)) - off
            tlg = to_logit(tz["response"].astype(np.float32)) - off
            cands.append((d.name, dn, hn, tn, dlg, hlg, tlg))
    print(f"[{method}] exporting ALL {len(cands)} clean honest-OOF models!", flush=True)
    dn, hn, tn = cands[0][1], cands[0][2], cands[0][3]
    def align(names_src, arr_src, names_dst):
        idx = {n: i for i, n in enumerate(names_src)}
        return arr_src[[idx[n] for n in names_dst]]
    out[f"{method}_names_m"] = np.array([x[0] for x in cands])
    out[f"{method}_dev_names"] = np.array(dn)
    out[f"{method}_hold_names"] = np.array(hn)
    out[f"{method}_test_names"] = np.array(tn)
    out[f"{method}_X_dev"] = np.stack([align(x[1], x[4], dn) for x in cands], axis=1).astype(np.float16)
    out[f"{method}_X_hold"] = np.stack([align(x[2], x[5], hn) for x in cands], axis=1).astype(np.float16)
    out[f"{method}_X_test"] = np.stack([align(x[3], x[6], tn) for x in cands], axis=1).astype(np.float16)

out_p = "/p/home/jusers/qin7/jureca/jureca_bridge/ens_top8_v7_fix01.npz"
np.savez_compressed(out_p, **out)
print(f"SAVED {out_p} ({pathlib.Path(out_p).stat().st_size/1e6:.1f} MB)", flush=True)
