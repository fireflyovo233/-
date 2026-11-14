# -*- coding: utf-8 -*-
"""Honest out-of-fold calibration for the *final* ensemble.

For every training image we only use models that never saw it, average their
probability maps exactly like the delivered ensemble does, then sweep the
binarisation threshold.

Checkpoint priority per fold (newest / strongest first):
    f{i}v3  (2000 steps, w20)  ->  f{i}v2  (1000 steps, w20)  ->  fold{i} (300 steps, w16)

Output: outputs/thr_ens.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vseg.data import RetinaSet
from vseg.infer import dice, predict_probs

ROOT = r"D:/机器学习课设/task4"
# (canonical fold name, validation indices); indices are 0-based, image id = index + 1
FOLDS = [("fold0", [0, 5, 10, 15]), ("fold1", [2, 7, 13, 18]),
         ("fold2", [3, 8, 14, 19]), ("fold3", [1, 6, 11, 16])]
CANDIDATES = {
    "fold0": ["f0v3_ema", "f0v3_best", "f0v2_ema", "f0v2_best", "fold0_ema"],
    "fold1": ["f1v3_ema", "f1v3_best", "f1v2_ema", "f1v2_best", "fold1_ema"],
    "fold2": ["f2v3_ema", "f2v3_best", "f2v2_ema", "f2v2_best", "fold2_ema"],
    "fold3": ["f3v3_ema", "f3v3_best", "f3v2_ema", "f3v2_best", "fold3_ema"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--per-fold", type=int, default=2,
                    help="how many checkpoints per fold (strongest first)")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "thr_ens.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from predict import load_models

    ds = RetinaSet("train")
    probs, used, picked = {}, {}, {}
    for name, vid in FOLDS:
        cks = []
        for stem in CANDIDATES[name]:
            p = os.path.join(ROOT, "models", f"{stem}.pt")
            if os.path.exists(p) and p not in cks:
                cks.append(p)
        picked[name] = cks[:args.per_fold]
        if not picked[name]:
            print("skip", name, "(no checkpoint)")
            continue
        ms = [load_models([c])[0] for c in picked[name]]
        for i in vid:
            acc = np.zeros(ds.imgs[i].shape[:2], dtype=np.float32)
            for m in ms:
                acc += predict_probs(m, ds.imgs[i], n_tta=args.tta, fov=None) * ds.fovs[i]
            probs[i] = (acc / len(ms)).astype(np.float32)
            used[i] = [os.path.basename(c) for c in picked[name]]
        print(f"{name}: {[ds.ids[i] for i in vid]} <- {[os.path.basename(c) for c in picked[name]]}",
              flush=True)

    rows = []
    for thr in np.arange(0.30, 0.81, 0.025):
        d = [dice((probs[i] > thr).astype(np.uint8), ds.labels[i]) for i in probs]
        cov = [100 * float((probs[i] > thr).mean()) for i in probs]
        rows.append({"thr": round(float(thr), 3), "dice": float(np.mean(d)),
                     "cov": float(np.mean(cov)), "cov_std": float(np.std(cov)),
                     "cov_min": float(np.min(cov)), "cov_max": float(np.max(cov))})
    best = max(rows, key=lambda r: r["dice"])
    gt_cov = 100 * float(np.mean([l.mean() for l in ds.labels]))

    # Risk-hedged pick: the Dice curve is flat, so prefer a slightly lower threshold
    # (higher coverage). Under-segmentation was the observed failure mode online.
    hedge = min(rows, key=lambda r: abs(r["cov"] - gt_cov * 1.08))
    use = hedge if hedge["dice"] >= best["dice"] - 0.004 else best

    print(f"{'thr':>6} {'dice':>7} {'cov%':>7} {'covsd':>6}")
    for r in rows:
        mark = ""
        if r["thr"] == best["thr"]:
            mark += "  <- max dice"
        if r["thr"] == hedge["thr"]:
            mark += "  <- coverage hedged"
        if r["thr"] == use["thr"]:
            mark += "  ** adopted"
        print(f"{r['thr']:>6.3f} {r['dice']:>7.4f} {r['cov']:>7.2f} {r['cov_std']:>6.2f}{mark}")
    print(f"gt coverage {gt_cov:.2f}% | max-dice {best['dice']:.4f}@{best['thr']} | adopted {use['thr']}")

    per = []
    for i in sorted(probs):
        m = (probs[i] > use["thr"]).astype(np.uint8)
        per.append({"id": ds.ids[i], "dice": round(float(dice(m, ds.labels[i])), 4),
                    "models": used[i],
                    "cov_pred": round(100 * float(m.mean()), 2),
                    "cov_gt": round(100 * float(ds.labels[i].mean()), 2)})
    per.sort(key=lambda r: int(r["id"]))
    mean_use = float(np.mean([p["dice"] for p in per]))

    json.dump({"tta": args.tta, "n_images": len(probs), "gt_cov": gt_cov,
               "best": best, "hedged": hedge, "adopted": use,
               "dice_at_adopted": mean_use,
               "grid": rows, "per_image": per,
               "checkpoints": {k: [os.path.basename(c) for c in v] for k, v in picked.items()}},
              open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"wrote {args.out}  (dice at adopted thr = {mean_use:.4f})")


if __name__ == "__main__":
    main()
