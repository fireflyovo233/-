# -*- coding: utf-8 -*-
"""Out-of-fold threshold calibration.

Each fold model never saw its validation images, so concatenating the four folds
gives an honest 20-image estimate of the train distribution. We sweep the
binarisation threshold there and write the recommendation to outputs/thr.json.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vseg.data import RetinaSet
from vseg.infer import dice

ROOT = r"D:/机器学习课设/task4"
# NOTE: image-1-based ids, indices = id-1
FOLDS = [("fold0", [0, 5, 10, 15]), ("fold1", [2, 7, 13, 18]),
         ("fold2", [3, 8, 14, 19]), ("fold3", [1, 6, 11, 16])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "thr.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from predict import load_models, ensemble_probs

    ds = RetinaSet("train")
    probs, gts, where = {}, {}, {}
    for name, vid in FOLDS:
        ck = os.path.join(ROOT, "models", f"{name}_ema.pt")
        if not os.path.exists(ck):
            ck = os.path.join(ROOT, "models", f"{name}_last.pt")
        if not os.path.exists(ck):
            print("missing", ck, "-> skip")
            continue
        ms = load_models([ck])
        for i in vid:
            probs[i] = ensemble_probs(ms, ds.imgs[i], ds.fovs[i], args.tta)
            gts[i] = ds.labels[i]
            where[i] = name
        print(f"{name}: oof images {[ds.ids[i] for i in vid]} done", flush=True)

    rows = []
    for thr in np.arange(0.30, 0.86, 0.025):
        d = [dice((probs[i] > thr).astype(np.uint8), gts[i]) for i in probs]
        rows.append({"thr": round(float(thr), 3), "dice": float(np.mean(d))})
    best = max(rows, key=lambda r: r["dice"])
    for r in rows:
        print(f"thr {r['thr']:.3f} dice {r['dice']:.4f}")
    print("BEST", best)

    per = [{"id": ds.ids[i], "fold": where[i], "dice": float(dice((probs[i] > best["thr"]).astype(np.uint8), gts[i])),
            "vessel_pred_pct": float(100 * (probs[i] > best["thr"]).mean()),
            "vessel_gt_pct": float(100 * gts[i].mean())} for i in sorted(probs)]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"tta": args.tta, "best": best, "grid": rows, "per_image": per,
                   "n_images": len(probs)}, f, indent=1, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
