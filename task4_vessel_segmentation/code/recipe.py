# -*- coding: utf-8 -*-
"""Pick the mask-building recipe that matches the grader's thickened ground truth.

Three ways to produce a 512x512 mask that is as "thick" as the official labels:

  A  prob512 > t                       (lower the threshold -> fatter blobs)
  B  (prob > t) at full res -> PIL BILINEAR to 512 -> > 0
     exactly the transformation the official helper applies, so a thin mask
     gains the same one-pixel rim the labels gained
  C  prob512 > t then 1-px dilation

All are scored against GT rebuilt with the official rule. Cached OOF probability
maps are reused, so this is a pure post-processing sweep (no network forward).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vseg.data import RetinaSet
from vseg.infer import dice
from oof_convention import gt_official, prob512, dilate1, SIDE


def mask_B(prob, t):
    m = (prob > t).astype(np.uint8) * 255
    im = Image.fromarray(m).resize((SIDE, SIDE), Image.BILINEAR)
    return (np.array(im) > 0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=r"D:/机器学习课设/task4/outputs/oof_probs")
    ap.add_argument("--out", default=r"D:/机器学习课设/task4/outputs/recipe.json")
    args = ap.parse_args()

    ds = RetinaSet("train")
    probs = {}
    for f in glob.glob(os.path.join(args.cache, "*.npy")):
        pid = os.path.splitext(os.path.basename(f))[0]
        probs[ds.ids.index(pid)] = np.load(f)
    gts = {i: gt_official(ds.labels[i]) for i in probs}
    print("images", len(probs), "gt coverage %.2f%%" % (100 * np.mean([g.mean() for g in gts.values()])))

    rows = []
    for t in np.arange(0.10, 0.71, 0.05):
        for mode in ("A", "B", "C"):
            d, cov = [], []
            for i, p in probs.items():
                if mode == "A":
                    m = (prob512(p) > t).astype(np.uint8)
                elif mode == "B":
                    m = mask_B(p, t)
                else:
                    m = dilate1((prob512(p) > t).astype(np.uint8))
                d.append(dice(m, gts[i]))
                cov.append(100 * float(m.mean()))
            rows.append({"thr": round(float(t), 2), "mode": mode,
                         "dice": float(np.mean(d)), "cov": float(np.mean(cov))})

    best = max(rows, key=lambda r: r["dice"])
    for mode in ("A", "B", "C"):
        sub = [r for r in rows if r["mode"] == mode]
        b = max(sub, key=lambda r: r["dice"])
        print(f"mode {mode}: best dice {b['dice']:.4f} @ thr {b['thr']:.2f} (cov {b['cov']:.2f}%)")
    print(f"overall best: {best}")

    print(f"\n{'thr':>5} " + " ".join(f"{m:>7}" for m in ("A", "B", "C")) + "   covA  covB  covC")
    for t in sorted(set(r["thr"] for r in rows)):
        cells = {r["mode"]: r for r in rows if r["thr"] == t}
        print(f"{t:>5.2f} " + " ".join(f"{cells[m]['dice']:>7.4f}" for m in ("A", "B", "C")) +
              "   " + " ".join(f"{cells[m]['cov']:>5.2f}" for m in ("A", "B", "C")))

    per = []
    for i, p in probs.items():
        if best["mode"] == "A":
            m = (prob512(p) > best["thr"]).astype(np.uint8)
        elif best["mode"] == "B":
            m = mask_B(p, best["thr"])
        else:
            m = dilate1((prob512(p) > best["thr"]).astype(np.uint8))
        per.append({"id": ds.ids[i], "dice": round(float(dice(m, gts[i])), 4),
                    "cov": round(100 * float(m.mean()), 2),
                    "cov_gt": round(100 * float(gts[i].mean()), 2)})
    per.sort(key=lambda r: int(r["id"]))

    out = {"best": best, "per_image": per, "gt_cov": 100 * float(np.mean([g.mean() for g in gts.values()])),
           "grid": rows,
           "by_mode": {m: max([r for r in rows if r["mode"] == m], key=lambda r: r["dice"])
                       for m in ("A", "B", "C")}}
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
