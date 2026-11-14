# -*- coding: utf-8 -*-
"""Threshold calibration under the *grader's* label convention.

The provided segmentation_to_csv.py turns a mask into RLE with

    image = Image.open(p).convert('L').resize((512, 512), BILINEAR)
    image[image > 0] = 1

i.e. after the bilinear resize every pixel that received *any* weight becomes
foreground. Applied to a label, this thickens vessels by roughly one pixel
(measured: 8.7% -> 12.9% coverage). If the grader builds its ground truth the
same way, a thin prediction loses ~6 Dice points -- which is exactly the gap
between our 0.79 self-estimate and the 0.73 online score.

This script therefore evaluates the out-of-fold predictions under BOTH
conventions, sweeps (threshold x optional 1-px dilation) and reports which
setting wins under the thickening convention.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vseg.data import RetinaSet
from vseg.infer import dice, predict_probs

ROOT = r"D:/机器学习课设/task4"
FOLDS = [("fold0", [0, 5, 10, 15], ["f0v3_ema", "f0v3_best"]),
         ("fold1", [2, 7, 13, 18], ["f1v3_ema", "f1v3_best"]),
         ("fold2", [3, 8, 14, 19], ["f2v3_ema", "f2v3_best"]),
         ("fold3", [1, 6, 11, 16], ["f3v3_ema", "f3v3_best"])]
SIDE = 512


def prob512(prob):
    t = torch.from_numpy(prob)[None, None]
    return torch.nn.functional.interpolate(t, size=(SIDE, SIDE), mode="area")[0, 0].numpy()


def gt_official(g):
    """Label converted exactly like the official helper (BILINEAR then >0)."""
    im = Image.fromarray((g * 255).astype(np.uint8)).resize((SIDE, SIDE), Image.BILINEAR)
    return (np.array(im) > 0).astype(np.uint8)


def gt_native(g):
    t = torch.from_numpy(g)[None, None]
    return (torch.nn.functional.interpolate(t, size=(SIDE, SIDE), mode="area")[0, 0].numpy() > 0.5).astype(np.uint8)


def dilate1(m):
    p = np.pad(m, 1, mode="constant")
    return (p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--cache", default=os.path.join(ROOT, "outputs", "oof_probs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "thr_official.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from predict import load_models

    ds = RetinaSet("train")
    os.makedirs(args.cache, exist_ok=True)

    probs = {}
    for name, vid, stems in FOLDS:
        cks = [os.path.join(ROOT, "models", f"{s}.pt") for s in stems]
        cks = [c for c in cks if os.path.exists(c)]
        ms = [load_models([c])[0] for c in cks]
        for i in vid:
            f = os.path.join(args.cache, f"{ds.ids[i]}.npy")
            if os.path.exists(f):
                probs[i] = np.load(f)
                continue
            acc = np.zeros(ds.imgs[i].shape[:2], dtype=np.float32)
            for m in ms:
                acc += predict_probs(m, ds.imgs[i], n_tta=args.tta, fov=None) * ds.fovs[i]
            probs[i] = (acc / len(ms)).astype(np.float32)
            np.save(f, probs[i])
        print(f"{name} ok ({len(cks)} models) {[ds.ids[i] for i in vid]}", flush=True)

    gts_off = {i: gt_official(ds.labels[i]) for i in probs}
    gts_nat = {i: gt_native(ds.labels[i]) for i in probs}
    p512 = {i: prob512(p) for i, p in probs.items()}

    rows = []
    for thr in np.arange(0.20, 0.76, 0.025):
        for dil in (0, 1):
            d_off, d_nat, cov = [], [], []
            for i in p512:
                m = (p512[i] > thr).astype(np.uint8)
                if dil:
                    m = dilate1(m)
                d_off.append(dice(m, gts_off[i]))
                d_nat.append(dice(m, gts_nat[i]))
                cov.append(100 * float(m.mean()))
            rows.append({"thr": round(float(thr), 3), "dilate": dil,
                         "dice_official": float(np.mean(d_off)),
                         "dice_native": float(np.mean(d_nat)),
                         "cov": float(np.mean(cov))})

    best_off = max(rows, key=lambda r: r["dice_official"])
    best_nat = max(rows, key=lambda r: r["dice_native"])
    print(f"{'thr':>6} {'dil':>3} {'dice(官方>0)':>13} {'dice(原始)':>11} {'cov%':>6}")
    for r in rows:
        if r["thr"] % 0.05 > 0.026 and r["dilate"] == 0:
            continue
        mark = ""
        if r is best_off:
            mark = "  ** 官方口径最优"
        print(f"{r['thr']:>6.3f} {r['dilate']:>3d} {r['dice_official']:>13.4f} "
              f"{r['dice_native']:>11.4f} {r['cov']:>6.2f}{mark}")
    print("best(官方口径):", best_off)
    print("best(原始口径):", best_nat)

    per = []
    for i in sorted(p512):
        m = (p512[i] > best_off["thr"]).astype(np.uint8)
        if best_off["dilate"]:
            m = dilate1(m)
        per.append({"id": ds.ids[i],
                    "dice_official": round(float(dice(m, gts_off[i])), 4),
                    "dice_native": round(float(dice(m, gts_nat[i])), 4),
                    "cov_pred": round(100 * float(m.mean()), 2),
                    "cov_gt_official": round(100 * float(gts_off[i].mean()), 2),
                    "cov_gt_native": round(100 * float(gts_nat[i].mean()), 2)})
    per.sort(key=lambda r: int(r["id"]))

    json.dump({"tta": args.tta, "n_images": len(p512),
               "best_official": best_off, "best_native": best_nat,
               "grid": rows, "per_image": per,
               "note": "dice_official = GT rebuilt with the official BILINEAR+>0 rule"},
              open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
