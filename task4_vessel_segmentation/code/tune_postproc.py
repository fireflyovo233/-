# -*- coding: utf-8 -*-
"""Search post-processing hyper-parameters on the held-out validation images.

Searches (probability threshold x minimum object size) and reports the best combo
under two objectives: plain Dice and a topology-aware score (Dice + clDice)/2.
Results are written to outputs/postproc.json and reused by predict.py / report.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vseg.data import RetinaSet
from vseg.infer import postprocess


def cl_dice(pred, gt):
    """Simplified centre-line Dice via skeletonisation of both masks."""
    from skimage.morphology import skeletonize
    sp = skeletonize(pred.astype(bool))
    sg = skeletonize(gt.astype(bool))
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    # precision: how much of the predicted skeleton lies on the GT mask
    tprec = (sp & gt.astype(bool)).sum() / (sp.sum() + 1e-8)
    # sensitivity: how much of the GT skeleton is covered by the prediction
    tsens = (sg & pred.astype(bool)).sum() / (sg.sum() + 1e-8)
    return float(2 * tprec * tsens / (tprec + tsens + 1e-8))


def dice(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    return float(2 * (p & g).sum() / (p.sum() + g.sum() + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--val", type=str, required=True)
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--out", type=str, default="outputs/postproc.json")
    args = ap.parse_args()

    from predict import load_models, ensemble_probs

    torch.set_num_threads(6)
    models = load_models(args.ckpts)
    ds = RetinaSet("train")
    vid = [int(v) for v in args.val.split(",")]

    prob_cache = {}
    for i in vid:
        prob_cache[i] = ensemble_probs(models, ds.imgs[i], ds.fovs[i], args.tta)
        print("probs done", ds.ids[i], flush=True)

    # per-image skeleton / area stats are computed once and reused for every (thr, ms)
    stats = {}
    for i in vid:
        g = ds.labels[i].astype(bool)
        sk = skeletonize(g)
        stats[i] = {"g": g, "sk": sk, "gsum": int(g.sum()), "sksum": int(sk.sum())}

    rows = []
    for thr, ms in itertools.product(np.arange(0.30, 0.71, 0.05), [0, 20, 50, 100, 200]):
        ds_, cl_, tp_, fp_, fn_ = [], [], 0, 0, 0
        for i in vid:
            m = postprocess(prob_cache[i], ds.fovs[i], float(thr), int(ms)).astype(bool)
            st = stats[i]
            g = st["g"]
            ds_.append(dice(m, g))
            # clDice: precision along predicted skeleton, sensitivity along GT skeleton
            sp = skeletonize(m)
            if sp.sum() == 0 or st["sksum"] == 0:
                cl_.append(0.0)
            else:
                tprec = (sp & g).sum() / (sp.sum() + 1e-8)
                tsens = (st["sk"] & m).sum() / (st["sksum"] + 1e-8)
                cl_.append(float(2 * tprec * tsens / (tprec + tsens + 1e-8)))
            tp_ += int((m & g).sum())
            fp_ += int((m & ~g).sum())
            fn_ += int((~m & g).sum())
        prec = tp_ / (tp_ + fp_ + 1e-8)
        rec = tp_ / (tp_ + fn_ + 1e-8)
        rows.append({"thr": round(float(thr), 3), "min_size": int(ms),
                     "dice": float(np.mean(ds_)), "cldice": float(np.mean(cl_)),
                     "precision": float(prec), "recall": float(rec),
                     "f1": float(2 * prec * rec / (prec + rec + 1e-8))})

    rows.sort(key=lambda r: -r["dice"])
    best_dice = rows[0]
    best_topo = sorted(rows, key=lambda r: -(r["dice"] + r["cldice"]) / 2)[0]
    for r in rows[:8]:
        print("thr %.2f min %3d dice %.4f cldice %.4f P %.3f R %.3f"
              % (r["thr"], r["min_size"], r["dice"], r["cldice"], r["precision"], r["recall"]))
    print("BEST dice:", best_dice)
    print("BEST topo:", best_topo)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"ckpts": args.ckpts, "val": vid, "tta": args.tta,
                   "best_dice": best_dice, "best_topo": best_topo,
                   "grid": sorted(rows, key=lambda r: (r["thr"], r["min_size"]))}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
