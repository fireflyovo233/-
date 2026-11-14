"""Ablation on held-out validation images: TTA / ensemble / FOV / post-processing / threshold."""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vseg.data import RetinaSet
from vseg.model import ResUNet
from vseg.infer import predict_probs, dice

ROOT = r"D:/机器学习课设/task4"


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    m = ResUNet(3, 1, a["width"], a["depth"], dilate_bottleneck=False)
    m.load_state_dict(ck["sd"])
    m.eval()
    return m


def dilate_fov(fov, k=7):
    return cv2.dilate((fov > 0.5).astype(np.uint8), np.ones((k, k), np.uint8)).astype(np.float32)


def rm_small(m, ms):
    if ms <= 1:
        return m
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    out = m.astype(np.uint8).copy()
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] < ms:
            out[lab == j] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "logs", "ablation.json"))
    args = ap.parse_args()

    ds = RetinaSet("train")
    vid = [int(v) for v in args.val.split(",")]
    models = [load(p) for p in args.ckpts]

    # cache probability maps: model index x tta
    cache = {}
    for mi, m in enumerate(models):
        for tta in (1, 4, 8):
            for i in vid:
                cache[(mi, tta, i)] = predict_probs(m, ds.imgs[i], n_tta=tta, fov=None)

    def ens_probs(sel, tta, i):
        p = np.mean([cache[(mi, tta, i)] for mi in sel], axis=0)
        return p

    rows = []
    cfgs = [("单模型 / 无TTA", [0], 1), ("单模型 / TTA-4", [0], 4), ("单模型 / TTA-8", [0], 8)]
    if len(models) > 1:
        cfgs += [("集成 / TTA-4", list(range(len(models))), 4),
                 ("集成 / TTA-8", list(range(len(models))), 8)]
    for name, sel, tta in cfgs:
        for fov_mode in ("none", "fov", "fov_dilate"):
            best = None
            for thr in np.arange(0.30, 0.71, 0.05):
                d = []
                for i in vid:
                    p = ens_probs(sel, tta, i)
                    if fov_mode == "fov":
                        p = p * ds.fovs[i]
                    elif fov_mode == "fov_dilate":
                        p = p * dilate_fov(ds.fovs[i], 9)
                    d.append(dice((p > thr).astype(np.uint8), ds.labels[i]))
                md = float(np.mean(d))
                if best is None or md > best[1]:
                    best = (float(thr), md)
            rows.append({"setting": name, "fov": fov_mode, "thr": best[0], "dice": best[1]})
            print("%-22s fov=%-10s thr=%.2f dice=%.4f" % (name, fov_mode, best[0], best[1]))

    # post-processing at the best overall setting
    best_row = max(rows, key=lambda r: r["dice"])
    print("best:", best_row)
    pp = []
    for ms in (0, 20, 40, 80):
        d = []
        for i in vid:
            p = ens_probs(list(range(len(models))), 8, i) * ds.fovs[i]
            d.append(dice(rm_small((p > best_row["thr"]).astype(np.uint8), ms), ds.labels[i]))
        pp.append({"min_size": ms, "dice": float(np.mean(d))})
        print("  min_size %3d -> %.4f" % (ms, np.mean(d)))

    json.dump({"rows": rows, "post": pp, "best": best_row},
              open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
