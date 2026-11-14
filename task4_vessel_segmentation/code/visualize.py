"""Qualitative figures: original / ground truth / probability map / prediction."""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vseg.data import RetinaSet, imread_u
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


def overlay(rgb, mask, color=(0, 255, 0), alpha=0.55):
    out = rgb.copy()
    m = mask.astype(bool)
    col = np.zeros_like(out)
    col[:, :] = color
    out[m] = (out[m] * (1 - alpha) + col[m] * alpha).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids", default="1,5,9,13")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "fig_predictions.png"))
    ap.add_argument("--tta", type=int, default=8)
    args = ap.parse_args()

    models = [load(p) for p in args.ckpts]
    ds = RetinaSet(args.split, with_labels=(args.split == "train"))
    want = [int(v) for v in args.ids.split(",")]
    rows = []
    for wid in want:
        i = ds.ids.index(str(wid))
        acc = 0
        for m in models:
            acc = acc + predict_probs(m, ds.imgs[i], n_tta=args.tta, fov=None)
        p = acc / len(models) * ds.fovs[i]
        rgb = ds.raws[i]
        pred = (p > args.thr).astype(np.uint8)
        cells = [rgb, overlay(rgb, pred)]
        if args.split == "train":
            gt = ds.labels[i].astype(np.uint8)
            cells.insert(1, overlay(rgb, gt, color=(255, 0, 0)))
            cells.append((np.stack([pred * 255] * 3, -1)))
            d = dice(pred, gt)
            cells[0] = cv2.copyMakeBorder(cells[0], 0, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(cells[0], f"Dice {d:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 0), 2)
        else:
            cells.append((np.stack([pred * 255] * 3, -1)))
        rows.append(np.concatenate(cells, axis=1))
        print("id", wid, "vessel %.3f%%" % (100 * pred.mean()))

    grid = np.concatenate(rows, axis=0)
    h, w = grid.shape[:2]
    if w > 1500:
        s = 1500 / w
        grid = cv2.resize(grid, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print("saved", args.out)


if __name__ == "__main__":
    main()
