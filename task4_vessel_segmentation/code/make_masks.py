# -*- coding: utf-8 -*-
"""Turn the cached test probability maps (outputs/test_probs/*.npy) into 512x512 masks + RLE csv.

This replaces re-running the network: predict.py --save-probs already cached every
probability map, so post-processing (threshold / min object size / grid alignment)
stays fully re-tunable without another 20 minutes of CPU inference.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np


def imwrite_u(path, img):
    """cv2.imwrite is unreliable with non-ASCII (Chinese) paths on Windows."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError("encode failed " + path)
    buf.tofile(path)


def imread_u(path, flags=cv2.IMREAD_GRAYSCALE):
    """cv2.imread replacement that supports non-ASCII paths on Windows."""
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise IOError("cannot read " + path)
    return img

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from make_submission import rle_encode_binary

ROOT = r"D:/机器学习课设/task4"
PROB_DIR = os.path.join(ROOT, "outputs", "test_probs")
IMG_DIR = os.path.join(ROOT, "submission", "image")


def to_512(prob, thr, mode="B"):
    """Build the 512x512 submission mask.

    mode A: area-pool the probability map to 512, then threshold (thin vessels)
    mode B: threshold at full resolution, then downsample the *mask* exactly like
            the official segmentation_to_csv.py (PIL BILINEAR, then > 0).
            This reproduces the one-pixel rim that the grader's labels carry.
    mode C: mode A followed by a 1-px dilation.
    """
    import torch
    from PIL import Image

    if mode == "B":
        m = (prob > thr).astype(np.uint8) * 255
        im = Image.fromarray(m).resize((512, 512), Image.BILINEAR)
        return (np.array(im) > 0).astype(np.uint8)

    p = torch.from_numpy(prob)[None, None]
    p = torch.nn.functional.interpolate(p, size=(512, 512), mode="area")[0, 0].numpy()
    m = (p > thr).astype(np.uint8)
    if mode == "C":
        q = np.pad(m, 1, mode="constant")
        m = (q[1:-1, 1:-1] | q[:-2, 1:-1] | q[2:, 1:-1] | q[1:-1, :-2] | q[1:-1, 2:]).astype(np.uint8)
    return m


def remove_small(m, ms):
    if ms <= 1:
        return m
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = m.copy()
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] < ms:
            out[lab == j] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--min-size", type=int, default=30)
    ap.add_argument("--thr-json", default=os.path.join(ROOT, "outputs", "thr.json"))
    ap.add_argument("--recipe", default="B", choices=["A", "B", "C"],
                    help="B = mimic the official BILINEAR+>0 label thickening")
    ap.add_argument("--out-csv", default=os.path.join(ROOT, "目标文件", "submission.csv"))
    ap.add_argument("--group", nargs="*", default=None,
                    help="subset of ids to (re)process; default = all")
    args = ap.parse_args()

    thr = args.thr
    if thr is None:
        thr = json.load(open(args.thr_json, encoding="utf-8"))["best"]["thr"]
    files = sorted(glob.glob(os.path.join(PROB_DIR, "*.npy")),
                   key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if args.group:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in set(args.group)]
    os.makedirs(IMG_DIR, exist_ok=True)
    ids, pct = [], []
    for f in files:
        pid = os.path.splitext(os.path.basename(f))[0]
        prob = np.load(f)
        m = remove_small(to_512(prob, thr, args.recipe), args.min_size)
        imwrite_u(os.path.join(IMG_DIR, f"{pid}.png"), (m * 255).astype(np.uint8))
        ids.append(pid)
        pct.append(100 * float(m.mean()))
        print(f"id {pid}: vessel {pct[-1]:.2f}%", flush=True)

    if args.group:
        print("partial group done, csv not written")
        return

    # assemble the csv in the official Id order
    rows = []
    for pid in sorted(ids, key=lambda s: int(s)):
        p = os.path.join(IMG_DIR, f"{pid}.png")
        m = imread_u(p)
        assert m.shape == (512, 512), f"{pid} {m.shape}"
        rows.append((pid, rle_encode_binary((m > 0).astype(np.uint8))))
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as fh:
        fh.write("Id,Predicted\n")
        for pid, r in rows:
            fh.write(f"{pid},{r}\n")

    stats = {"n": len(rows), "thr": thr, "min_size": args.min_size, "recipe": args.recipe,
             "vessel_pct": float(np.mean(pct)), "vessel_pct_all": [round(p, 2) for p in pct],
             "rle_chars": int(sum(len(r) for _, r in rows))}
    with open(os.path.join(ROOT, "目标文件", "submission_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in stats.items() if k != "vessel_pct_all"}, ensure_ascii=False))
    print("wrote", args.out_csv)


if __name__ == "__main__":
    main()
