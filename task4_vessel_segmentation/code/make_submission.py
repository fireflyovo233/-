"""Convert predicted binary masks (512x512 PNG, vessel=255) to the competition RLE csv."""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd


def rle_encode_binary(mask01, one_based=True):
    """Row-major run-length encoding, identical to the official helper script."""
    flat = np.asarray(mask01).reshape(-1)
    dots = np.flatnonzero(flat == 1)
    runs = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            runs.extend((int(b) + (1 if one_based else 0), 0))
        runs[-1] += 1
        prev = b
    return " ".join(str(r) for r in runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default=r"D:/机器学习课设/task4/submission/image")
    ap.add_argument("--out", default=r"D:/机器学习课设/task4/目标文件/submission.csv")
    args = ap.parse_args()

    files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(".png")],
                   key=lambda s: int(os.path.splitext(s)[0]))
    ids, preds = [], []
    vpct = []
    for f in files:
        p = os.path.join(args.image_dir, f)
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        assert m.shape == (512, 512), f"{f} is {m.shape}, must be 512x512"
        preds.append(rle_encode_binary((m > 0).astype(np.uint8)))
        ids.append(os.path.splitext(f)[0])
        vpct.append(float((m > 0).mean()))
    df = pd.DataFrame({"Id": ids, "Predicted": preds})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.head(3).to_string())
    print("wrote", args.out, "rows", len(df))

    stats = {"n": len(df), "vessel_pct": float(np.mean(vpct) * 100),
             "rle_chars": int(sum(len(s) for s in preds)), "image_dir": args.image_dir}
    with open(os.path.join(os.path.dirname(args.out), "submission_stats.json"), "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(stats)

    # self-check: decode back and compare
    for i, s in enumerate(preds[:3]):
        pos = np.zeros(512 * 512, dtype=np.uint8)
        toks = s.split()
        for k in range(0, len(toks), 2):
            st, ln = int(toks[k]), int(toks[k + 1])
            pos[st - 1:st - 1 + ln] = 1
        m = cv2.imread(os.path.join(args.image_dir, files[i]), cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(pos.reshape(512, 512), (m > 0).astype(np.uint8)), "RLE roundtrip failed"
    print("RLE roundtrip check passed")


if __name__ == "__main__":
    main()
