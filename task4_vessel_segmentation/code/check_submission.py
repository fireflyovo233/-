# -*- coding: utf-8 -*-
"""Validate submission.csv against the official metric rules.

Checks, in the same order the grader does:
  1. the csv has exactly the 20 test ids, no NaN / empty prediction;
  2. every id RLE-decodes to a 512x512 mask;
  3. pairs are sorted, positive, non-overlapping (no duplicated pixels);
  4. decoded masks are re-encoded identically (round-trip);
and prints a per-image summary (vessel coverage, number of runs).
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = r"D:/机器学习课设/task4"


def decode(rle, side=512):
    out = np.zeros(side * side, dtype=np.uint8)
    toks = rle.split()
    if len(toks) % 2:
        raise ValueError("odd number of tokens")
    starts, lengths = [int(t) for t in toks[0::2]], [int(t) for t in toks[1::2]]
    prev_end = 0
    for s, l in zip(starts, lengths):
        if s <= 0 or l <= 0:
            raise ValueError(f"non-positive pair {s} {l}")
        if s <= prev_end:
            raise ValueError(f"overlap / unsorted at {s} (prev end {prev_end})")
        out[s - 1:s - 1 + l] = 1
        prev_end = s - 1 + l
    return out.reshape(side, side), len(starts)


def encode(mask01):
    flat = mask01.reshape(-1)
    dots = np.flatnonzero(flat == 1)
    runs, prev = [], -2
    for b in dots:
        if b > prev + 1:
            runs.extend((int(b) + 1, 0))
        runs[-1] += 1
        prev = b
    return " ".join(str(r) for r in runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "目标文件", "submission.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype={"Id": str})
    print("columns:", list(df.columns), "rows:", len(df))
    assert list(df.columns) == ["Id", "Predicted"], "column names must be Id,Predicted"
    assert len(df) == 20, "expected 20 rows"
    assert df["Predicted"].notna().all(), "NaN prediction found"

    rows = []
    for _, r in df.iterrows():
        m, nrun = decode(str(r["Predicted"]))
        re = encode(m)
        assert re == str(r["Predicted"]).strip(), f"round-trip mismatch for id {r['Id']}"
        rows.append((r["Id"], int(m.sum()), round(100 * m.mean(), 2), nrun,
                     round(float(m[256].mean()), 3)))

    print(f"{'Id':>3} {'px':>7} {'%':>6} {'runs':>6} {'row256覆盖':>9}")
    for x in rows:
        print(f"{x[0]:>3} {x[1]:>7d} {x[2]:>6.2f} {x[3]:>6d} {x[4]:>9.3f}")
    px = np.array([x[1] for x in rows])
    print("mean coverage %.2f%%, min %.2f%%, max %.2f%%"
          % (100 * px.mean() / 512 ** 2, 100 * px.min() / 512 ** 2, 100 * px.max() / 512 ** 2))
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
