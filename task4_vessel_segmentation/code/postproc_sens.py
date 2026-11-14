# -*- coding: utf-8 -*-
"""Post-processing sensitivity on the cached test probability maps.

There is no test label, so the diagnostic is distribution-based: the vessel
coverage of every test mask should stay inside the range observed on the 20
training ground truths (6.0%-11.6%, mean 8.6%). This tells us whether a
threshold / min-size setting produces plausible, non-degenerate masks.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from make_masks import to_512, remove_small
from vseg.data import RetinaSet

ROOT = r"D:/机器学习课设/task4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "postproc_sens.json"))
    args = ap.parse_args()

    ds = RetinaSet("test", with_labels=False)
    probs = {ds.ids.index(os.path.splitext(os.path.basename(f))[0]): np.load(f) * ds.fovs[ds.ids.index(os.path.splitext(os.path.basename(f))[0])]
             for f in glob.glob(os.path.join(ROOT, "outputs", "test_probs", "*.npy"))}
    m5 = {i: to_512(p, 0.5).astype(bool) for i, p in probs.items()}

    rows = []
    for thr in (0.475, 0.525, 0.575, 0.625, 0.675):
        cov = []
        for i, p in probs.items():
            m = to_512(p, thr)
            cov.append(100 * float(m.mean()))
        rows.append({"thr": thr, "min_size": 0, "mean_cov": float(np.mean(cov)),
                     "std_cov": float(np.std(cov)), "min_cov": float(np.min(cov)),
                     "max_cov": float(np.max(cov)),
                     "in_gt_range": int(sum(6.0 <= c <= 11.6 for c in cov))})
    print(f"{'thr':>6} {'min':>5} {'mean%':>7} {'std':>6} {'range':>14} {'inGT':>5}")
    for r in rows:
        print(f"{r['thr']:>6.3f} {r['min_size']:>5d} {r['mean_cov']:>7.2f} {r['std_cov']:>6.2f} "
              f"{r['min_cov']:>6.2f}-{r['max_cov']:<6.2f} {r['in_gt_range']:>4d}/20")

    # boundary shift between thresholds (fraction of pixels that flip)
    d = [100 * float((m5[i] != to_512(p, 0.625).astype(bool)).mean()) for i, p in probs.items()]
    print("flip rate 0.500 -> 0.625: mean %.2f%% of pixels" % np.mean(d))

    # how much area survives min-size filtering at the chosen threshold
    for ms in (0, 30, 100):
        keep = [100 * float(remove_small(to_512(p, 0.575), ms).sum() / max(1, to_512(p, 0.575).sum()))
                for p in probs.values()]
        print(f"min_size {ms:>3}: keeps {np.mean(keep):.2f}% of the thresholded area")

    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
