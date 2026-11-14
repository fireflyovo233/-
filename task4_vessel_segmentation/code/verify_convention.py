# -*- coding: utf-8 -*-
"""Back-test the label-convention hypothesis against the observed online score.

We submitted the v1 four-fold ensemble at threshold 0.575 and the grader returned
~0.73. This script rebuilds that exact out-of-fold prediction and scores it under
both label conventions:

    native   : GT downsampled with area pooling, > 0.5
    official : GT downsampled like segmentation_to_csv.py (BILINEAR then > 0)

Whichever convention reproduces ~0.73 explains the gap and tells us which
threshold to optimise for.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vseg.data import RetinaSet
from vseg.infer import dice, predict_probs
from oof_convention import gt_native, gt_official, prob512

ROOT = r"D:/机器学习课设/task4"
FOLDS = [("fold0", [0, 5, 10, 15]), ("fold1", [2, 7, 13, 18]),
         ("fold2", [3, 8, 14, 19]), ("fold3", [1, 6, 11, 16])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--thr", type=float, default=0.575)
    ap.add_argument("--cache", default=os.path.join(ROOT, "outputs", "oof_probs_v1"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "verify_convention.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from predict import load_models

    ds = RetinaSet("train")
    os.makedirs(args.cache, exist_ok=True)
    probs = {}
    for name, vid in FOLDS:
        ck = os.path.join(ROOT, "models", f"{name}_ema.pt")
        if not os.path.exists(ck):
            print("missing", ck)
            continue
        m = load_models([ck])[0]
        for i in vid:
            f = os.path.join(args.cache, f"{ds.ids[i]}.npy")
            if os.path.exists(f):
                probs[i] = np.load(f)
                continue
            probs[i] = predict_probs(m, ds.imgs[i], n_tta=args.tta, fov=ds.fovs[i])
            np.save(f, probs[i])
        print(name, "ok", [ds.ids[i] for i in vid], flush=True)

    p512 = {i: prob512(p) for i, p in probs.items()}
    out = {"thr": args.thr, "n_images": len(p512), "rows": []}
    print(f"{'thr':>6} {'native':>8} {'official':>9}")
    for thr in (0.45, 0.5, 0.525, 0.55, 0.575, 0.6, 0.65):
        dn = float(np.mean([dice((p512[i] > thr).astype(np.uint8), gt_native(ds.labels[i]))
                            for i in p512]))
        do = float(np.mean([dice((p512[i] > thr).astype(np.uint8), gt_official(ds.labels[i]))
                            for i in p512]))
        out["rows"].append({"thr": thr, "native": round(dn, 4), "official": round(do, 4)})
        print(f"{thr:>6.3f} {dn:>8.4f} {do:>9.4f}")

    at = [r for r in out["rows"] if abs(r["thr"] - args.thr) < 1e-6]
    if at:
        out["at_submitted_thr"] = at[0]
        print(f"\n提交时用的是 thr={args.thr}: native {at[0]['native']:.4f} / official {at[0]['official']:.4f}")
        print("线上实测约 0.73 —— 与哪一列吻合，评测端就用的哪种口径。")
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
