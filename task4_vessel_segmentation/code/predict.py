"""Predict test masks with an ensemble of checkpoints + TTA, then export RLE submission."""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vseg.data import RetinaSet, imread_u
from vseg.model import ResUNet
from vseg.infer import predict_probs, dice, postprocess

ROOT = r"D:/机器学习课设/task4"


def load_models(paths):
    models = []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        a = ck["args"]
        m = ResUNet(3, 1, a["width"], a["depth"], dilate_bottleneck=False)
        m.load_state_dict(ck["sd"])
        m.eval()
        models.append(m)
        print("loaded", os.path.basename(p))
    return models


def parse_scales(s):
    return tuple(float(x) for x in str(s).replace(" ", "").split(",") if x)


@torch.no_grad()
def ensemble_probs(models, img, fov, n_tta=8, scales=(1.0,)):
    acc = None
    for m in models:
        p = predict_probs(m, img, n_tta=n_tta, fov=None, scales=scales)
        acc = p if acc is None else acc + p
    p = acc / len(models)
    if fov is not None:
        p = p * fov
    return p


def to_512_binary(prob, thr):
    """Area-resize probability map to 512x512 (competition grid) then threshold."""
    p = torch.from_numpy(prob)[None, None]
    p = torch.nn.functional.interpolate(p, size=(512, 512), mode="area")[0, 0].numpy()
    return (p > thr).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-size", type=int, default=30)
    ap.add_argument("--val", type=str, default="")
    ap.add_argument("--out-dir", type=str, default=os.path.join(ROOT, "submission", "image"))
    ap.add_argument("--save-probs", action="store_true")
    ap.add_argument("--scales", type=str, default="1.0", help="e.g. 1.0 or 1.0,0.8,1.25")
    ap.add_argument("--max-hours", type=float, default=0.0, help="soft wall-clock budget")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    models = load_models(args.ckpts)

    if args.val:
        ds = RetinaSet("train")
        vid = [int(v) for v in args.val.split(",")]
        dscores = []
        for i in vid:
            p = ensemble_probs(models, ds.imgs[i], ds.fovs[i], args.tta, parse_scales(args.scales))
            m = postprocess(p, ds.fovs[i], args.thr, args.min_size)
            d = dice(m, ds.labels[i])
            dscores.append(d)
            print(f"val {ds.ids[i]}: dice {d:.4f}")
        print(f"VAL MEAN DICE (ens={len(models)}, tta={args.tta}): {np.mean(dscores):.4f}")

    import time
    t0 = time.time()
    stop_at = t0 + args.max_hours * 3600 if args.max_hours else None
    ds = RetinaSet("test", with_labels=False)
    prob_dir = os.path.join(ROOT, "outputs", "test_probs")
    if args.save_probs:
        os.makedirs(prob_dir, exist_ok=True)
    for i in range(len(ds)):
        p = ensemble_probs(models, ds.imgs[i], ds.fovs[i], args.tta, parse_scales(args.scales))
        if args.save_probs:
            np.save(os.path.join(prob_dir, f"{ds.ids[i]}.npy"), p)
        m512 = to_512_binary(p, args.thr)
        if args.min_size > 1:
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m512, 8)
            for j in range(1, n):
                if stats[j, cv2.CC_STAT_AREA] < args.min_size:
                    m512[lab == j] = 0
        png = (m512 * 255).astype(np.uint8)
        out = os.path.join(args.out_dir, f"{ds.ids[i]}.png")
        cv2.imwrite(out, png)
        print("saved", out, "vessel px %.3f%%" % (100 * m512.mean()), flush=True)
        if stop_at and time.time() > stop_at and i >= len(ds) // 2:
            print("wall-clock budget reached at image %d/%d" % (i + 1, len(ds)), flush=True)
            break


if __name__ == "__main__":
    main()
