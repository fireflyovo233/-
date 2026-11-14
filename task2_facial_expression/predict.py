"""单张/批量图片推理：集成 5 折权重 + 水平翻转 TTA。

用法:
    python predict.py --image path/to/face.jpg
    python predict.py --dir path/to/folder
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image

from fer_data import CLASSES, IMG_SIZE, SUBMIT_LABEL, to_tensor
from model import FERNet

BASE = os.path.dirname(os.path.abspath(__file__))
MEAN, STD = 0.5070, 0.2551  # 训练集统计量，见 analyze_data.py 输出


def load_ensemble(runs: str, width: int = 32):
    models = []
    for p in sorted(glob.glob(os.path.join(runs, "fold*.pt"))):
        m = FERNet(len(CLASSES), width, 0.3)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.eval()
        models.append(m)
    if not models:
        raise FileNotFoundError(f"{runs} 下没有 fold*.pt，请先训练")
    return models


@torch.no_grad()
def predict_paths(models, paths: list[str]) -> np.ndarray:
    x = np.stack([
        np.asarray(Image.open(p).convert("L").resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)
        for p in paths
    ])
    t = to_tensor(x, MEAN, STD)
    prob = torch.zeros(len(paths), len(CLASSES))
    for m in models:
        prob += (m(t).softmax(1) + m(torch.flip(t, dims=[3])).softmax(1)) / 2
    return (prob / len(models)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--dir")
    ap.add_argument("--runs", default=os.path.join(BASE, "runs"))
    ap.add_argument("--conf", type=float, default=0.0, help="只输出置信度高于该值的样本")
    args = ap.parse_args()

    if args.image:
        paths = [args.image]
    elif args.dir:
        paths = sorted(glob.glob(os.path.join(args.dir, "*.jpg")))
    else:
        ap.error("需要 --image 或 --dir")

    models = load_ensemble(args.runs)
    prob = predict_paths(models, paths)
    for p, pr in zip(paths, prob):
        k = int(pr.argmax())
        if pr[k] < args.conf:
            continue
        print(f"{os.path.basename(p):<20} -> {CLASSES[k]:<9} (提交标签 {int(SUBMIT_LABEL[k])}) p={pr[k]:.3f}  "
              + " ".join(f"{c}:{v:.2f}" for c, v in zip(CLASSES, pr)))


if __name__ == "__main__":
    main()
