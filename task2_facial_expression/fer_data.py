"""FER2013 风格表情数据集读取（48x48 灰度，6 类）。

标签映射采用目录名的字典序：[Angry, Fear, Happy, Neutral, Sad, Surprise] -> 0..5，
与 submission 模板的 0..5 整数编号对应。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

CLASSES = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 48


@dataclass
class Split:
    x: np.ndarray          # (N, 48, 48) uint8
    y: np.ndarray | None   # (N,) int64
    names: list[str]       # 文件名（含扩展名）


def _load_images(paths: list[str]) -> np.ndarray:
    out = np.empty((len(paths), IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            out[i] = np.asarray(im.convert("L"), dtype=np.uint8)
    return out


def load_train(root: str, cache: str | None = None) -> Split:
    """读取训练集；cache 为 .npz 路径时复用，避免重复解码 2.8 万张图。"""
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return Split(z["x"], z["y"], list(z["names"]))

    names, labels = [], []
    for cls in CLASSES:
        d = os.path.join(root, "train", cls)
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                names.append(os.path.join(d, fn))
                labels.append(CLASS_TO_IDX[cls])

    x = _load_images(names)
    y = np.asarray(labels, dtype=np.int64)
    short = [os.path.basename(n) for n in names]
    if cache:
        np.savez_compressed(cache, x=x, y=y, names=np.asarray(short, dtype=object))
    return Split(x, y, short)


def load_test(root: str, cache: str | None = None) -> Split:
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return Split(z["x"], None, list(z["names"]))

    d = os.path.join(root, "test")
    names = sorted(
        fn for fn in os.listdir(d) if fn.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    x = _load_images([os.path.join(d, n) for n in names])
    if cache:
        np.savez_compressed(cache, x=x, names=np.asarray(names, dtype=object))
    return Split(x, None, names)


def to_tensor(x: np.ndarray, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """(N,48,48) uint8 -> (N,1,48,48) float32 归一化张量。"""
    t = torch.from_numpy(np.ascontiguousarray(x)).float().div_(255.0)
    t = (t - mean) / std
    return t.unsqueeze(1)


def dataset_stats(x: np.ndarray) -> tuple[float, float]:
    m = float(x.mean()) / 255.0
    s = float(x.std()) / 255.0
    return m, s


# --- 提交编码 ---
# 模型内部顺序（目录字典序）: 0 Angry, 1 Fear, 2 Happy, 3 Neutral, 4 Sad, 5 Surprise
# Kaggle 要求: 0 愤怒, 1 恐惧, 2 快乐, 3 悲伤, 4 惊讶, 5 中性
# 即内部索引 -> 提交标签 = [0, 1, 2, 5, 3, 4]
SUBMIT_LABEL = np.array([0, 1, 2, 5, 3, 4], dtype=np.int64)


def to_submit(idx):
    """把模型内部类别索引转换为提交标签。"""
    return SUBMIT_LABEL[np.asarray(idx)]
