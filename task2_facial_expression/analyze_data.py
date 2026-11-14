"""数据探索：类别分布、像素统计、样张网格，用于报告的数据分析章节。"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fer_data import CLASSES, load_test, load_train

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "根文件", "fer_data")
FIG = os.path.join(BASE, "..", "结果", "图")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    os.makedirs(FIG, exist_ok=True)
    tr = load_train(DATA, os.path.join(BASE, "runs", "cache", "train.npz"))
    te = load_test(DATA, os.path.join(BASE, "runs", "cache", "test.npz"))
    cnt = np.bincount(tr.y, minlength=len(CLASSES))

    print(f"{'类别':<10}{'训练':>8}{'占比':>10}")
    for i, c in enumerate(CLASSES):
        print(f"{c:<10}{cnt[i]:>8}{cnt[i]/cnt.sum()*100:>9.2f}%")
    print(f"合计训练 {len(tr.x)}，测试 {len(te.x)}，图像 {tr.x.shape[1]}x{tr.x.shape[2]} 灰度")
    print(f"像素均值 {tr.x.mean():.2f}/255，标准差 {tr.x.std():.2f}；"
          f"最大类/最小类 = {cnt.max()/cnt.min():.2f}")

    # 图 1：类别分布
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), dpi=150,
                             gridspec_kw={"width_ratios": [1.1, 1]})
    bars = axes[0].bar(CLASSES, cnt, color="#4c78a8")
    axes[0].bar_label(bars, fmt="%d", fontsize=9)
    axes[0].set_ylabel("样本数")
    axes[0].set_title("训练集类别分布（共 28275 张）")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].hist(tr.x.reshape(-1), bins=64, range=(0, 255), color="#f58518", alpha=0.85)
    axes[1].set_xlabel("灰度值"); axes[1].set_ylabel("像素数")
    axes[1].set_title("像素灰度直方图")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "data_overview.png"))
    plt.close(fig)

    # 图 2：每类样张
    fig, axes = plt.subplots(len(CLASSES), 8, figsize=(9, 6.4), dpi=150)
    rng = np.random.default_rng(0)
    for i, c in enumerate(CLASSES):
        idx = rng.choice(np.where(tr.y == i)[0], 8, replace=False)
        for a, j in zip(axes[i], idx):
            a.imshow(tr.x[j], cmap="gray", vmin=0, vmax=255)
            a.axis("off")
        axes[i, 0].text(-0.28, 0.5, c, transform=axes[i, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=10)
    fig.suptitle("训练集各类别样张（48×48 灰度）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "samples.png"))
    plt.close(fig)
    print("图已保存到", os.path.abspath(FIG))


if __name__ == "__main__":
    main()
