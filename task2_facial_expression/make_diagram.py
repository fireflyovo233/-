"""绘制整体方案流程图（报告用）。"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "..", "结果", "图")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C1, C2, C3 = "#e8f0fb", "#fdf0e3", "#e9f7ef"


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=color, ec="#8a8f98", lw=1.0))
    ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color="#1a1a1a")
    ax.text(x + w / 2, y + h - 0.135, body, ha="center", va="top",
            fontsize=8.4, color="#333333", linespacing=1.45)


def arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                                 lw=1.1, color="#6b7280"))


def main():
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=170)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    box(ax, 0.015, 0.66, 0.20, 0.28, "数据", "训练 28275 张\n测试 7065 张\n48×48 灰度 / 6 类", C1)
    box(ax, 0.245, 0.66, 0.20, 0.28, "预处理", "灰度读取 → /255\n按训练集均值方差标准化", C1)
    box(ax, 0.475, 0.66, 0.235, 0.28, "数据增强",
        "随机裁剪(pad=4)\n水平翻转 · 旋转±10°\n平移缩放 · 随机擦除", C2)
    box(ax, 0.74, 0.66, 0.245, 0.28, "模型 FERNet",
        "stem stride-2 (24×24)\n3 stage 残差 + SE 注意力\nGAP → Dropout → FC", C3)

    box(ax, 0.74, 0.30, 0.245, 0.24, "分层 K 折交叉验证", "StratifiedKFold(5)\n每折独立训练与早停", C2)
    box(ax, 0.475, 0.30, 0.235, 0.24, "训练策略",
        "AdamW + 余弦退火\n标签平滑 0.1 · EMA 权重", C2)
    box(ax, 0.245, 0.30, 0.20, 0.24, "集成预测", "5 折概率平均\n+ 水平翻转 TTA", C1)
    box(ax, 0.015, 0.30, 0.20, 0.24, "输出", "submission.csv\nID,Emotion(0-5)", C3)

    arrow(ax, (0.215, 0.80), (0.245, 0.80))
    arrow(ax, (0.445, 0.80), (0.475, 0.80))
    arrow(ax, (0.71, 0.80), (0.74, 0.80))
    arrow(ax, (0.8625, 0.66), (0.8625, 0.54))
    arrow(ax, (0.74, 0.42), (0.71, 0.42))
    arrow(ax, (0.475, 0.42), (0.445, 0.42))
    arrow(ax, (0.245, 0.42), (0.215, 0.42))

    ax.text(0.5, 0.10, "单张图片推理：predict.py（同一套集成权重 + TTA）",
            ha="center", fontsize=9, color="#6b7280",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f6f7f9", ec="#d0d4da"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "pipeline.png"), bbox_inches="tight")
    print("saved", os.path.abspath(os.path.join(FIG, "pipeline.png")))


if __name__ == "__main__":
    main()
