"""评测与可视化：混淆矩阵、分类报告、训练曲线、提交文件校验。"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from fer_data import CLASSES, load_train

BASE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(BASE, "runs")
FIG = os.path.join(BASE, "..", "结果", "图")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    os.makedirs(FIG, exist_ok=True)
    metrics = json.load(open(os.path.join(RUNS, "metrics.json"), encoding="utf-8"))
    tr = load_train(os.path.join(BASE, "..", "根文件", "fer_data"),
                    os.path.join(RUNS, "cache", "train.npz"))
    oof = np.load(os.path.join(RUNS, "oof.npy"))
    done = oof.sum(1) > 0
    y = tr.y[done]
    yp = oof.argmax(1)[done]

    acc = (y == yp).mean()
    print(f"OOF accuracy = {acc:.4f}  (n={len(y)})")
    print(classification_report(y, yp, target_names=CLASSES, digits=4))

    cm = confusion_matrix(y, yp)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title(f"5 折交叉验证混淆矩阵（归一化，准确率 {acc*100:.2f}%）")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cmn[i,j]*100:.1f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.55 else "#1a1a1a", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "confusion_matrix.png"))
    plt.close(fig)

    # 训练曲线
    folds = [k for k in metrics if k.startswith("fold")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    for k in folds:
        h = pd.DataFrame(metrics[k]["history"])
        axes[0].plot(h.epoch, h.loss, lw=1.2, label=k)
        axes[1].plot(h.epoch, h.val_acc * 100, lw=1.2, label=f"{k} ({metrics[k]['val_acc']*100:.2f}%)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("训练损失"); axes[0].set_title("各折训练损失")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("验证准确率 (%)"); axes[1].set_title("各折验证准确率")
    for a in axes:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "training_curves.png"))
    plt.close(fig)

    # 每类准确率
    per = {CLASSES[i]: float((yp[y == i] == i).mean()) for i in range(len(CLASSES))}
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=150)
    bars = ax.bar(list(per), [v * 100 for v in per.values()], color="#4c78a8")
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9)
    ax.set_ylim(0, 100); ax.set_ylabel("召回率 (%)")
    ax.set_title("各类别识别准确率（OOF）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "per_class_acc.png"))
    plt.close(fig)

    # 提交文件校验
    sub = pd.read_csv(os.path.join(RUNS, "submission.csv"))
    tmpl = pd.read_csv(os.path.join(BASE, "..", "目标文件", "submission.csv"))
    print("submission:", sub.shape, "列:", sub.columns.tolist())
    print("ID 与模板一致:", list(sub.ID) == list(tmpl.ID))
    print("取值合法(0-5):", sub.Emotion.between(0, 5).all(), "| 缺失:", sub.Emotion.isna().sum())
    print("预测分布:", sub.Emotion.value_counts().sort_index().to_dict())

    summary = {
        "cv_accuracy": acc,
        "n_train": int(len(y)),
        "per_class_acc": per,
        "fold_val_acc": {k: metrics[k]["val_acc"] for k in folds},
        "pred_distribution": sub.Emotion.value_counts().sort_index().to_dict(),
    }
    json.dump(summary, open(os.path.join(BASE, "..", "结果", "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
