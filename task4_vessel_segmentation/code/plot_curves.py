"""Plot training curves from a history json."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", nargs="+", required=True)
    ap.add_argument("--out", default=r"D:/机器学习课设/task4/outputs/fig_curves.png")
    ap.add_argument("--labels", nargs="*", default=None)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k, h in enumerate(args.hist):
        d = json.load(open(h))
        hist = d["hist"]
        labs = args.labels[k] if args.labels and k < len(args.labels) else os.path.basename(h)
        ep = [r["epoch"] for r in hist]
        axes[0].plot(ep, [r["loss"] for r in hist], label=labs, lw=1.8)
        pts = [(r["epoch"], r["val_dice_ema"]) for r in hist if "val_dice_ema" in r]
        if pts:
            axes[1].plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3.5,
                         label=labs + " (EMA)", lw=1.8)
        pts2 = [(r["epoch"], r["val_dice_raw"]) for r in hist if "val_dice_raw" in r]
        if pts2:
            axes[1].plot([p[0] for p in pts2], [p[1] for p in pts2], marker="s", ms=3,
                         ls="--", lw=1.2, alpha=0.7, label=labs + " (raw)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("训练损失"); axes[0].set_title("损失曲线")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("验证集 Dice"); axes[1].set_title("验证集 Dice（4 折外样本）")
    axes[1].set_ylim(0.5, 0.85)
    axes[1].grid(alpha=0.3); axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
