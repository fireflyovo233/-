# -*- coding: utf-8 -*-
"""Compare the three training iterations on the same held-out images, and chart
the two score estimates (self-evaluated vs what the grader's label convention gives).
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"D:/机器学习课设/task4"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FOLDS = ["fold0", "fold1", "fold2", "fold3"]
VALS = {"fold0": "1,6,11,16", "fold1": "3,8,14,19", "fold2": "4,9,15,20", "fold3": "2,7,12,17"}
LOGS = {
    "v1": {"fold0": "fold0.out", "fold1": "fold1.out", "fold2": "fold2.out", "fold3": "fold3.out"},
    "v2": {"fold0": "v2_f0v2.out", "fold1": "v2_f1v2.out", "fold2": "v2_f2v2.out", "fold3": "v2_f3v2.out"},
    "v3": {"fold0": "v3_f0.out", "fold1": "v3_f1.out", "fold2": "v3_f2.out", "fold3": "v3_f3.out"},
}
COLORS = {"fold0": "#1f6feb", "fold1": "#2ea043", "fold2": "#d29922", "fold3": "#c2410c"}


def last_dice(path):
    if not os.path.exists(path):
        return None
    txt = open(path, encoding="utf-8").read()
    vals = re.findall(r"valDice ema ([0-9.]+)", txt)
    return float(vals[-1]) if vals else None


def main():
    out = {"folds": {}, "means": {}}
    for f in FOLDS:
        out["folds"][f] = {tag: last_dice(os.path.join(ROOT, "outputs", "logs", mp[f]))
                           for tag, mp in LOGS.items()}
    for tag in LOGS:
        vs = [out["folds"][f][tag] for f in FOLDS if out["folds"][f][tag]]
        out["means"][tag] = round(sum(vs) / len(vs), 4) if vs else None

    vf = os.path.join(ROOT, "outputs", "verify_convention.json")
    rec = os.path.join(ROOT, "outputs", "recipe.json")
    if os.path.exists(vf):
        out["convention"] = json.load(open(vf, encoding="utf-8"))
    if os.path.exists(rec):
        out["recipe"] = json.load(open(rec, encoding="utf-8"))
    json.dump(out, open(os.path.join(ROOT, "outputs", "iter_compare.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.4, 1]})

    ax = axes[0]
    labels = ["第一轮\n300步/w16", "第二轮\n1000步/w20", "第三轮\n2000步/w20"]
    for k, f in enumerate(FOLDS):
        ys = [out["folds"][f].get(t) for t in ("v1", "v2", "v3")]
        xs = [i + (k - 1.5) * 0.16 for i in range(3)]
        ax.plot(xs, [y if y else float("nan") for y in ys], "o-", lw=1.2, ms=4.5,
                color=COLORS[f], alpha=0.85, label=f"留出 {VALS[f]}")
    means = [out["means"].get(t) for t in ("v1", "v2", "v3")]
    ax.plot([0, 1, 2], means, "k--", lw=2.2, marker="D", ms=6, label="四折平均")
    for i, m in enumerate(means):
        if m:
            ax.annotate(f"{m:.4f}", (i, m), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("留出集 Dice")
    ax.set_ylim(0.74, 0.85)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("同一批留出图像：三轮训练对比", fontsize=11)

    ax = axes[1]
    v1 = out.get("convention", {}).get("at_submitted_thr", {})
    names = ["原始口径\n(>0.5)", "官方口径\n(BILINEAR+>0)", "线上实测"]
    vals = [v1.get("native", 0), v1.get("official", 0), 0.73]
    cols = ["#9fb4cc", "#1f6feb", "#d9534f"]
    bars = ax.bar(names, vals, color=cols, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=10,
                fontweight="bold")
    ax.set_ylim(0.6, 0.85)
    ax.set_ylabel("Dice")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("第一轮提交：自评 vs 线上\n（只有官方口径解释得通）", fontsize=10.5)
    ax.tick_params(axis="x", labelsize=8.5)

    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "outputs", "fig_iter_compare.png"), dpi=150)
    print(json.dumps(out["means"], ensure_ascii=False))
    print("saved outputs/fig_iter_compare.png")


if __name__ == "__main__":
    main()
