"""把 FERNet 5 折集成与 ResNet 微调模型融合，生成最终提交。

用法:
    python blend_submit.py --w_ft 0.65        # 微调模型权重
    python blend_submit.py --search          # 在 holdout 上搜索最优权重
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from fer_data import SUBMIT_LABEL, load_train

BASE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(BASE, "runs")
FT = os.path.join(BASE, "runs_ft")
ROOT = os.path.join(BASE, "..")


def load_ft_probs(kind="test"):
    probs, names = [], []
    for p in sorted(glob.glob(os.path.join(FT, f"*_{kind}.npy"))):
        if "_val_idx" in p:
            continue
        probs.append(np.load(p))
        names.append(os.path.basename(p).replace(f"_{kind}.npy", ""))
    return probs, names


def search_weight():
    """在微调模型的 holdout 上，用各折模型在该子集的表现近似搜索融合权重。"""
    tr = load_train(os.path.join(ROOT, "根文件", "fer_data"), os.path.join(RUNS, "cache", "train.npz"))
    oof = np.load(os.path.join(RUNS, "oof.npy"))
    idx_files = sorted(glob.glob(os.path.join(FT, "*_val_idx.npy")))
    val_files = [f.replace("_val_idx", "") for f in idx_files]
    if not idx_files:
        print("没有 holdout 预测，跳过搜索")
        return None
    idx = np.load(idx_files[0])
    y = tr.y[idx]
    p_ft = np.mean([np.load(f) for f in val_files], axis=0)
    p_cv = oof[idx]
    print(f"holdout 单独表现：微调 {(p_ft.argmax(1)==y).mean():.4f}  |  "
          f"5 折集成 {(p_cv.argmax(1)==y).mean():.4f}  "
          f"（注意：该 holdout 中部分样本被部分折模型见过，5 折数值偏乐观）")
    best_w, best_acc = 0.65, -1
    for w in np.arange(0.0, 1.01, 0.05):
        acc = ((w * p_ft + (1 - w) * p_cv).argmax(1) == y).mean()
        if acc > best_acc:
            best_acc, best_w = acc, w
    print(f"最优权重 w_ft={best_w:.2f} 时 holdout 准确率 {best_acc:.4f}")
    return float(best_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w_ft", type=float, default=0.65)
    ap.add_argument("--search", action="store_true")
    args = ap.parse_args()

    p_cv = np.load(os.path.join(RUNS, "test_prob.npy"))
    p_fts, names = load_ft_probs("test")
    if not p_fts:
        raise FileNotFoundError("runs_ft 下没有测试集概率，请先跑 finetune.py")
    p_ft = np.mean(p_fts, axis=0)
    print("融合模型:", names)

    w = args.w_ft
    if args.search:
        got = search_weight()
        if got is not None:
            w = got

    prob = w * p_ft + (1 - w) * p_cv
    np.save(os.path.join(RUNS, "test_prob_blend.npy"), prob)
    pred = SUBMIT_LABEL[prob.argmax(1)]
    np.savetxt(os.path.join(RUNS, "test_conf_blend.txt"), prob.max(1), fmt="%.4f")

    tmpl = pd.read_csv(os.path.join(ROOT, "目标文件", "submission.csv"))
    te_names = sorted(os.listdir(os.path.join(ROOT, "根文件", "fer_data", "test")))
    lut = pd.Series(pred.astype(int), index=te_names)
    sub = pd.DataFrame({"ID": tmpl["ID"], "Emotion": lut.reindex(tmpl["ID"]).values})
    assert sub.Emotion.notna().all() and sub.Emotion.between(0, 5).all()
    sub.to_csv(os.path.join(ROOT, "结果", "submission.csv"), index=False)
    sub.to_csv(os.path.join(ROOT, "目标文件", "submission.csv"), index=False)

    names_cn = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    vc = sub.Emotion.value_counts().sort_index()
    print(f"融合权重 w_ft={w:.2f}；写入 {len(sub)} 行")
    for i, n in enumerate(names_cn):
        print(f"  {i} {n:<9}{vc.get(i,0):>6} ({vc.get(i,0)/len(sub)*100:5.1f}%)")
    print(f"平均置信度 {prob.max(1).mean():.3f}")


if __name__ == "__main__":
    main()
