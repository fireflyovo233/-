"""把集成预测写入提交模板：结果/submission.csv 与 目标文件/submission.csv。

模板的 ID 顺序与测试集目录的字典序不同，因此按 ID 对齐而不是按位置写入。
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(BASE, "runs")
ROOT = os.path.join(BASE, "..")


def main():
    tmpl = pd.read_csv(os.path.join(ROOT, "目标文件", "submission.csv"))
    pred = pd.read_csv(os.path.join(RUNS, "submission.csv"))
    assert list(pred.ID) == list(tmpl.ID), "ID 顺序需与模板一致"
    assert pred.Emotion.between(0, 5).all(), "标签越界"
    assert pred.Emotion.notna().all(), "存在缺失预测"

    out = os.path.join(ROOT, "结果")
    os.makedirs(out, exist_ok=True)
    pred.to_csv(os.path.join(out, "submission.csv"), index=False)
    shutil.copyfile(os.path.join(out, "submission.csv"),
                    os.path.join(ROOT, "目标文件", "submission.csv"))

    names = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]  # 提交标签顺序
    vc = pred.Emotion.value_counts().sort_index()
    print(f"共 {len(pred)} 行；已写入 结果/submission.csv 与 目标文件/submission.csv")
    for i, n in enumerate(names):
        print(f"  {i} {n:<9}{vc.get(i, 0):>6}  ({vc.get(i,0)/len(pred)*100:5.1f}%)")
    if os.path.exists(os.path.join(RUNS, "test_conf.txt")):
        c = np.loadtxt(os.path.join(RUNS, "test_conf.txt"))
        print(f"平均置信度 {c.mean():.3f}；置信度<0.5 的样本 {np.mean(c < 0.5)*100:.1f}%")


if __name__ == "__main__":
    main()
