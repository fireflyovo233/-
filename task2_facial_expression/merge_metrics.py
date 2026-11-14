"""合并分两次训练得到的各折训练曲线（metrics_f012.json + metrics.json）。"""
from __future__ import annotations

import glob
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(BASE, "runs")


def main():
    merged: dict = {}
    for p in [os.path.join(RUNS, "metrics_f012.json"), os.path.join(RUNS, "metrics.json")]:
        if not os.path.exists(p):
            continue
        for k, v in json.load(open(p, encoding="utf-8")).items():
            if k.startswith("fold"):
                merged[k] = v
    merged = dict(sorted(merged.items()))
    json.dump(merged, open(os.path.join(RUNS, "metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("合并后的折:", list(merged), "| 各折最优验证准确率:",
          {k: round(v["val_acc"], 4) for k, v in merged.items()})


if __name__ == "__main__":
    main()
