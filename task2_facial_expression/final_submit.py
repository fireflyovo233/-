"""最终提交：ResNet18 微调（多尺度 TTA）+ FERNet 5 折集成，用逻辑回归 stacking 融合。"""
from __future__ import annotations
import json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from fer_data import SUBMIT_LABEL, load_train

BASE = os.path.dirname(os.path.abspath(__file__))
RUNS, FT = os.path.join(BASE, "runs"), os.path.join(BASE, "runs_ft")
ROOT = os.path.join(BASE, "..")


def main():
    tr = load_train(os.path.join(ROOT, "根文件", "fer_data"), os.path.join(RUNS, "cache", "train.npz"))
    idx = np.load(os.path.join(FT, "resnet18_s0_val_idx.npy"))
    y = tr.y[idx]
    ft_v, cv_v = np.load(os.path.join(FT, "tta_ms_val.npy")), np.load(os.path.join(RUNS, "oof.npy"))[idx]
    ft_t, cv_t = np.load(os.path.join(FT, "tta_ms_test.npy")), np.load(os.path.join(RUNS, "test_prob.npy"))

    feats = lambda a, b: np.hstack([np.log(a + 1e-6), np.log(b + 1e-6)])
    Xv, Xt = feats(ft_v, cv_v), feats(ft_t, cv_t)

    # 用 holdout 上的 5 折交叉验证估计 stacking 的真实水平（避免自评）
    cv_acc = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xv, y):
        m = LogisticRegression(max_iter=1000).fit(Xv[a], y[a])
        cv_acc.append((m.predict(Xv[b]) == y[b]).mean())
    cv_acc = float(np.mean(cv_acc))

    st = LogisticRegression(max_iter=1000).fit(Xv, y)
    pred = SUBMIT_LABEL[st.predict(Xt)]

    tmpl = pd.read_csv(os.path.join(ROOT, "目标文件", "submission.csv"))
    names = sorted(os.listdir(os.path.join(ROOT, "根文件", "fer_data", "test")))
    sub = pd.DataFrame({"ID": tmpl["ID"],
                        "Emotion": pd.Series(pred, index=names).reindex(tmpl["ID"]).values.astype(int)})
    assert len(sub) == len(tmpl) and sub.Emotion.notna().all() and sub.Emotion.between(0, 5).all()
    sub.to_csv(os.path.join(ROOT, "结果", "submission.csv"), index=False)
    sub.to_csv(os.path.join(ROOT, "目标文件", "submission.csv"), index=False)

    probe = st.predict_proba(Xt).max(1)
    metrics = {"stacking_cv_acc": cv_acc,
               "ft_holdout": float((ft_v.argmax(1) == y).mean()),
               "cv5_oof_single": float((cv_v.argmax(1) == y).mean()),
               "equal_blend": float(((ft_v + cv_v).argmax(1) == y).mean()),
               "mean_conf": float(probe.mean())}
    json.dump(metrics, open(os.path.join(ROOT, "结果", "final_metrics.json"), "w"), indent=2)
    print(json.dumps(metrics, indent=2))
    cn = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    vc = sub.Emotion.value_counts().sort_index()
    print("写盘", len(sub), "行 |", "  ".join(f"{i}{cn[i]}:{vc.get(i,0)}" for i in range(6)))


if __name__ == "__main__":
    main()
