"""用已训练好的折模型做 OOF 评估与测试集集成预测（支持部分折）。"""
from __future__ import annotations
import glob, json, os
import numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold
from fer_data import CLASSES, SUBMIT_LABEL, load_test, load_train, to_tensor
from model import FERNet

BASE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(BASE, "runs")
MEAN, STD = 0.5070, 0.2551
torch.set_num_threads(16)


@torch.no_grad()
def predict(m, x, tta=True, bs=512):
    m.eval(); out = []
    for i in range(0, len(x), bs):
        xb = x[i:i+bs]
        p = m(xb).softmax(1)
        if tta:
            p = (p + m(torch.flip(xb, dims=[3])).softmax(1)) / 2
        out.append(p)
    return torch.cat(out).numpy()


def main():
    tr = load_train(os.path.join(BASE, "..", "根文件", "fer_data"), os.path.join(RUNS, "cache", "train.npz"))
    te = load_test(os.path.join(BASE, "..", "根文件", "fer_data"), os.path.join(RUNS, "cache", "test.npz"))
    xall = to_tensor(tr.x, MEAN, STD); xte = to_tensor(te.x, MEAN, STD)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(tr.y)), tr.y))

    paths = sorted(glob.glob(os.path.join(RUNS, "fold*.pt")))
    print("使用模型:", [os.path.basename(p) for p in paths])
    oof = np.zeros((len(tr.y), len(CLASSES)), np.float32)
    test_prob = np.zeros((len(xte), len(CLASSES)), np.float32)
    per_fold = {}
    for p in paths:
        k = int(os.path.basename(p)[4])
        m = FERNet(len(CLASSES), 32, 0.3)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        _, iva = splits[k]
        oof[iva] = predict(m, xall[iva])
        acc = float((oof[iva].argmax(1) == tr.y[iva]).mean())
        per_fold[f"fold{k}"] = acc
        print(f"fold{k} OOF acc = {acc:.4f}")
        test_prob += predict(m, xte)
        np.save(os.path.join(RUNS, "test_prob.npy"), test_prob / len(per_fold))
    test_prob /= len(per_fold)

    done = oof.sum(1) > 0
    cv = float((oof.argmax(1)[done] == tr.y[done]).mean())
    np.save(os.path.join(RUNS, "oof.npy"), oof)
    np.savetxt(os.path.join(RUNS, "test_conf.txt"), test_prob.max(1), fmt="%.4f")

    pred = test_prob.argmax(1)
    lut = pd.Series(SUBMIT_LABEL[pred].astype(int), index=te.names)
    tmpl = pd.read_csv(os.path.join(BASE, "..", "目标文件", "submission.csv"))
    sub = pd.DataFrame({"ID": tmpl["ID"], "Emotion": lut.reindex(tmpl["ID"]).values})
    sub.to_csv(os.path.join(RUNS, "submission.csv"), index=False)

    metrics = {"folds_used": len(per_fold), "fold_acc": per_fold, "cv_acc": cv,
               "n_oof": int(done.sum())}
    json.dump(metrics, open(os.path.join(RUNS, "metrics_summary.json"), "w"), indent=2)
    print("CV(OOF) acc =", round(cv, 4), "| folds =", len(per_fold))
    print("submission:", sub.shape)


if __name__ == "__main__":
    main()
