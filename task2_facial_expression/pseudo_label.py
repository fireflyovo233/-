"""伪标签自训练：用已微调的模型给测试集打高置信标签，混入训练集再训几轮。

用法:
    python pseudo_label.py --src runs_ft/resnet18_s0.pt --epochs 4 --thr 0.85
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2

from fer_data import load_test, load_train
from finetune import (FT, IMAGENET_MEAN, IMAGENET_STD, build_model, predict, to_rgb)

BASE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(BASE, "runs")
torch.set_num_threads(16)


class EMA:
    def __init__(self, model, decay=0.998):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for s, p in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                s.copy_(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--src", default=os.path.join(FT, "resnet18_s0.pt"))
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--thr", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="pl")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tr = load_train(os.path.join(BASE, "..", "根文件", "fer_data"),
                    os.path.join(RUNS, "cache", "train.npz"))
    te = load_test(os.path.join(BASE, "..", "根文件", "fer_data"),
                   os.path.join(RUNS, "cache", "test.npz"))

    prep = lambda arr: (torch.from_numpy(arr).float().div_(255.0).unsqueeze(1) - 0.5070) / 0.2551
    x = prep(tr.x)
    y = torch.from_numpy(tr.y)
    xte = prep(te.x)

    idx_tr, idx_va = train_test_split(np.arange(len(y)), test_size=0.1,
                                      stratify=y.numpy(), random_state=42)

    model = build_model(args.arch).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(args.src, map_location="cpu"))
    model.eval()

    p_te = predict(model, xte, args.size)
    conf = p_te.max(1)
    keep = conf >= args.thr
    xp = xte[keep]
    yp = torch.from_numpy(p_te.argmax(1)[keep]).long()
    print(f"伪标签样本 {int(keep.sum())}/{len(keep)}（阈值 {args.thr}，"
          f"平均置信度 {conf[keep].mean():.3f}）", flush=True)

    xtr = torch.cat([x[idx_tr], xp])
    ytr = torch.cat([y[idx_tr], yp])
    xva, yva = x[idx_va], y[idx_va]
    print(f"训练集 {len(xtr)}（原 {len(idx_tr)} + 伪 {len(xp)}），验证 {len(xva)}", flush=True)

    ema = EMA(model, 0.998)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(len(xtr) // args.batch, 1)
    total = args.epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + np.cos(np.pi * s / total)))
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    best, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        perm = torch.randperm(len(xtr))
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb = to_rgb(xtr[idx], args.size, train=True)
            yb = ytr[idx]
            loss = lossf(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ema.update(model)
            run_loss += loss.item() * len(yb)
            seen += len(yb)
        va = (predict(ema.shadow, xva, args.size, tta=False).argmax(1) == yva.numpy()).mean()
        print(f"[pl] ep{ep+1} loss={run_loss/seen:.4f} holdout={va:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if va > best:
            best = va
            best_state = copy.deepcopy(ema.shadow.state_dict())

    model.load_state_dict(best_state)
    out = os.path.join(FT, f"{args.arch}_s{args.seed}_{args.tag}.pt")
    torch.save(best_state, out)
    pv = predict(model, xva, args.size)
    pte = predict(model, xte, args.size)
    np.save(out.replace(".pt", "_val.npy"), pv)
    np.save(out.replace(".pt", "_val_idx.npy"), idx_va)
    np.save(out.replace(".pt", "_test.npy"), pte)
    json.dump({"arch": args.arch, "tag": args.tag, "thr": args.thr,
               "best_holdout": float(best),
               "val_acc_tta": float((pv.argmax(1) == yva.numpy()).mean()),
               "n_pseudo": int(keep.sum())},
              open(out.replace(".pt", ".json"), "w"), indent=2)
    print(f"best holdout = {best:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
