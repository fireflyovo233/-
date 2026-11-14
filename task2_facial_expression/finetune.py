"""在 ImageNet 预训练的 ResNet 上微调（48x48 上采样到 112），用于突破 70% 准确率。

与从零训练的 FERNet 相比，迁移学习把小图数据不足的问题交给 ImageNet 特征解决：
浅层保留通用纹理/边缘滤波器，深层重新适配表情语义，只需 20 轮即可收敛。

用法:
    python finetune.py --arch resnet18 --size 112 --epochs 20 --seed 0

输出:
    runs_ft/{arch}_s{seed}.pt        最优权重（按 holdout 准确率）
    runs_ft/{arch}_s{seed}_test.npy  测试集概率（含 TTA）
    runs_ft/{arch}_s{seed}.json      训练曲线与指标
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision import models
from torchvision.transforms import v2

from fer_data import CLASSES, load_test, load_train

BASE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(BASE, "runs")
FT = os.path.join(BASE, "runs_ft")
torch.set_num_threads(16)

IMAGENET_MEAN, IMAGENET_STD = 0.449, 0.226  # 灰度三通道复制后的等效值


def build_model(arch: str, num_classes: int = 6):
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif arch == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(arch)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m


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


def to_rgb(x: torch.Tensor, size: int, train: bool, aug_strength: float = 1.0):
    """(B,1,48,48) 归一化灰度 -> (B,3,size,size) ImageNet 归一化张量。"""
    x = x * 0.2551 + 0.5070            # 反归一化回 [0,1]
    x = x.repeat(1, 3, 1, 1)           # 灰度复制成三通道
    if train:
        big = int(round(size * 1.14))
        x = F.interpolate(x, size=big, mode="bilinear", align_corners=False)
        x = v2.RandomCrop(size)(x)
        x = v2.RandomHorizontalFlip(0.5)(x)
        x = v2.RandomAffine(degrees=10 * aug_strength, translate=(0.06, 0.06),
                            scale=(0.95, 1.05))(x)
        x = v2.RandomErasing(p=0.25, scale=(0.02, 0.1), value=0.0)(x)
    else:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


@torch.no_grad()
def predict(model, x48, size, batch=128, tta=True):
    model.eval()
    out = []
    for i in range(0, len(x48), batch):
        xb = to_rgb(x48[i:i + batch], size, train=False)
        p = model(xb).softmax(1)
        if tta:
            p = (p + model(torch.flip(xb, dims=[3])).softmax(1)) / 2
        out.append(p)
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(FT, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tr = load_train(os.path.join(BASE, "..", "根文件", "fer_data"),
                    os.path.join(RUNS, "cache", "train.npz"))
    te = load_test(os.path.join(BASE, "..", "根文件", "fer_data"),
                   os.path.join(RUNS, "cache", "test.npz"))

    x = torch.from_numpy(tr.x).float().div_(255.0).unsqueeze(1)
    x = (x - 0.5070) / 0.2551            # 复用 FERNet 的归一化，to_rgb 里会还原
    y = torch.from_numpy(tr.y)
    xte = torch.from_numpy(te.x).float().div_(255.0).unsqueeze(1)
    xte = (xte - 0.5070) / 0.2551

    idx_tr, idx_va = train_test_split(np.arange(len(y)), test_size=args.holdout,
                                      stratify=y.numpy(), random_state=42)
    xtr, ytr = x[idx_tr], y[idx_tr]
    xva, yva = x[idx_va], y[idx_va]
    print(f"train={len(xtr)} holdout={len(xva)} size={args.size} arch={args.arch}", flush=True)

    model = build_model(args.arch).to(memory_format=torch.channels_last)
    for p in model.parameters():
        p.requires_grad_(True)
    ema = EMA(model, args.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(len(xtr) // args.batch, 1)
    total, warm = args.epochs * steps, args.warmup * steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm
        else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(total - warm, 1))))
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=args.ls)

    best, best_state, hist = -1.0, None, []
    for ep in range(args.epochs):
        model.train()
        t0, run_loss, run_acc, seen = time.time(), 0.0, 0.0, 0
        perm = torch.randperm(len(xtr))
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb = to_rgb(xtr[idx], args.size, train=True, aug_strength=args.aug)
            yb = ytr[idx]
            logits = model(xb)
            loss = lossf(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ema.update(model)
            run_loss += loss.item() * len(yb)
            run_acc += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)
        va = (predict(ema.shadow, xva, args.size, tta=False).argmax(1) == yva.numpy()).mean()
        hist.append({"epoch": ep + 1, "loss": run_loss / seen, "train_acc": run_acc / seen,
                     "val_acc": float(va), "sec": time.time() - t0})
        print(f"[{args.arch} s{args.seed}] ep{ep+1:02d} loss={run_loss/seen:.4f} "
              f"train={run_acc/seen:.4f} holdout={va:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if va > best:
            best = va
            best_state = copy.deepcopy(ema.shadow.state_dict())
            torch.save(best_state, os.path.join(FT, f"{args.arch}_s{args.seed}.pt"))

    model.load_state_dict(best_state)
    pv = predict(model, xva, args.size)
    pte = predict(model, xte, args.size)
    np.save(os.path.join(FT, f"{args.arch}_s{args.seed}_val.npy"), pv)
    np.save(os.path.join(FT, f"{args.arch}_s{args.seed}_val_idx.npy"), idx_va)
    np.save(os.path.join(FT, f"{args.arch}_s{args.seed}_test.npy"), pte)
    json.dump({"arch": args.arch, "size": args.size, "seed": args.seed,
               "best_holdout": float(best),
               "val_acc_tta": float((pv.argmax(1) == yva.numpy()).mean()),
               "history": hist},
              open(os.path.join(FT, f"{args.arch}_s{args.seed}.json"), "w"), indent=2)
    print(f"[{args.arch} s{args.seed}] best holdout = {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
