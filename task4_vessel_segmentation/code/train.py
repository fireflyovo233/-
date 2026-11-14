"""Train a ResUNet for retinal vessel segmentation (patch based, CPU friendly)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vseg.data import RetinaSet, make_batch
from vseg.model import ResUNet, ComboLoss
from vseg.infer import predict_probs, dice, postprocess

torch.set_num_threads(min(16, os.cpu_count() or 8))


class EMA:
    """Exponential moving average with a warm-up schedule (fast at the beginning)."""

    def __init__(self, model, decay=0.99):
        self.decay_max = decay
        self.step = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay_max, (1.0 + self.step) / (50.0 + self.step))
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                s.copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


def lr_at(step, total, base_lr, warmup=120, min_lr_ratio=0.01):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", type=str, default="")          # e.g. "1,6,12,17" (0-based idx)
    ap.add_argument("--val-fold", type=str, default="")     # "k/n" 5-fold style split
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--out", type=str, default="model")
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--init", type=str, default="")
    ap.add_argument("--eval-every-steps", type=int, default=0)  # mid-epoch validation
    ap.add_argument("--patience", type=int, default=0)          # early stop on val epochs
    ap.add_argument("--max-hours", type=float, default=0.0)     # wall-clock budget
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    root = r"D:/机器学习课设/task4"
    ckpt_dir = os.path.join(root, "models")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = os.path.join(root, "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed + 1234)

    ds = RetinaSet("train")
    n = len(ds)
    ids = list(range(n))
    if args.val:
        val_idx = [int(v) for v in args.val.split(",") if v != ""]
    elif args.val_fold:
        k, m = [int(v) for v in args.val_fold.split("/")]
        val_idx = [i for i in ids if i % m == k % m]
    else:
        val_idx = []
    tr_idx = [i for i in ids if i not in val_idx]
    print(f"train imgs {len(tr_idx)} val imgs {len(val_idx)}: {[ds.ids[i] for i in val_idx]}", flush=True)

    tr_x = [ds.imgs[i] for i in tr_idx]
    tr_y = [ds.labels[i] for i in tr_idx]
    tr_v = [ds.vessel_pts[i] for i in tr_idx]

    model = ResUNet(3, 1, args.width, args.depth, dilate_bottleneck=False)
    print("params %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["sd"])
        print("initialised from", args.init, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = ComboLoss(w_bce=0.4, w_dice=0.4, w_cl=0.2, pos_weight=2.0, cl_iters=6)
    crit_nocl = ComboLoss(w_bce=0.5, w_dice=0.5, w_cl=0.0, pos_weight=2.0)
    ema = EMA(model, decay=0.99)

    total = args.epochs * args.iters
    step = 0
    hist = []
    best = -1.0
    since_best = 0
    t_start = time.time()
    best_path = os.path.join(ckpt_dir, f"{args.out}_best.pt")

    def evaluate(tag):
        """Validate the EMA weights and the current weights; keep the best EMA snapshot."""
        nonlocal best, since_best
        ema_model = ResUNet(3, 1, args.width, args.depth, dilate_bottleneck=False)
        ema.copy_to(ema_model)
        ema_model.eval()
        ds_e, ds_r = [], []
        for i in val_idx:
            pe = predict_probs(ema_model, ds.imgs[i], n_tta=4, fov=ds.fovs[i])
            ds_e.append(dice(pe > 0.5, ds.labels[i]))
            pr = predict_probs(model.eval(), ds.imgs[i], n_tta=4, fov=ds.fovs[i])
            ds_r.append(dice(pr > 0.5, ds.labels[i]))
        model.train()
        rec = {"val_dice_ema": float(np.mean(ds_e)), "val_dice_raw": float(np.mean(ds_r)),
               "sec": round(time.time() - t_start, 1), "note": tag}
        if rec["val_dice_ema"] > best:
            best = rec["val_dice_ema"]
            since_best = 0
            torch.save({"sd": ema.shadow, "args": vars(args), "dice": best},
                       best_path)
            rec["best"] = True
        else:
            since_best += 1
        print("  eval step %d (%s): ema %.4f / raw %.4f%s"
              % (step, tag, rec["val_dice_ema"], rec["val_dice_raw"],
                 "  *best*" if rec.get("best") else ""), flush=True)
        return rec

    stop = False
    for ep in range(args.epochs):
        model.train()
        run_loss = 0.0
        for it in range(args.iters):
            lr = lr_at(step, total, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            xb, yb = make_batch(tr_x, tr_y, tr_v, args.patch, args.bs, rng)
            x = torch.from_numpy(xb)
            y = torch.from_numpy(yb)
            out = model(x)
            # centreline-dice (topology aware) on every other step to halve its cost
            loss = (crit if step % 2 == 0 else crit_nocl)(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ema.update(model)
            run_loss += float(loss.detach())
            step += 1
            if (val_idx and args.eval_every_steps and step % args.eval_every_steps == 0
                    and not (it == args.iters - 1)):
                hist.append(dict(evaluate("mid"), epoch=ep + 1 + it / args.iters, loss=None))
        run_loss /= args.iters
        rec = {"epoch": ep + 1, "loss": run_loss, "lr": lr, "sec": round(time.time() - t_start, 1)}

        if val_idx and ((ep + 1) % args.eval_every == 0 or ep == args.epochs - 1):
            rec.update(evaluate("epoch"))
        hist.append(rec)
        msg = f"ep{ep+1}/{args.epochs} loss {run_loss:.4f}"
        if "val_dice_ema" in rec:
            msg += f" valDice ema {rec['val_dice_ema']:.4f} / raw {rec['val_dice_raw']:.4f}"
        msg += f" {time.time()-t_start:.0f}s"
        print(msg, flush=True)

        if args.patience and since_best >= args.patience:
            print(f"early stop: no val improvement in {since_best} evaluations", flush=True)
            stop = True
        if args.max_hours and (time.time() - t_start) / 3600.0 > args.max_hours:
            print("early stop: wall-clock budget reached", flush=True)
            stop = True
        if stop:
            break

    torch.save({"sd": ema.shadow, "args": vars(args), "hist": hist},
               os.path.join(ckpt_dir, f"{args.out}_last.pt"))
    # best EMA snapshot is what we ship as "last" quality-wise: also keep a copy
    if val_idx and os.path.exists(best_path):
        torch.save({"sd": torch.load(best_path, map_location="cpu", weights_only=False)["sd"],
                    "args": vars(args), "hist": hist, "best_dice": best},
                   os.path.join(ckpt_dir, f"{args.out}_ema.pt"))
    with open(os.path.join(log_dir, f"{args.tag}_history.json"), "w") as f:
        json.dump({"args": vars(args), "hist": hist}, f, indent=1)
    print("done in %.1f min" % ((time.time() - t_start) / 60), flush=True)


if __name__ == "__main__":
    main()
