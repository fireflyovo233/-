#!/usr/bin/env bash
# One-click reproduction (Windows / Git Bash). Override PY to use another interpreter.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-C:/Users/pc/.workbuddy/binaries/python/envs/default/Scripts/python.exe}"

echo "[1/6] dev fold (16 train / 4 held-out): loss + validation curve"
$PY code/train.py --epochs 30 --iters 50 --eval-every 3 --val 2,7,13,18 --seed 0 --threads 11 \
    --out dev0 --tag dev0 --max-hours 0.25

echo "[2/6] four folds -> out-of-fold threshold calibration"
$PY code/train.py --epochs 1 --iters 300 --patch 96 --bs 12 --seed 11 --val 0,5,10,15 --out fold0 --tag fold0 &
$PY code/train.py --epochs 1 --iters 300 --patch 96 --bs 12 --seed 12 --val 2,7,13,18 --out fold1 --tag fold1 &
$PY code/train.py --epochs 1 --iters 300 --patch 96 --bs 12 --seed 13 --val 3,8,14,19 --out fold2 --tag fold2 &
$PY code/train.py --epochs 1 --iters 300 --patch 96 --bs 12 --seed 1  --val 1,6,11,16  --out fold3 --tag fold3 &
wait
$PY code/oof_threshold.py --tta 4 --out outputs/thr.json

echo "[3/6] two full-data seeds (used for the ensemble-consistency check)"
$PY code/train.py --epochs 2 --iters 500 --patch 96 --bs 12 --seed 21 --val "" --out s1 --tag s1 &
$PY code/train.py --epochs 2 --iters 500 --patch 96 --bs 12 --seed 22 --val "" --out s2 --tag s2 &
wait

echo "[4/6] inference: 4-fold ensemble + 8-way TTA (caches probability maps)"
$PY code/predict.py --ckpts models/fold0_ema.pt models/fold1_ema.pt models/fold2_ema.pt models/fold3_ema.pt \
    --tta 8 --thr 0.575 --min-size 30 --save-probs

echo "[5/6] probability maps -> 512 masks -> RLE csv -> validation"
$PY code/make_masks.py --min-size 30
$PY code/postproc_sens.py
$PY code/check_submission.py

echo "[6/6] figures + report"
$PY code/plot_curves.py --hist outputs/logs/dev0_history.json --labels "dev0 (16/4)"
$PY code/gen_report.py
echo "done"
