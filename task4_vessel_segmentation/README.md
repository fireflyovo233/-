# 眼底血管分割（2026 task-4 / DRIVE 风格）—— 复现与交付说明

无 GPU，全部训练与推理在 **CPU（16 核）** 上完成。

## 0. 交付物一览

| 文件 | 说明 |
| --- | --- |
| `目标文件/submission.csv` | **最终提交文件**：20 行 `Id,Predicted`，512×512、行优先 1-based RLE，按「官方缩放口径」构造 |
| `目标文件/submission_fallback_thin.csv` | 保守备份（细掩膜，阈值 0.575）。若新提交分数反而下降，改交这一份 |
| `交付物/实验报告.docx` | 实验报告（Word）：数据、方法、三轮迭代、线上差距归因、复现步骤 |
| `交付物/实验报告.html` | 同内容网页版，图表内嵌 |
| `submission/image/*.png` | 20 张 512×512 二值掩膜（血管 255 / 背景 0） |
| `models/*.pt` | 4 折 × 三代（`fold*` / `f*v2` / `f*v3`），最终用 `f*v3` 的 ema+best 共 8 个 |
| `outputs/` | 日志、袋外标定、口径反推、方案对比、图表 |

## 1. 关键结论（含线上反馈后的两轮迭代）

| 版本 | 训练量 | 留出 Dice | 袋外 Dice（官方口径） | 线上 |
| --- | --- | --- | --- | --- |
| 第一轮 | 300 步 / w16 | 0.7984 | 0.7373 | **0.73** |
| 第二轮 | 1000 步 / w20 | 0.8053 | — | — |
| 第三轮 | 2000 步 / w20 | **0.8102** | **0.8414** | 待测 |

**最重要的一条**：线上 0.73 与「原始口径」自评 0.782 差 5 个点，而与「官方口径」0.737 几乎一致。
原因是 `segmentation_to_csv.py` 用 `BILINEAR resize` + `image[image>0]=1` 生成掩膜 ——
对 0/255 标签做双线性下采样时，1 px 宽的血管会被摊到相邻像素，只要权重非零就算前景，
**标签被系统性加粗约 1 px，覆盖率 8.7% → 13.1%**。评测端若同样处理，细预测必然欠分割。

因此最终提交不是靠调阈值，而是对预测施加**同一种变换**：全分辨率阈值 0.50 二值化 →
`PIL BILINEAR` 缩到 512 → `>0`。袋外 Dice 从 0.737 提升到 0.841。

## 2. 环境

```bash
python -m venv <venv>
<venv>/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
<venv>/Scripts/pip install numpy pandas pillow opencv-python-headless scikit-image scikit-learn matplotlib python-docx
```

本机：`C:/Users/pc/.workbuddy/binaries/python/envs/default/Scripts/python.exe`

## 3. 目录

```
code/
  vseg/data.py        读图(支持中文路径)、CLAHE 绿通道、FOV 掩膜、patch 采样与增强
  vseg/model.py       ResUNet + BCE/SoftDice/clDice
  vseg/infer.py       整图推理、二面体 TTA、多尺度 TTA、Dice
  train.py            训练（warmup+余弦、预热 EMA、中途验证、早停/时间预算、--init 续训）
  oof_ensemble.py     最终集成的袋外阈值标定（每张图只用没见过它的模型）
  verify_convention.py 反推评测端的标签口径（对照线上分数）
  recipe.py           比较三种掩膜构造方案（A 低阈值 / B 复刻官方缩放 / C 膨胀）
  oof_convention.py   两种口径下的阈值扫描（含缓存）
  predict.py          多模型集成推理 + 缓存概率图
  make_masks.py       概率图 -> 512 掩膜（--recipe A/B/C）-> RLE csv
  check_submission.py 按评测口径完整校验 submission.csv
  iter_compare.py     三轮训练对比 + 自评/线上对比图
  gen_report.py       生成 docx + html 报告
  run_all.sh          一键复现
```

## 4. 复现

```bash
# 四折训练（4 进程并行、每进程 4 线程；线程多了几乎不提速：4 线程 0.74 s/iter，12 线程 0.81 s/iter）
python code/train.py --epochs 4 --iters 500 --patch 112 --bs 10 --width 20 --lr 2e-3 \
       --seed 11 --threads 4 --eval-every 2 --val 0,5,10,15 --out f0v2 --tag f0v2
# 续训到 2000 步
python code/train.py --epochs 2 --iters 500 --patch 112 --bs 10 --width 20 --lr 1e-3 \
       --seed 11 --threads 4 --eval-every 2 --val 0,5,10,15 --init models/f0v2_best.pt --out f0v3 --tag f0v3

python code/verify_convention.py --tta 4      # 确认评测口径
python code/recipe.py                          # 选掩膜构造方案
python code/predict.py --ckpts models/f0v3_ema.pt models/f0v3_best.pt models/f1v3_ema.pt \
       models/f1v3_best.pt models/f2v3_ema.pt models/f2v3_best.pt models/f3v3_ema.pt \
       models/f3v3_best.pt --tta 4 --thr 0.50 --save-probs
python code/make_masks.py --thr 0.50 --recipe B --min-size 30
python code/check_submission.py
python code/iter_compare.py && python code/gen_report.py
```

折划分（0-based 下标，id = 下标+1）：fold0 `1,6,11,16` / fold1 `3,8,14,19` / fold2 `4,9,15,20` / fold3 `2,7,12,17`。

## 5. 踩过的坑

1. **EMA 用固定 decay=0.995**：1500 步时权重仍是「随机初始化 + 少量训练」的混合体，验证 Dice 恒为 0。
   改成 `decay=min(0.99,(1+step)/(50+step))` 后第 3 轮即恢复 0.68。
2. **`cv2.imread/imwrite` 在中文路径下静默失败**（imwrite 返回 True 但文件不存在）→ 统一 `imencode/imdecode + tofile`。
3. **后台训练必须让父进程 `wait`**：`nohup ... &` + 短 sleep 的包装脚本退出后子进程被回收，白跑过两次。
4. **线程扩展性极差**：CPU 卷积网络 4 线程≈12 线程速度，正确做法是多进程少线程。
5. **自评口径必须与评测端同构**：否则 0.79 与 0.73 之间的 6 个点会莫名其妙消失。
