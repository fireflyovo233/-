# 机器学习课程设计

本仓库包含四个基于深度学习的计算机视觉任务实验：

| 任务 | 课题 | 方法 | 核心指标 |
|------|------|------|----------|
| Task 1 | 植物幼苗图像分类 | ResNet34 迁移学习 | 准确率 94.7% |
| Task 2 | 人脸表情识别 | FERNet + ResNet18 集成 | 留出准确率 69.48% |
| Task 3 | 眼底黄斑中央凹定位 | ResNet18+FPN 热图回归 | 留折 RMSE 9.23 px |
| Task 4 | 眼底血管图像分割 | ResUNet + 组合损失 | 袋外 Dice 0.8414 |

## 运行环境

- Python 3.11+
- PyTorch 2.x (CPU)
- 依赖: torchvision, numpy, pandas, scikit-learn, matplotlib, Pillow, opencv-python

## 快速开始

```bash
# Task 1: 植物分类
cd task1_plant_classification
python train.py

# Task 2: 表情识别
cd task2_facial_expression
python train_cv.py --folds 5 --epochs 30

# Task 3: 黄斑定位
cd task3_fovea_detection
python src/prepare_data.py
python src/train.py

# Task 4: 血管分割
cd task4_vessel_segmentation
python code/vseg/train.py
```

## 代码仓库

https://github.com/fireflyovo233/llie-adaptive-gamma-guided-filter

## 报告

详细实验报告见 `report/机器学习课程设计报告.docx`