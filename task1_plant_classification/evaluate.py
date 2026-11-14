# -*- coding: utf-8 -*-
"""
植物幼苗分类 - 评估脚本（验证集混淆矩阵与分类报告）
"""
import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import argparse

from dataset import get_train_loader, CLASSES, IDX_TO_CLASS
from model import create_model


@torch.no_grad()
def evaluate_detailed(model, loader, device):
    """返回所有预测值、真实值、文件名"""
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='数据集根目录')
    parser.add_argument('--csv_path', type=str, required=True, help='训练标签CSV路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重路径')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    model = create_model(num_classes=5, freeze_backbone=False, device=device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # 验证数据（使用全部训练集评估）
    from dataset import PlantDataset
    from torch.utils.data import DataLoader
    ds = PlantDataset(args.data_dir, args.csv_path, is_train=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    preds, labels = evaluate_detailed(model, loader, device)

    print('=' * 60)
    print('分类报告:')
    print(classification_report(labels, preds, target_names=CLASSES, digits=4))

    print('=' * 60)
    print('混淆矩阵:')
    cm = confusion_matrix(labels, preds)
    header = ' ' * 12 + ''.join(f'{c[:6]:>8s}' for c in CLASSES)
    print(header)
    for i, row in enumerate(cm):
        row_str = f'{CLASSES[i]:>12s}' + ''.join(f'{v:8d}' for v in row)
        print(row_str)

    acc = (preds == labels).mean()
    print(f'\n总准确率: {acc:.4f} ({acc*100:.2f}%)')


if __name__ == '__main__':
    main()