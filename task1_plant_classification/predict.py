# -*- coding: utf-8 -*-
"""
植物幼苗分类 - 测试集推理与提交生成
"""
import os
import torch
import numpy as np
import pandas as pd
import argparse
from dataset import get_test_loader, IDX_TO_CLASS
from model import create_model


@torch.no_grad()
def predict(model, loader, device, use_tta=True):
    """推理并返回 (文件名列表, 预测类别列表)"""
    model.eval()
    filenames_all = []
    preds_all = []

    for imgs, fnames in loader:
        imgs = imgs.to(device)
        batch_size = imgs.size(0)

        # 标准前向
        logits = model(imgs)

        # TTA: 水平翻转后再次前向
        if use_tta:
            imgs_flip = torch.flip(imgs, dims=[3])
            logits_flip = model(imgs_flip)
            logits = (logits + logits_flip) / 2.0

        preds = logits.argmax(dim=1).cpu().numpy()
        filenames_all.extend(fnames)
        preds_all.extend(preds)

    return filenames_all, preds_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='数据集根目录（含 test/ 子目录）')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重路径')
    parser.add_argument('--output', type=str, default='submission.csv', help='输出CSV路径')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--no_tta', action='store_true', help='禁用TTA')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'使用设备: {device}')

    # 加载模型
    model = create_model(num_classes=5, freeze_backbone=False, device=device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f'已加载权重: {args.checkpoint}')

    # 测试数据
    test_loader = get_test_loader(args.data_dir, batch_size=args.batch_size)
    print(f'测试集: {len(test_loader.dataset)} 张')

    # 推理
    filenames, preds = predict(model, test_loader, device, use_tta=not args.no_tta)

    # 写提交文件
    categories = [IDX_TO_CLASS[p] for p in preds]
    df = pd.DataFrame({'ID': filenames, 'Category': categories})
    df.to_csv(args.output, index=False)
    print(f'提交文件已保存至: {args.output}')
    print(f'各类别预测数量:')
    for cat, cnt in df['Category'].value_counts().items():
        print(f'  {cat}: {cnt}')


if __name__ == '__main__':
    main()