# -*- coding: utf-8 -*-
"""
植物幼苗分类 - 数据集加载与预处理
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd

# 5个类别
CLASSES = ['Black-grass', 'Common wheat', 'Loose Silky-bent', 'Scentless Mayweed', 'Sugar beet']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# ImageNet 统计量（RGB）
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class PlantDataset(Dataset):
    """植物幼苗数据集，支持训练和测试模式"""

    def __init__(self, data_dir, csv_path=None, transform=None, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        self.samples = []

        if is_train and csv_path:
            # 训练模式：从CSV读取标签
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                img_path = os.path.join(data_dir, 'train', row['Category'], row['ID'])
                if os.path.exists(img_path):
                    self.samples.append((img_path, CLASS_TO_IDX[row['Category']]))
        else:
            # 测试模式：读取所有图片，无标签
            test_dir = os.path.join(data_dir, 'test')
            for fname in sorted(os.listdir(test_dir)):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(test_dir, fname), -1))

        if transform is None:
            if is_train:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEAN, std=STD),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEAN, std=STD),
                ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if self.is_train:
            return img, label
        else:
            return img, os.path.basename(img_path)


def get_train_loader(data_dir, csv_path, batch_size=16, val_split=0.2):
    """返回训练和验证 DataLoader"""
    from torch.utils.data import random_split
    full_dataset = PlantDataset(data_dir, csv_path, is_train=True)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def get_test_loader(data_dir, batch_size=16):
    """返回测试 DataLoader"""
    test_ds = PlantDataset(data_dir, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader