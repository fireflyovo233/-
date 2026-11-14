# -*- coding: utf-8 -*-
"""
植物幼苗分类 - ResNet34 迁移学习模型
"""
import torch
import torch.nn as nn
from torchvision import models


class PlantClassifier(nn.Module):
    """基于 ResNet34 的植物分类器（ImageNet 预训练 + 微调）"""

    def __init__(self, num_classes=5, freeze_backbone=True):
        super().__init__()
        # 加载预训练 ResNet34
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # 冻结浅层，仅微调 layer3/layer4
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False

        # 替换分类头
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def create_model(num_classes=5, freeze_backbone=True, device='cpu'):
    """工厂函数：创建并返回模型实例"""
    model = PlantClassifier(num_classes=num_classes, freeze_backbone=freeze_backbone)
    model = model.to(device)
    return model