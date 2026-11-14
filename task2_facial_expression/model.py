"""面向 48x48 灰度人脸表情的轻量 ResNet（含 SE 注意力）。

设计取舍：输入只有 48x48，若按 ImageNet 的 56x56 起点逐层下采样，前两层
特征图过大、CPU 上算力开销不合理。这里让 stem 直接 stride=2 到 24x24，
用 3 个 stage（24/12/6）覆盖“局部纹理→五官部件→整体构型”的层次。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        hidden = max(ch // r, 4)
        self.fc1 = nn.Conv2d(ch, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, ch, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True)))
        return x * s


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.se = SE(cout) if use_se else nn.Identity()
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + idt, inplace=True)


class FERNet(nn.Module):
    def __init__(self, num_classes: int = 6, width: int = 32, drop: float = 0.3,
                 blocks: tuple[int, ...] = (2, 2, 2)):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(1, w, 3, 2, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
        )
        chans = [w, w * 2, w * 4]
        layers, cin = [], w
        for i, (cout, n) in enumerate(zip(chans, blocks)):
            layers.append(self._stage(cin, cout, n, 1 if i == 0 else 2))
            cin = cout
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop),
            nn.Linear(cin, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _stage(cin, cout, n, stride):
        blocks = [BasicBlock(cin, cout, stride)]
        blocks += [BasicBlock(cout, cout, 1) for _ in range(n - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        return self.head(x)
