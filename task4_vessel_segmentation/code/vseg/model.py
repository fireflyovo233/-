"""Retinal vessel segmentation models (pure PyTorch, CPU friendly)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(cin, cout, k=3, s=1, p=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """Two 3x3 convs + identity (projection when channels change)."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                       nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        x = self.act(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        return self.act(x + idt)


class Up(nn.Module):
    def __init__(self, cin, cskip, cout, mode="bilinear"):
        super().__init__()
        self.mode = mode
        if mode == "bilinear":
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.reduce = nn.Conv2d(cin, cout, 1, bias=False)
        else:
            self.up = nn.ConvTranspose2d(cin, cout, 2, 2)
            self.reduce = nn.Identity()
        self.block = ResBlock(cout + cskip, cout)

    def forward(self, x, skip):
        x = self.reduce(self.up(x))
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ResUNet(nn.Module):
    """U-Net with residual blocks. width=base channels, depth = #downsamples."""

    def __init__(self, in_ch=3, out_ch=1, width=32, depth=4, dilate_bottleneck=True):
        super().__init__()
        chs = [width * (2 ** i) for i in range(depth + 1)]
        self.stem = nn.Sequential(conv_bn_act(in_ch, chs[0]), ResBlock(chs[0], chs[0]))
        self.enc = nn.ModuleList()
        for i in range(depth):
            self.enc.append(nn.Sequential(nn.MaxPool2d(2), ResBlock(chs[i], chs[i + 1])))
        # dilated bottleneck for larger receptive field on thin elongated structures
        if dilate_bottleneck:
            self.mid = nn.Sequential(
                nn.Conv2d(chs[-1], chs[-1], 3, 1, 2, dilation=2, bias=False),
                nn.BatchNorm2d(chs[-1]), nn.ReLU(inplace=True),
                nn.Conv2d(chs[-1], chs[-1], 3, 1, 4, dilation=4, bias=False),
                nn.BatchNorm2d(chs[-1]), nn.ReLU(inplace=True),
            )
        else:
            self.mid = nn.Identity()
        self.dec = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.dec.append(Up(chs[i], chs[i - 1], chs[i - 1]))
        self.head = nn.Sequential(conv_bn_act(chs[0], chs[0]), nn.Conv2d(chs[0], out_ch, 1))

    def forward(self, x):
        x = self.stem(x)
        skips = [x]
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
        x = self.mid(skips[-1])
        for j, up in enumerate(self.dec):
            x = up(x, skips[-2 - j])
        return self.head(x)


# ---------------------------------------------------------------- losses
def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def _soft_erode(x):
    return -F.max_pool2d(-x, 3, 1, 1)


def _soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def soft_skel(x, iters=8):
    x1 = _soft_dilate(_soft_erode(x))
    for _ in range(iters - 1):
        x1 = _soft_dilate(_soft_erode(x1))
    return x1


def cl_dice_loss(logits, target, iters=6, eps=1.0):
    """centerline Dice (Shit et al. 2021) - emphasises topology of thin vessels."""
    p = torch.sigmoid(logits)
    skel_pred = soft_skel(p, iters)
    skel_true = soft_skel(target, iters)
    tprec = (torch.sum(skel_pred * target, dim=(1, 2, 3)) + eps) / (
        torch.sum(skel_pred, dim=(1, 2, 3)) + eps)
    tsens = (torch.sum(skel_true * p, dim=(1, 2, 3)) + eps) / (
        torch.sum(skel_true, dim=(1, 2, 3)) + eps)
    return (1 - 2 * tprec * tsens / (tprec + tsens)).mean()


class ComboLoss(nn.Module):
    def __init__(self, w_bce=0.5, w_dice=0.5, w_cl=0.0, pos_weight=2.0, cl_iters=6):
        super().__init__()
        self.register_buffer("pw", torch.tensor(float(pos_weight)))
        self.w = (w_bce, w_dice, w_cl)
        self.iters = cl_iters

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pw)
        d = soft_dice_loss(logits, target)
        loss = self.w[0] * bce + self.w[1] * d
        if self.w[2] > 0:
            loss = loss + self.w[2] * cl_dice_loss(logits, target, self.iters)
        return loss
