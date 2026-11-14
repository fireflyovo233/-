"""Inference utilities: full-image prediction, TTA, Dice, post-processing."""
import numpy as np
import torch
import torch.nn.functional as F
import cv2


def pad_to_multiple(x, m=16):
    """x: (1,C,H,W) tensor -> padded + pad info."""
    H, W = x.shape[-2:]
    ph = (m - H % m) % m
    pw = (m - W % m) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x, (ph, pw)


def _tta_apply(x, k):
    """8 dihedral views."""
    if k < 4:
        if k & 1:
            x = torch.flip(x, dims=[3])
        if k & 2:
            x = torch.flip(x, dims=[2])
        return x
    y = torch.rot90(x, k - 4, dims=[2, 3])
    return y


def _tta_invert(x, k):
    if k < 4:
        if k & 1:
            x = torch.flip(x, dims=[3])
        if k & 2:
            x = torch.flip(x, dims=[2])
        return x
    return torch.rot90(x, -(k - 4), dims=[2, 3])


@torch.no_grad()
def predict_probs(model, img_chw, n_tta=8, fov=None, scales=(1.0,)):
    """img_chw: HxWx3 float32 -> probability map HxW (float32).

    n_tta: number of dihedral views (1,2,4 or 8); scales: multi-scale factors.
    Probabilities are averaged over (scale x view) after resizing back to HxW.
    """
    x0 = torch.from_numpy(img_chw.transpose(2, 0, 1))[None]
    H, W = x0.shape[-2:]
    acc = torch.zeros(1, 1, H, W)
    ks = [0, 1, 2, 3, 4, 5, 6, 7][:n_tta]
    n = 0
    for s in scales:
        if abs(s - 1.0) < 1e-6:
            x = x0
        else:
            x = F.interpolate(x0, scale_factor=s, mode="bilinear", align_corners=False,
                              recompute_scale_factor=False)
        x, (ph, pw) = pad_to_multiple(x, 16)
        hh, ww = x.shape[-2:]
        for k in ks:
            out = model(_tta_apply(x, k))
            out = torch.sigmoid(_tta_invert(out, k))
            out = out[..., :hh - ph, :ww - pw] if (ph or pw) else out
            if out.shape[-2:] != (H, W):
                out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
            acc += out
            n += 1
    acc = acc / max(1, n)
    p = acc[0, 0].numpy().astype(np.float32)
    if fov is not None:
        p = p * fov
    return p


def dice(pred_bin, gt):
    pred = pred_bin.astype(bool).ravel()
    gt = gt.astype(bool).ravel()
    inter = np.logical_and(pred, gt).sum()
    return float(2.0 * inter / (pred.sum() + gt.sum() + 1e-8))


def postprocess(p, fov=None, thr=0.5, min_size=0, keep_largest=False):
    m = (p > thr).astype(np.uint8)
    if fov is not None:
        m = m * (fov > 0.5).astype(np.uint8)
    if min_size > 1:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                m[lab == i] = 0
    return m
