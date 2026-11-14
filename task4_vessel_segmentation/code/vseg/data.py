"""Data pipeline: DRIVE-style retinal fundus images, CLAHE green channel, patch sampling."""
import os
import glob
import cv2
import numpy as np

cv2.setNumThreads(1)

DATA_ROOT = r"D:/机器学习课设/task4/根文件/2026task-4/segmentation"


def list_ids(split="train"):
    d = os.path.join(DATA_ROOT, split, "image")
    files = glob.glob(os.path.join(d, "*.jpg"))
    return sorted(files, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))


def fov_mask(rgb):
    """Field of view of the fundus circle (robust, no external files)."""
    g = rgb[:, :, 1]
    m = (g > 12).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == keep).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    # fill holes
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cnts, -1, 1, -1)
    return out.astype(np.float32)


def preprocess(rgb):
    """rgb uint8 -> normalized 3-channel float32 [clahe(G), G, B]."""
    g = rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gc = clahe.apply(g).astype(np.float32) / 255.0
    g0 = g.astype(np.float32) / 255.0
    b0 = rgb[:, :, 2].astype(np.float32) / 255.0
    x = np.stack([gc, g0, b0], axis=-1)
    # per-image, per-channel z-score (robust to illumination differences)
    for c in range(3):
        ch = x[:, :, c]
        mu, sd = float(ch.mean()), float(ch.std())
        x[:, :, c] = (ch - mu) / (sd + 1e-6)
    return x.astype(np.float32)


def imread_u(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread replacement that supports non-ASCII (Chinese) paths on Windows."""
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise IOError("cannot read " + path)
    return img


class RetinaSet:
    """Holds all images in memory (20 x 584 x 565 is tiny)."""

    def __init__(self, split="train", with_labels=True):
        self.paths = list_ids(split)
        self.ids = [os.path.splitext(os.path.basename(p))[0] for p in self.paths]
        self.with_labels = with_labels
        self.imgs, self.labels, self.fovs, self.raws = [], [], [], []
        for p in self.paths:
            bgr = imread_u(p, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.raws.append(rgb)
            self.imgs.append(preprocess(rgb))
            self.fovs.append(fov_mask(rgb))
            if with_labels:
                lp = p.replace("/image/", "/label/").replace(os.sep + "image" + os.sep,
                                                             os.sep + "label" + os.sep)
                lab = imread_u(lp, cv2.IMREAD_GRAYSCALE)
                self.labels.append((lab > 127).astype(np.float32))
        self.vessel_pts = []
        for i in range(len(self.imgs)):
            if with_labels:
                ys, xs = np.nonzero(self.labels[i] > 0.5)
                self.vessel_pts.append(np.stack([xs, ys], axis=1))
            else:
                self.vessel_pts.append(None)

    def __len__(self):
        return len(self.imgs)

    def shape(self, i):
        return self.imgs[i].shape[:2]


def random_patch(imgs, labels, vpts, idx, size=128, rng=None):
    """Sample one augmented patch from image `idx`."""
    rng = rng or np.random
    H, W = imgs[idx].shape[:2]
    if rng.rand() < 0.55 and vpts is not None and len(vpts[idx]) > 0:
        cx, cy = vpts[idx][rng.randint(len(vpts[idx]))]
        cx, cy = int(cx), int(cy)
    else:
        cx, cy = rng.randint(W), rng.randint(H)

    ang = rng.uniform(-180, 180)
    scale = rng.uniform(0.85, 1.25)
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), ang, scale)
    M[0, 2] += size / 2.0 - cx
    M[1, 2] += size / 2.0 - cy

    img = cv2.warpAffine(imgs[idx], M, (size, size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    lab = cv2.warpAffine(labels[idx], M, (size, size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101) if labels is not None else None

    # flips / rot90
    if rng.rand() < 0.5:
        img, lab = img[:, ::-1], (lab[:, ::-1] if lab is not None else None)
    if rng.rand() < 0.5:
        img, lab = img[::-1], (lab[::-1] if lab is not None else None)
    if rng.rand() < 0.5:
        k = rng.randint(1, 4)
        img = np.rot90(img, k)
        lab = np.rot90(lab, k) if lab is not None else None

    img = np.ascontiguousarray(img, dtype=np.float32)
    # ---- photometric: per-channel colour jitter (robust to camera/illumination shift)
    if rng.rand() < 0.5:
        img = img * rng.uniform(0.92, 1.08, size=(1, 1, 3)).astype(np.float32)     # per-channel gain
    if rng.rand() < 0.4:
        img = img + rng.uniform(-0.08, 0.08, size=(1, 1, 3)).astype(np.float32)    # per-channel offset
    # global gain / gamma / shift
    if rng.rand() < 0.7:
        img = img * rng.uniform(0.85, 1.18)
    if rng.rand() < 0.5:
        gamma = rng.uniform(0.70, 1.40)
        img = np.sign(img) * np.power(np.abs(img), gamma)
    if rng.rand() < 0.4:
        img = img + rng.uniform(-0.12, 0.12) * (1.0 if rng.rand() < 0.5 else -1.0)
    # illumination gradient (uneven retina lighting)
    if rng.rand() < 0.25:
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        ang = rng.uniform(0, 2 * np.pi)
        ramp = (np.cos(ang) * xx + np.sin(ang) * yy) / size - 0.5
        img = img + (rng.uniform(-0.25, 0.25) * ramp)[:, :, None]
    # blur / noise / compression-ish artefacts (test JPEGs can be softer)
    if rng.rand() < 0.40:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.4))
    if rng.rand() < 0.30:
        img = img + rng.normal(0, rng.uniform(0.01, 0.07), img.shape).astype(np.float32)

    img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
    if lab is not None:
        lab = (np.ascontiguousarray(lab)[None] > 0.5).astype(np.float32)
    return img, lab


def make_batch(imgs, labels, vpts, size, bs, rng=None, hard_ids=None):
    rng = rng or np.random
    xs, ys = [], []
    n_img = min(len(imgs), max(2, bs // 2))
    pool = hard_ids if (hard_ids is not None and rng.rand() < 0.4) else list(range(len(imgs)))
    for j in range(bs):
        idx = int(rng.choice(pool))
        x, y = random_patch(imgs, labels, vpts, idx, size, rng)
        xs.append(x)
        ys.append(y)
    return np.stack(xs), np.stack(ys)
