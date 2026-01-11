"""
Augmentations that operate on BGR numpy images + polygon lists.
Transforms return (img_bgr, polygons_transformed).

Usage:
  transform = get_transform(cfg_data, is_train=True)
  img_aug, polys_aug = transform(img_bgr, polys_orig)
"""

import random
from typing import List, Callable, Tuple, Dict, Any
import numpy as np
import cv2


# =============================
# Geometry Transform Utilities
# =============================

def _transform_polys_with_matrix(polys: List[np.ndarray], M: np.ndarray) -> List[np.ndarray]:
    """Apply a 2x3 affine transform matrix to a list of polygons."""
    res = []
    A = M[:, :2]  # rotation + scale
    b = M[:, 2]   # translation
    for p in polys:
        if p.size == 0:
            res.append(p)
            continue
        p2 = p.astype(np.float32)
        p_trans = p2.dot(A.T) + b.reshape(1, 2)
        res.append(p_trans.astype(np.float32))
    return res


def random_scale(img: np.ndarray, polys: List[np.ndarray], scale_range: List[float]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Randomly scale image and polygons by a uniform factor."""
    if not scale_range:
        return img, polys
    s = float(random.choice(scale_range) if isinstance(scale_range, (list, tuple)) else scale_range)
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * s)))
    new_h = max(1, int(round(h * s)))
    img_s = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    polys_s = [(p * s).astype(np.float32) for p in polys]
    return img_s, polys_s


def random_rotate(img: np.ndarray, polys: List[np.ndarray], max_angle: float) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Rotate image and polygons by the same random angle."""
    if max_angle <= 0:
        return img, polys
    h, w = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_r = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    polys_r = _transform_polys_with_matrix(polys, M)
    return img_r, polys_r


def random_horizontal_flip(img: np.ndarray, polys: List[np.ndarray], p: float = 0.5) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Randomly flip image and polygons horizontally with probability p."""
    if random.random() > p:
        return img, polys
    h, w = img.shape[:2]
    img_f = cv2.flip(img, 1)
    polys_f = []
    for ppoly in polys:
        if ppoly.size == 0:
            polys_f.append(ppoly)
            continue
        p2 = ppoly.copy()
        p2[:, 0] = (w - 1) - p2[:, 0]
        polys_f.append(p2.astype(np.float32))
    return img_f, polys_f


def random_crop_keep_polys(img: np.ndarray, polys: List[np.ndarray],
                           scale_range=(0.7, 1.0), attempts=8) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Randomly crop a region of the image while trying to keep at least one polygon inside.
    Returns cropped image and shifted polygons.
    """
    h, w = img.shape[:2]
    for _ in range(attempts):
        scale = random.uniform(scale_range[0], scale_range[1])
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        if new_h >= h or new_w >= w:
            continue
        y0 = random.randint(0, max(0, h - new_h))
        x0 = random.randint(0, max(0, w - new_w))
        kept_polys = []
        for poly in polys:
            if poly.size == 0:
                continue
            cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
            if x0 <= cx < x0 + new_w and y0 <= cy < y0 + new_h:
                pnew = poly.copy()
                pnew[:, 0] -= x0
                pnew[:, 1] -= y0
                kept_polys.append(pnew.astype(np.float32))
        if len(kept_polys) == 0:
            continue
        crop = img[y0:y0 + new_h, x0:x0 + new_w].copy()
        return crop, kept_polys
    return img, polys


# =============================
# Color / Photometric Augments
# =============================

def color_jitter_bgr(img: np.ndarray, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02):
    """
    Simple color jitter for BGR images (OpenCV): adjust brightness, contrast, saturation, and hue.
    """
    if img.ndim == 2:
        return img
    out = img.astype(np.float32) / 255.0

    # brightness
    bv = random.uniform(-brightness, brightness)
    out = out * (1.0 + bv)

    # contrast
    cv = random.uniform(-contrast, contrast)
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = (out - mean) * (1.0 + cv) + mean

    # saturation + hue
    if saturation > 0 or hue > 0:
        hsv = cv2.cvtColor((np.clip(out, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        if saturation > 0:
            sv = random.uniform(-saturation, saturation)
            hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + sv), 0, 255)
        if hue > 0:
            hv = random.uniform(-hue, hue) * 180.0
            hsv[..., 0] = (hsv[..., 0] + hv) % 180.0
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


# =============================
# Compose Pipeline from YAML
# =============================

def get_transform(data_cfg: Dict[str, Any], is_train: bool = True) -> Callable:
    """
    Build a transform callable from data_cfg (portion of YAML).
    The returned callable has signature: (img_bgr, polys) -> (img_bgr, polys)
    """
    def _identity(img, polys):
        return img, polys

    if data_cfg is None:
        return _identity

    aug_cfg = data_cfg.get("custom_aug" if is_train else "test_aug", data_cfg.get("custom_aug", {}))
    syn_aug = data_cfg.get("syn_aug", {})
    ops = []

    # order: scale -> rotate -> crop -> flip -> color jitter
    if is_train and aug_cfg.get("random_scale", {}).get("option", False):
        rs = aug_cfg["random_scale"].get("range", [1.0])
        ops.append(lambda img, polys: random_scale(img, polys, rs))

    if is_train and aug_cfg.get("random_rotate", {}).get("option", False):
        max_angle = float(aug_cfg["random_rotate"].get("max_angle", 20))
        ops.append(lambda img, polys: random_rotate(img, polys, max_angle))

    if is_train and aug_cfg.get("random_crop", {}).get("option", False):
        scale = tuple(aug_cfg["random_crop"].get("scale", [0.7, 0.9]))
        ops.append(lambda img, polys: random_crop_keep_polys(img, polys, scale_range=scale))

    if is_train and aug_cfg.get("random_horizontal_flip", {}).get("option", False):
        ops.append(lambda img, polys: random_horizontal_flip(img, polys, p=0.5))

    if is_train and aug_cfg.get("random_colorjitter", {}).get("option", False):
        cj = aug_cfg["random_colorjitter"]
        b = cj.get("brightness", 0.2)
        c = cj.get("contrast", 0.2)
        s = cj.get("saturation", 0.2)
        h = cj.get("hue", 0.02)
        ops.append(lambda img, polys: (color_jitter_bgr(img, brightness=b, contrast=c, saturation=s, hue=h), polys))

    def composed_transform(img: np.ndarray, polys: List[np.ndarray]):
        im, ps = img, polys
        for op in ops:
            im, ps = op(im, ps)
        return im, ps

    return composed_transform
