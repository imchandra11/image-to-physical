"""
CRAFT GT builder: region + affinity maps.

generate_region_affinity_maps((H, W), polys, words, gauss_cfg=None)

Inputs:
  - image_size: (H, W) tuple of the target output map size
  - polys: list of numpy arrays, each of shape (4,2) (float32) - quadrilateral polygon per character
  - words: list of lists, each inner list contains indices (ints) of characters that form a 'word'
           (For character-level only, you can pass [[0], [1], [2], ...] or treat each char separately)
  - gauss_cfg: optional dict with keys:
       - gauss_init_size: baseline size for scale (default 200)
       - gauss_sigma: baseline sigma value (default 40)
       - enlarge_region: [x_mul, y_mul] (default [0.5, 0.5]) -> used for dilation if needed
       - enlarge_affinity: [x_mul, y_mul] (default [0.5, 0.5])
       - min_sigma: minimum sigma (default 1.0)

Returns:
  - region_map, affinity_map : np.ndarray float32 in [0,1], shape (H, W)
"""
from typing import List, Tuple, Sequence, Optional, Dict, Any
import numpy as np
import cv2


def _make_odd(x: int) -> int:
    x = int(x)
    if x % 2 == 0:
        x += 1
    return max(1, x)


def _polygon_to_mask(poly: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Rasterize polygon (Nx2 float) into a binary mask (uint8) of given shape (H, W)."""
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.round(poly).astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _bbox_of_poly(poly: np.ndarray) -> Tuple[int, int, int, int]:
    xs = poly[:, 0]
    ys = poly[:, 1]
    left, top = int(xs.min()), int(ys.min())
    right, bottom = int(xs.max()), int(ys.max())
    return left, top, right, bottom


def _scaled_sigma_for_poly(poly: np.ndarray, gauss_init_size: float, gauss_sigma: float, min_sigma: float = 1.0) -> float:
    """Scale base sigma according to polygon size (use longer edge of bbox)."""
    left, top, right, bottom = _bbox_of_poly(poly)
    w = max(1.0, right - left)
    h = max(1.0, bottom - top)
    long_edge = max(w, h)
    # scale so that sigma is proportional to object size
    sigma = float(gauss_sigma) * (long_edge / float(gauss_init_size))
    return max(float(min_sigma), sigma)


def split_polygon_into_two(poly: np.ndarray, gap_ratio: float = 0.15) -> List[np.ndarray]:
    """
    Split a single quadrilateral polygon into two sub-polygons along its longer axis.
    Adds a gap between the two halves to prevent visual merging after Gaussian blur.

    The annotations in this project are axis-aligned rectangles (4 points),
    so we derive two rectangles from the bounding box:

    - If width >= height  -> split into LEFT and RIGHT halves.
    - If width < height   -> split into TOP and BOTTOM halves.

    Args:
        poly: np.ndarray of shape (4, 2), float32
        gap_ratio: Fraction of the split dimension to use as gap (default 0.15 = 15%)
    """
    if poly.shape != (4, 2):
        poly = poly.reshape(4, 2)

    left, top, right, bottom = _bbox_of_poly(poly.astype(np.float32))
    w = max(1.0, float(right - left))
    h = max(1.0, float(bottom - top))

    if w >= h:
        # Vertical split into left and right halves with gap
        mid_x = (left + right) / 2.0
        gap = w * gap_ratio / 2.0  # Half gap on each side of center
        poly1 = np.array(
            [
                [left, top],
                [mid_x - gap, top],
                [mid_x - gap, bottom],
                [left, bottom],
            ],
            dtype=np.float32,
        )
        poly2 = np.array(
            [
                [mid_x + gap, top],
                [right, top],
                [right, bottom],
                [mid_x + gap, bottom],
            ],
            dtype=np.float32,
        )
    else:
        # Horizontal split into top and bottom halves with gap
        mid_y = (top + bottom) / 2.0
        gap = h * gap_ratio / 2.0  # Half gap on each side of center
        poly1 = np.array(
            [
                [left, top],
                [right, top],
                [right, mid_y - gap],
                [left, mid_y - gap],
            ],
            dtype=np.float32,
        )
        poly2 = np.array(
            [
                [left, mid_y + gap],
                [right, mid_y + gap],
                [right, bottom],
                [left, bottom],
            ],
            dtype=np.float32,
        )

    return [poly1, poly2]


def generate_region_affinity_maps(
    image_size: Tuple[int, int],
    polys: List[np.ndarray],
    words: Optional[List[List[int]]] = None,
    gauss_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build region and affinity maps for given character polygons and word grouping.

    polys: list of (4,2) float32 numpy arrays. Coordinates are assumed to be in the same
           coordinate system as image_size (i.e. pixel coords).
    words: list of lists of indices into `polys`. For character-level only, pass [[0],[1],...].
           If None, each poly is considered an independent 'word' (no affinity created).
    gauss_cfg: see docstring above.
    """
    H, W = image_size
    gauss_cfg = gauss_cfg or {}
    gauss_init_size = float(gauss_cfg.get("gauss_init_size", 200))
    gauss_sigma = float(gauss_cfg.get("gauss_sigma", 40))
    enlarge_region = gauss_cfg.get("enlarge_region", [0.5, 0.5])
    enlarge_affinity = gauss_cfg.get("enlarge_affinity", [0.5, 0.5])
    min_sigma = float(gauss_cfg.get("min_sigma", 1.0))

    # output maps
    region_map = np.zeros((H, W), dtype=np.float32)
    affinity_map = np.zeros((H, W), dtype=np.float32)

    if len(polys) == 0:
        return region_map, affinity_map

    # 1) Build per-character masks, and keep them for affinity creation
    char_masks = []
    for i, poly in enumerate(polys):
        # Clip polygon coordinates to image boundaries
        poly_clipped = poly.copy()
        poly_clipped[:, 0] = np.clip(poly_clipped[:, 0], 0, W - 1)
        poly_clipped[:, 1] = np.clip(poly_clipped[:, 1], 0, H - 1)
        mask = _polygon_to_mask(poly_clipped, (H, W))  # uint8 0/255
        char_masks.append(mask)

    # # 2) Region map: for each character mask, apply Gaussian blur with sigma scaled to polygon size
    # for i, (poly, mask) in enumerate(zip(polys, char_masks)):
    #     sigma = _scaled_sigma_for_poly(poly, gauss_init_size, gauss_sigma, min_sigma=min_sigma)
    #     # OpenCV GaussianBlur accepts ksize or sigma; using ksize=0 and sigmaX=sigma is fine.
    #     # But sigma must be > 0; we ensure min_sigma above.
    #     # convert mask to float (0..1), blur, and accumulate via max-first (to keep sharp peaks)
    #     m_f = (mask.astype(np.float32) / 255.0)
    #     if sigma <= 0.0:
    #         blurred = m_f
    #     else:
    #         blurred = cv2.GaussianBlur(m_f, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    #     # Use max to avoid summing overlapping characters into too-large values
    #     region_map = np.maximum(region_map, blurred)





    # 2) Region map: enlarge polygon slightly before Gaussian blur
    for i, (poly, mask) in enumerate(zip(polys, char_masks)):
        sigma = _scaled_sigma_for_poly(poly, gauss_init_size, gauss_sigma, min_sigma=min_sigma)

        # --- optional enlargement step ---
        if enlarge_region is not None and any(v > 0 for v in enlarge_region):
            # Determine dilation kernel based on bbox size and enlarge_region multipliers
            left, top, right, bottom = _bbox_of_poly(poly)
            w = max(1, right - left)
            h = max(1, bottom - top)
            kx = _make_odd(int(round(w * enlarge_region[0])))
            ky = _make_odd(int(round(h * enlarge_region[1])))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
            mask = cv2.dilate(mask, kernel, iterations=1)

        # --- apply Gaussian blur ---
        m_f = (mask.astype(np.float32) / 255.0)
        blurred = cv2.GaussianBlur(m_f, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
        region_map = np.maximum(region_map, blurred)









    # 3) Affinity map: for each word, connect adjacent characters (i->i+1)
    # If words is None, no affinity is produced. If word contains one index, no affinity for that single char.
    if words is None:
        words = [[i] for i in range(len(polys))]

    for word in words:
        # ensure indices valid
        if not isinstance(word, (list, tuple)) or len(word) <= 1:
            continue
        for idx in range(len(word) - 1):
            i1 = int(word[idx])
            i2 = int(word[idx + 1])
            if i1 < 0 or i2 < 0 or i1 >= len(polys) or i2 >= len(polys):
                continue
            m1 = char_masks[i1]
            m2 = char_masks[i2]

            # union of two masks
            union = ((m1 > 0) | (m2 > 0)).astype(np.uint8) * 255

            # dilate the union to create a linking band
            # kernel size proportional to distance between centroids
            # compute centroids
            y1, x1 = cv2.moments(m1)["m01"], cv2.moments(m1)["m10"]  # careful: we'll compute centroid robustly
            # fallback centroid function (if moments zero)
            def centroid(mask):
                ys, xs = np.where(mask > 0)
                if ys.size == 0:
                    return 0.0, 0.0
                return float(xs.mean()), float(ys.mean())
            c1x, c1y = centroid(m1)
            c2x, c2y = centroid(m2)
            # Euclidean distance
            dx = c2x - c1x
            dy = c2y - c1y
            dist = np.hypot(dx, dy)
            # kernel size (affinity width) scaled by dist and enlarge_affinity factor
            # base kernel proportional to max(3, dist/10)
            k = int(max(3.0, round(dist * 0.1 * max(enlarge_affinity) )))  # heuristic
            k = _make_odd(k)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            dil = cv2.dilate(union, kernel, iterations=1)

            # remove the character areas to get only the link band
            link_band = dil.copy()
            link_band[m1 > 0] = 0
            link_band[m2 > 0] = 0

            if link_band.sum() == 0:
                # fallback: connect by drawing rectangle between centroids
                x_min = int(round(min(c1x, c2x)))
                x_max = int(round(max(c1x, c2x)))
                y_min = int(round(min(c1y, c2y)))
                y_max = int(round(max(c1y, c2y)))
                # small thickness based on character height
                left1, top1, right1, bottom1 = _bbox_of_poly(polys[i1])
                left2, top2, right2, bottom2 = _bbox_of_poly(polys[i2])
                h1 = bottom1 - top1
                h2 = bottom2 - top2
                thickness = max(3, int(round(0.2 * max(h1, h2))))
                cv2.rectangle(link_band, (x_min, y_min - thickness), (x_max, y_max + thickness), 255, thickness=-1)

            # Blur the link band with sigma scaled to average char size
            # compute average character sigma
            sigma1 = _scaled_sigma_for_poly(polys[i1], gauss_init_size, gauss_sigma, min_sigma)
            sigma2 = _scaled_sigma_for_poly(polys[i2], gauss_init_size, gauss_sigma, min_sigma)
            sigma_link = max(1.0, 0.5 * (sigma1 + sigma2))
            link_f = (link_band.astype(np.float32) / 255.0)
            link_blurred = cv2.GaussianBlur(link_f, ksize=(0, 0), sigmaX=sigma_link, sigmaY=sigma_link, borderType=cv2.BORDER_REPLICATE)

            # accumulate via max (keeps peaks)
            affinity_map = np.maximum(affinity_map, link_blurred)

    # Optionally, we might attenuate affinity where region is very strong to avoid double counting.
    # But keep both maps independent for BCE loss as in original CRAFT (paper uses separate supervision).

    # Clip to [0,1]
    region_map = np.clip(region_map, 0.0, 1.0).astype(np.float32)
    affinity_map = np.clip(affinity_map, 0.0, 1.0).astype(np.float32)

    return region_map, affinity_map





