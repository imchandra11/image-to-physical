# utils/craft_postprocess.py
import cv2
import numpy as np
from typing import List, Tuple, Optional

def _order_quad_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points clockwise starting from top-left.
    Input: (4,2) float array (may be any order).
    Output: (4,2) float array ordered.
    """
    # compute centroid
    c = pts.mean(axis=0)
    # compute angles
    angles = np.arctan2(pts[:,1] - c[1], pts[:,0] - c[0])
    # sort by angle (clockwise)
    order = np.argsort(angles)
    ordered = pts[order]
    # put top-left first: choose point with smallest (x+y)
    sums = ordered.sum(axis=1)
    idx0 = int(np.argmin(sums))
    ordered = np.roll(ordered, -idx0, axis=0)
    return ordered

def _contour_to_quad(contour: np.ndarray, approx_eps_ratio: float = 0.01) -> np.ndarray:
    """
    Try to convert a contour (Nx1x2) into a 4-point polygon.
    1) approxPolyDP to get polygon with small epsilon; if it has 4 vertices -> good
    2) else, use minAreaRect -> 4 points
    Returns (4,2) float32 array.
    """
    cnt = contour.reshape(-1, 2).astype(np.float32)
    peri = cv2.arcLength(cnt, True)
    eps = max(1.0, approx_eps_ratio * peri)
    approx = cv2.approxPolyDP(cnt, eps, True)
    if approx.shape[0] == 4:
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        # fallback to minAreaRect
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)  # 4x2
        quad = box.astype(np.float32)
    quad = _order_quad_clockwise(quad)
    return quad

def craft_watershed(
    region_score: np.ndarray,
    affinity_score: np.ndarray,
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
    min_area: int = 10,
    debug_save: Optional[str] = None
) -> List[Tuple[np.ndarray, float]]:
    """
    Official CRAFT-style post-processing using watershed.

    Args:
      region_score: (H,W) float in [0,1]
      affinity_score: (H,W) float in [0,1]
      text_threshold: threshold to decide "text core"
      link_threshold: threshold for "affinity / links"
      low_text: lower threshold used to create mask for combined map
      min_area: minimal connected component area to keep
      debug_save: optional path prefix to save debug masks (for inspection)

    Returns:
      list of tuples: (polygon (4x2 float numpy array), score float)
      polygon coordinates are in same coordinate system as the input maps (caller must scale to original image).
    """

    assert region_score.ndim == 2 and affinity_score.ndim == 2
    H, W = region_score.shape

    # 1) threshold maps
    textmap = (region_score > text_threshold).astype(np.uint8)
    linkmap = (affinity_score > link_threshold).astype(np.uint8)
    low_textmap = (region_score > low_text).astype(np.uint8)

    # 2) combined mask (low_text OR link)
    combined = np.clip(low_textmap + linkmap, 0, 1).astype(np.uint8)

    # morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    textmap = cv2.morphologyEx(textmap, cv2.MORPH_OPEN, kernel, iterations=1)

    # debug saves
    if debug_save:
        cv2.imwrite(debug_save + "_region.png", (region_score * 255).astype(np.uint8))
        cv2.imwrite(debug_save + "_affinity.png", (affinity_score * 255).astype(np.uint8))
        cv2.imwrite(debug_save + "_textmap.png", (textmap * 255).astype(np.uint8))
        cv2.imwrite(debug_save + "_linkmap.png", (linkmap * 255).astype(np.uint8))
        cv2.imwrite(debug_save + "_combined.png", (combined * 255).astype(np.uint8))

    # 3) distance transform on core textmap to find seeds
    # Use distance on textmap (strong cores)
    if textmap.sum() == 0:
        return []

    dist = cv2.distanceTransform(textmap, distanceType=cv2.DIST_L2, maskSize=5)
    # markers from peaks: threshold fraction of max distance
    maxd = dist.max() if dist.size else 0.0
    if maxd <= 0:
        # fallback: connected components on textmap
        seeds = textmap.copy()
    else:
        _, sure_fg = cv2.threshold(dist, 0.4 * maxd, 1, 0)  # float -> binary (0/1)
        sure_fg = sure_fg.astype(np.uint8)

        # optionally dilate to connect close peaks inside same word
        sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_DILATE, kernel, iterations=1)
        seeds = sure_fg

    # 4) Markers: label seeds (connected components)
    ret, markers = cv2.connectedComponents(seeds)
    # markers: 0 = background, 1..N seeds
    # We need an int32 marker image for watershed, where unknown regions = 0
    # unknown regions = combined - seeds
    unknown = combined.copy()
    unknown[seeds > 0] = 0
    # shift markers so background is 1 for watershed
    markers = markers.astype(np.int32)
    markers = markers + 1  # ensure background (was 0) becomes 1
    markers[unknown == 1] = 0  # unknown area set to 0

    # 5) Watershed requires a 3-channel uint8 image
    ws_img = (combined * 255).astype(np.uint8)
    ws_bgr = cv2.cvtColor(ws_img, cv2.COLOR_GRAY2BGR)
    # apply watershed
    try:
        cv2.watershed(ws_bgr, markers)
    except Exception:
        # If watershed fails for some reason, fallback to connected components on combined
        markers = cv2.connectedComponents(combined)[1].astype(np.int32)

    # markers now contain -1 for borders, 1..K labels for regions
    unique_labels = np.unique(markers)
    polys_scores = []

    for lab in unique_labels:
        if lab <= 1:
            # skip background (1) and negative/border labels
            continue
        mask = (markers == lab).astype(np.uint8)
        # ignore tiny regions
        area = int(mask.sum())
        if area < min_area:
            continue

        # find contours, pick largest
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        # ensure contour area is large enough
        if cv2.contourArea(cnt) < min_area:
            continue

        # try to get accurate quad
        quad = _contour_to_quad(cnt, approx_eps_ratio=0.02)

        # compute score = mean region_score inside mask
        mask_bool = (mask > 0)
        if mask_bool.sum() == 0:
            score = float(region_score[mask_bool].mean()) if mask_bool.sum() else 0.0
        else:
            score = float(region_score[mask_bool].mean())

        polys_scores.append((quad, float(score)))

    return polys_scores
