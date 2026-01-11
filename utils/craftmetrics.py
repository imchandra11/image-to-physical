# utils/craftmetrics.py

import itertools
from typing import List, Tuple, Dict, Any

try:
    from shapely.geometry import Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    Polygon = None
    SHAPELY_AVAILABLE = False


class CraftMetrics:
    """
    Evaluate text detection using IoS-only matching logic.

    - For each ground truth polygon, compute IoS with all predicted polygons.
    - Count as TP if IoS exceeds threshold.
    - One prediction can match multiple GTs if IoS condition is satisfied.
    """

    def __init__(self, ios_threshold: float = 0.05, debug: bool = False):
        self.ios_threshold = ios_threshold
        self.debug = debug
        self.reset()

    def reset(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.image_count = 0
        self.details = [] if self.debug else None

    def compute_ios(self, pred_poly: List[Tuple[float, float]], gt_poly: List[Tuple[float, float]]) -> float:
        """
        Computes the Intersection-over-Smaller (IoS) area metric between two polygons.
        """
        if SHAPELY_AVAILABLE:
            try:
                pred_p = Polygon(pred_poly)
                gt_p = Polygon(gt_poly)
                if not pred_p.is_valid or not gt_p.is_valid:
                    return 0.0
                inter_area = pred_p.intersection(gt_p).area
                min_area = min(pred_p.area, gt_p.area)
                return inter_area / min_area if min_area > 0 else 0.0
            except Exception:
                return 0.0
        else:
            # Fallback AABB IoS
            def get_aabb(poly):
                xs, ys = zip(*poly)
                return min(xs), min(ys), max(xs), max(ys)

            def area(box):
                xmin, ymin, xmax, ymax = box
                return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)

            def intersection(box1, box2):
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                return w * h

            box_pred = get_aabb(pred_poly)
            box_gt = get_aabb(gt_poly)
            inter = intersection(box_pred, box_gt)
            return inter / min(area(box_pred), area(box_gt)) if min(area(box_pred), area(box_gt)) > 0 else 0.0

    def update(self, preds: List[List[Tuple[float, float]]], gts: List[List[Tuple[float, float]]], img_id: str = None):
        """
        Updates internal counters based on a single image's predictions and ground truths.
        """
        true_positive = 0
        false_positive_flags = list(itertools.repeat(True, len(preds)))
        false_negative_flags = list(itertools.repeat(True, len(gts)))
        match_records = []

        for gt_idx, gt in enumerate(gts):
            for pred_idx, pred in enumerate(preds):
                ios_val = self.compute_ios(pred, gt)
                if ios_val > self.ios_threshold:
                    true_positive += 1
                    false_positive_flags[pred_idx] = False
                    false_negative_flags[gt_idx] = False
                    if self.debug:
                        match_records.append((gt_idx, pred_idx, ios_val))

        false_positive = sum(false_positive_flags)
        false_negative = sum(false_negative_flags)

        self.total_tp += true_positive
        self.total_fp += false_positive
        self.total_fn += false_negative
        self.image_count += 1

        if self.debug:
            self.details.append({
                "image": img_id,
                "matches": match_records,
                "tp": true_positive,
                "fp": false_positive,
                "fn": false_negative,
            })

    def compute(self) -> Dict[str, Any]:
        tp = self.total_tp
        fp = self.total_fp
        fn = self.total_fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "hmean": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "num_images": self.image_count,
            "details" if self.debug else "_": self.details if self.debug else None
        }







"""Test CraftMetrics with synthetic data.
This test creates synthetic ground truth and predicted polygons for multiple images,
then computes detection metrics using IoS-based matching."""


import random
# from utils.craftmetrics import CraftMetrics

# Set random seed for reproducibility
random.seed(42)

def generate_polygon(x, y, w=50, h=30):
    """Generate a 4-point rectangular polygon from top-left (x,y)"""
    return [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h)
    ]

def generate_image_data(gt_count=20, pred_count=21, offset_range=15):
    """Generate a set of ground truth and predicted polygons for one image"""
    gt_polys = []
    pred_polys = []
    for i in range(gt_count):
        x = random.randint(0, 300)
        y = random.randint(0, 300)
        gt = generate_polygon(x, y)
        gt_polys.append(gt)

    for i in range(pred_count):
        # Some predictions close to GT, others randomly placed
        if i < int(0.8 * pred_count):
            gt = gt_polys[random.randint(0, gt_count - 1)]
            dx = random.randint(-offset_range, offset_range)
            dy = random.randint(-offset_range, offset_range)
            pred = [(x + dx, y + dy) for (x, y) in gt]
        else:
            x = random.randint(0, 300)
            y = random.randint(0, 300)
            pred = generate_polygon(x, y)
        pred_polys.append(pred)
    return gt_polys, pred_polys

def test_metrics_over_dataset():
    metrics = CraftMetrics(ios_threshold=0.5, debug=True)
    for img_idx in range(1, 10):
        gts, preds = generate_image_data()
        img_id = f"image_{img_idx:02d}.jpg"
        metrics.update(preds=preds, gts=gts, img_id=img_id)

    result = metrics.compute()

    print("\n📊 Final Metrics Summary (IoS-only matching):")
    for k, v in result.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

if __name__ == "__main__":
    test_metrics_over_dataset()
