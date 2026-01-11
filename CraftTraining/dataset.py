import os
from typing import List, Dict, Tuple, Optional, Any, Callable
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils.craftImageaugmentation import get_transform
from utils.craftTarget import generate_region_affinity_maps

IMAGE_INPUT_FOLDER = "Input"
IMAGE_OUTPUT_FOLDER = "Output"
IMAGE_EXTN = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

def _parse_polygon_line(line: str) -> Dict[str, Any]:
    line = line.replace("\ufeff", "").strip()
    if not line:
        raise ValueError("Empty label line")
    parts = line.split(",")
    if len(parts) < 9:
        raise ValueError(f"Malformed line (expected 9+ fields): {line}")
    coords = list(map(float, parts[:8]))
    transcription = ",".join(parts[8:]).strip().strip('"').strip("'")
    poly = np.array(coords, dtype=np.float32).reshape(4, 2)
    return {"poly": poly, "text": transcription}

def _polygon_to_axis_aligned_bbox(poly: np.ndarray) -> Tuple[int, int, int, int]:
    xs = poly[:, 0]
    ys = poly[:, 1]
    return int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))

class CraftDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        resize: Optional[int] = None,
        transforms: Optional[Callable[[np.ndarray, List[np.ndarray]], Tuple[np.ndarray, List[np.ndarray]]]] = None,
        gauss_cfg: Optional[Dict[str, Any]] = None,
        is_train: bool = True,
        data_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        input_path = os.path.join(self.data_dir, IMAGE_INPUT_FOLDER)
        self.input_path = input_path if os.path.exists(input_path) else self.data_dir
        output_path = os.path.join(self.data_dir, IMAGE_OUTPUT_FOLDER)
        self.output_path = output_path if os.path.exists(output_path) else self.data_dir


        print(f'\nLoading all {IMAGE_EXTN} from {self.input_path}')

        # Build transforms
        self.transforms = transforms or get_transform(data_cfg, is_train=is_train)
        self.files = [f for f in sorted(os.listdir(self.input_path)) if f.lower().endswith(IMAGE_EXTN)]
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {self.input_path}")
        print(f"Found {len(self.files)} images.\n")

        self.resize = resize
        self.gauss_cfg = gauss_cfg or {}

    def __len__(self):
        return len(self.files)

    def _read_label_file(self, image_name: str) -> List[Dict[str, Any]]:
        base = os.path.splitext(image_name)[0]
        txt_path = os.path.join(self.output_path, f"gt_{base}.txt")
        polys = []
        if not os.path.exists(txt_path):
            return polys
        with open(txt_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = _parse_polygon_line(line)
                    polys.append(parsed)
                except Exception:
                    continue
        return polys

    def _resize_keep_aspect_and_pad(self, img_gray: np.ndarray, target_long: int):
        """Resize while keeping aspect ratio and center the image on a square canvas."""
        h0, w0 = img_gray.shape[:2]
        long_edge = max(h0, w0)
        scale = float(target_long) / float(long_edge)
        new_w = int(round(w0 * scale))
        new_h = int(round(h0 * scale))
        resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((target_long, target_long), dtype=resized.dtype)
        y_offset = (target_long - new_h) // 2
        x_offset = (target_long - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        pad_offset = (x_offset, y_offset)
        return canvas, scale, pad_offset

    def __getitem__(self, idx: int):
        image_name = self.files[idx]
        image_path = os.path.join(self.input_path, image_name)
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read {image_path}")
        orig_h, orig_w = img_bgr.shape[:2]

        parsed = self._read_label_file(image_name)
        polys_orig = [p["poly"].astype(np.float32) for p in parsed]

        # Apply augmentations
        if self.transforms is not None:
            try:
                img_bgr, polys_orig = self.transforms(img_bgr, polys_orig)
            except Exception as e:
                print(f"Warning: transform failed for {image_name}: {e}")

        img_gray_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Resize and center pad
        scale = 1.0
        pad_offset = (0, 0)
        if self.resize is not None:
            img_gray_proc, scale, pad_offset = self._resize_keep_aspect_and_pad(img_gray_raw, self.resize)
            polys_scaled = [(p * scale + np.array(pad_offset)).astype(np.float32) for p in polys_orig]
            canvas_h = canvas_w = self.resize
        else:
            img_gray_proc = img_gray_raw.copy()
            polys_scaled = [p.astype(np.float32) for p in polys_orig]
            canvas_h, canvas_w = img_gray_proc.shape[:2]

        words = [[i] for i in range(len(polys_scaled))]
        region_map, affinity_map = generate_region_affinity_maps(
            (canvas_h, canvas_w), polys_scaled, words, gauss_cfg=self.gauss_cfg
        )

        img_tensor = torch.from_numpy(img_gray_proc.astype(np.float32) / 255.0).unsqueeze(0)
        region_tensor = torch.from_numpy(region_map).unsqueeze(0).float()
        affinity_tensor = torch.from_numpy(affinity_map).unsqueeze(0).float()

        boxes = []
        for p in polys_scaled:
            l, t, r, b = _polygon_to_axis_aligned_bbox(p)
            boxes.append([l, t, r, b])
        boxes_tensor = torch.tensor(boxes, dtype=torch.int32) if boxes else torch.zeros((0, 4), dtype=torch.int32)

        target = {
            "region": region_tensor,
            "affinity": affinity_tensor,
            "boxes": boxes_tensor,
            "polys": polys_scaled,
            "image_name": image_name,
            "orig_size": (orig_w, orig_h),
            "curr_size": (canvas_w, canvas_h),
            "scale": scale,
            "pad_offset": pad_offset,
            "raw_image": img_gray_raw,
        }
        return img_tensor, target, image_name
