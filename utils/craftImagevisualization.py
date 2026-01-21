import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

LINE_THICKNESS = 2
TEXT_THICKNESS = 1
TEXT_SIZE = 0.7
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_LINE_TYPE = cv2.LINE_AA


def drawCV2BBWithText(image, box, text=None, color=(255, 0, 0),
                      bbThickness=LINE_THICKNESS,
                      textThickness=TEXT_THICKNESS, textSize=TEXT_SIZE):
    """
    Draw bounding box with optional text on the image.

    Args:
        image: numpy array (H, W, 3)
        box: [left, top, right, bottom]
        text: Optional text string (e.g., class name or score)
        color: Box/text color (B, G, R)
    """
    left, top, right, bottom = map(int, box)
    cv2.rectangle(image, (left, top), (right, bottom), color, bbThickness)
    if text:
        cv2.putText(image, text,
                    (left, max(0, top - 5)),
                    TEXT_FONT, textSize, color,
                    textThickness, lineType=TEXT_LINE_TYPE)
    return image


def save_prediction_visual(image_path, boxes, scores, out_path):
    """
    Save prediction results (boxes + scores) drawn on the original image.

    Args:
        image_path: Path to input image
        boxes: list of [l, t, r, b]
        scores: list of float scores
        out_path: where to save output image
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    for i, box in enumerate(boxes):
        score = scores[i] if i < len(scores) else 0.0
        drawCV2BBWithText(img, box, f"{score:.2f}", (255, 0, 0))
    cv2.imwrite(out_path, img)


def visualizeOneBatchImages(batch, max_images=5):
    """
    Visualize one batch of images with bounding polygons instead of axis-aligned boxes.
    Uses region_mode-aware colors if available.
    """
    images, targets, names = batch
    loopMax = min(len(images), max_images)
    for i in range(loopMax):
        image = images[i].permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        target = targets[i]
        region_mode = target.get("region_mode", 0)
        
        # Define color palette based on region_mode
        palette_sizes = {0: 1, 1: 1, 2: 2, 3: 3, 4: 4}
        palette_len = palette_sizes.get(region_mode, 1)
        color_palette = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
        ]
        
        polys = target.get("polys", None)
        if polys is None:
            boxes = target["boxes"].cpu().numpy().astype(np.int32)
            for idx, box in enumerate(boxes):
                color = color_palette[idx % palette_len]
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        else:
            for idx, p in enumerate(polys):
                pts = p.reshape((-1, 1, 2)).astype(np.int32)
                color = color_palette[idx % palette_len]
                cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

        cv2.imshow(f"Augmented Image {i}", image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def _normalize_to_colormap(src: np.ndarray) -> np.ndarray:
    """
    Normalize a single-channel float or uint8 array to [0,255] uint8 and apply a JET colormap.
    """
    if src.dtype != np.uint8:
        min_v, max_v = float(src.min()), float(src.max())
        if max_v > min_v:
            norm = (src - min_v) / (max_v - min_v)
        else:
            norm = np.zeros_like(src, dtype=np.float32)
        src_u8 = (norm * 255.0).astype(np.uint8)
    else:
        src_u8 = src
    return cv2.applyColorMap(src_u8, cv2.COLORMAP_JET)


def visualizeOneBatchWithMaps(batch, max_images=5):
    """
    Visualize one batch of images together with:
      - Polygons overlaid on the grayscale image (like visualizeOneBatchImages).
      - Region map heatmap.
      - Affinity map heatmap.
      - Region and affinity heatmaps overlaid on the grayscale image.
    """
    images, targets, names = batch
    loopMax = min(len(images), max_images)

    for i in range(loopMax):
        image = images[i].permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        base_gray = image.squeeze()
        base_bgr = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)

        target = targets[i]
        polys = target.get("polys", [])
        region_mode = target.get("region_mode", 0)

        # Draw polygons on a copy of the base image with region_mode-aware colors
        img_with_polys = base_bgr.copy()
        
        # Define color palette based on region_mode
        # 1 color: Red, 2 colors: Red+Green, 3 colors: Red+Green+Blue, 4 colors: Red+Green+Blue+Yellow
        palette_sizes = {
            0: 1,  # Word mode: single color
            1: 1,  # Symbols no split: single color
            2: 2,  # 2 splits: two colors
            3: 3,  # 3 splits: three colors
            4: 4,  # 4 splits: four colors
        }
        palette_len = palette_sizes.get(region_mode, 1)
        
        # Color palette in BGR format: Red, Green, Blue, Yellow
        color_palette = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
        ]
        
        for idx, p in enumerate(polys):
            pts = np.asarray(p).reshape((-1, 1, 2)).astype(np.int32)
            color = color_palette[idx % palette_len]
            cv2.polylines(img_with_polys, [pts], isClosed=True, color=color, thickness=2)

        # Extract region and affinity maps (1, H, W) -> (H, W)
        region = target["region"].squeeze().detach().cpu().numpy()
        affinity = target["affinity"].squeeze().detach().cpu().numpy()

        # Ensure same spatial size as the padded image, resize if necessary
        h_img, w_img = base_gray.shape[:2]
        h_r, w_r = region.shape
        h_a, w_a = affinity.shape
        if (h_r, w_r) != (h_img, w_img):
            region_disp = cv2.resize(region, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        else:
            region_disp = region
        if (h_a, w_a) != (h_img, w_img):
            affinity_disp = cv2.resize(affinity, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        else:
            affinity_disp = affinity

        # Color maps
        region_color = _normalize_to_colormap(region_disp)
        affinity_color = _normalize_to_colormap(affinity_disp)

        # Overlays
        overlay_region = cv2.addWeighted(base_bgr, 0.5, region_color, 0.5, 0)
        overlay_affinity = cv2.addWeighted(base_bgr, 0.5, affinity_color, 0.5, 0)

        # Side-by-side panels: [image+polys | region heatmap | affinity heatmap]
        top_row = np.hstack([img_with_polys, region_color, affinity_color])

        # Overlays row: [overlay_region | overlay_affinity]
        bottom_row = np.hstack([overlay_region, overlay_affinity])

        # Stack vertically for a single canvas
        h_top, w_top, _ = top_row.shape
        h_bot, w_bot, _ = bottom_row.shape
        if w_bot != w_top:
            # Resize bottom row to match width
            bottom_row = cv2.resize(bottom_row, (w_top, h_bot), interpolation=cv2.INTER_LINEAR)
        canvas = np.vstack([top_row, bottom_row])

        win_name = f"Image {i} - {names[i]} with Region & Affinity"
        cv2.imshow(win_name, canvas)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def visualizeOneBatchWithMapsMatplotlib(batch, max_images=5):
    """
    Visualize one batch using Matplotlib in three panels:
      - Original grayscale image
      - Region map heatmap (with colorbar)
      - Affinity map heatmap (with colorbar)
    """
    images, targets, names = batch
    loopMax = min(len(images), max_images)

    for i in range(loopMax):
        image = images[i].permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        base_gray = image.squeeze()

        target = targets[i]
        region = target["region"].squeeze().detach().cpu().numpy()
        affinity = target["affinity"].squeeze().detach().cpu().numpy()
        region_mode = target.get("region_mode", 0)
        polys = target.get("polys", [])

        # Ensure region/affinity same spatial size as image for display
        h_img, w_img = base_gray.shape[:2]
        def _resize_if_needed(src: np.ndarray) -> np.ndarray:
            h_s, w_s = src.shape
            if (h_s, w_s) != (h_img, w_img):
                return cv2.resize(src, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
            return src

        region_disp = _resize_if_needed(region)
        affinity_disp = _resize_if_needed(affinity)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Original image with optional polygon overlay
        axes[0].imshow(base_gray, cmap="gray")
        
        # Overlay polygons with region_mode-aware colors if available
        if len(polys) > 0:
            palette_sizes = {0: 1, 1: 1, 2: 2, 3: 3, 4: 4}
            palette_len = palette_sizes.get(region_mode, 1)
            # Matplotlib uses RGB, so convert from BGR: Red, Green, Blue, Yellow
            color_palette_rgb = [
                (1.0, 0.0, 0.0),    # Red
                (0.0, 1.0, 0.0),    # Green
                (0.0, 0.0, 1.0),    # Blue
                (1.0, 1.0, 0.0),    # Yellow
            ]
            for idx, p in enumerate(polys):
                pts = np.asarray(p).reshape((-1, 2))
                color = color_palette_rgb[idx % palette_len]
                # Close the polygon by appending first point
                pts_closed = np.vstack([pts, pts[0:1]])
                axes[0].plot(pts_closed[:, 0], pts_closed[:, 1], color=color, linewidth=2, alpha=0.7)
        
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Region heatmap
        im1 = axes[1].imshow(region_disp, cmap="jet", vmin=0.0, vmax=1.0)
        axes[1].set_title("Region Map (heatmap)")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1])

        # Affinity heatmap
        vmax_aff = float(max(1e-6, affinity_disp.max()))
        im2 = axes[2].imshow(affinity_disp, cmap="jet", vmin=0.0, vmax=vmax_aff)
        axes[2].set_title("Affinity Map (heatmap)")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2])

        fig.suptitle(f"{names[i]}", y=1.02)
        plt.tight_layout()
        plt.show()
