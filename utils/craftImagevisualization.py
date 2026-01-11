import os
import cv2
import numpy as np
import torch

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
    """
    images, targets, names = batch
    loopMax = min(len(images), max_images)
    for i in range(loopMax):
        image = images[i].permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        target = targets[i]
        # ✅ Use 'raw_boxes' if you want axis-aligned boxes OR 'raw_polys' if available
        polys = target.get("polys", None)
        if polys is None:
            boxes = target["boxes"].cpu().numpy().astype(np.int32)
            for box in boxes:
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        else:
            for p in polys:
                pts = p.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=1)

        cv2.imshow(f"Augmented Image {i}", image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
