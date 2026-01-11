import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def preProcess(image, resize=None):
    """
    Color --> GreyScale --> Binary --> Resize --> max = 1
    Set resize['height'] and resize['width'] to also resize the image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = np.mean(image, 2)
    _, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
    
    if resize:
        image = cv2.resize(image, (resize['height'], resize['width']))
    
    image /= np.max(image)
    return image


def getTransform():
    return A.Compose(
        [A.Sequential([
            A.CropAndPad(percent=0.1, pad_cval=1, keep_size=False, p=0.8),
            A.RandomCropFromBorders(
                crop_left=0.15,
                crop_right=0.15,
                crop_top=0.15,
                crop_bottom=0.15,
                p=0.8
            ),
            # A.RandomSizedBBoxSafeCrop(width=img_size[1], height=img_size[0],
            #                           erosion_rate=0.2, p=0.8),
            A.RandomRotate90(0.3),
            A.Blur(blur_limit=3, p=0.6),
            A.GaussNoise(var_limit=(0.001, 0.005), p=0.5),
            A.InvertImg(p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.2,
                p=0.5
            ),
            # A.transforms.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.2),
            ToTensorV2(p=1.0),
        ])],
        bbox_params={
            'format': 'pascal_voc',
            'label_fields': ['labels'],
            'min_visibility': 0.35
        }
    )


def getNoTransform():
    return A.Compose(
        [ToTensorV2(p=1.0)],
        bbox_params={
            'format': 'pascal_voc',
            'label_fields': ['labels'],
            'min_visibility': 0.35
        }
    )