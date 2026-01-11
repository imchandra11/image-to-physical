import torch
import cv2
import numpy as np
import pandas as pd

from ObjectDetection.dataset import DatasetOD
from ObjectDetection.dataset import CSV_HEADERS

from utils.imageaugmentation import preProcess
from utils.image import convertToBoundingBox

# All SubLabels
MAIN_LABEL_FILTER = ["View"]
CSV_LABEL_TO_CLASS_MAP = {"MainView":"Main", "FrameTitleBlock":"Title", "BOMTable":"BOM"}


class DatasetVR(DatasetOD):
    def __init__(self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def __getitem__(self, idx):

        image_name, image_path, image_output_path = self._getImageDetails(idx)

        image = cv2.imread(image_path)
        image_width = image.shape[1]
        image_height = image.shape[0]
        image_resized = preProcess(image, {'height':self.resize_height, 'width':self.resize_width})
        image_resized = image_resized[..., np.newaxis]

        df = pd.read_csv(image_output_path, encoding='cp1252')[CSV_HEADERS]
        df = df.loc[(df["MainLabel"].isin(MAIN_LABEL_FILTER))] # & (df['Value'].isin(CLASSES.keys()))]
        df["Value"] = df["SubLabel"].map(CSV_LABEL_TO_CLASS_MAP)

        df = df.apply(lambda x: pd.Series(
            convertToBoundingBox(x.CenterX, x.CenterY, x.Width, x.Height, x.Value, image_width, image_height),
            index=['x1', 'y1', 'x2', 'y2', 'Value']), axis=1)

        df["x1"] = df["x1"].apply(lambda x: (x/image_width)*self.resize_width)
        df["x2"] = df["x2"].apply(lambda x: (x/image_width)*self.resize_width)
        df["y1"] = df["y1"].apply(lambda x: (x/image_height)*self.resize_height)
        df["y2"] = df["y2"].apply(lambda x: (x/image_height)*self.resize_height)

        boxes = df[["x1", "y1", "x2", "y2"]].to_numpy()
        labels = df["Value"].to_numpy()

        if len(labels) == 0:
            print(f'No output data for {image_name}')
            return None, None, image_name
        
        labels_classes = []
        for label in labels :
            labels_classes.append(self.classes[label])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels_classes = torch.as_tensor(labels_classes, dtype=torch.int64)

        # Prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels_classes
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        # Apply the image transforms
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = target["labels"])
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            target['labels'] = torch.as_tensor(sample['labels'], dtype=torch.int64)

        return [(image_resized, target, image_name)]

