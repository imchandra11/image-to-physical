import os
from typing import NamedTuple, Any
from torch.utils.data import Dataset
import cv2
import numpy as np
from utils.imageaugmentation import preProcess

# ToDo: Move to config?
IMAGE_INPUT_FOLDER="Input"
IMAGE_OUTPUT_FOLDER="Output"

# ToDo: Support other file formats/types? Readers?
IMAGE_EXTN = ("PNG", "JPG", "JPEG", "TIF")
IMAGE_OUTPUT_EXTN=".csv"
CSV_HEADERS = ["MainLabel","SubLabel","MinX", "MinY", "MaxX", "MaxY", "CenterX","CenterY","Width","Height","Value"]

# class DatasetODParameters(NamedTuple):
#     data_dir: str
#     resize: int
#     classes: dict
#     transforms: Any

class DatasetOD(Dataset):
    def __init__(self,
        data_dir: str,
        resize: int,
        classes: dict,
        transforms: Any
    ):
        r"""Dataset for Object Detection.

        Should be specialised and used.
        The inheriting class must implement __getitem__().

        """
        self.data_dir = data_dir
        self.resize = resize
        self.resize_width = self.resize
        self.resize_height = self.resize
        self.classes = classes
        self.transforms = transforms

        self.input_path = os.path.join(self.data_dir, IMAGE_INPUT_FOLDER)
        if not (os.path.exists(self.input_path)):
            self.input_path = self.data_dir
        print(f'Loading all {IMAGE_EXTN} from {self.input_path}')
        self.all_images = [fi for fi in os.listdir(self.input_path) if(fi.upper().endswith(IMAGE_EXTN))]
        if(len(self.all_images) == 0):
            print(f'WARNING: No images found at location')
        else:
            print(f'Found {len(self.all_images)} images')
        self.all_images = sorted(self.all_images)

        self.output_path = os.path.join(self.data_dir, IMAGE_OUTPUT_FOLDER)
        if not (os.path.exists(self.output_path)):
            self.output_path = self.data_dir

    def __len__(self):
        return len(self.all_images)

    def _getImageDetails(self, idx):
        image_name = self.all_images[idx]
        image_path = os.path.join(self.input_path, image_name)
        image_output_path = os.path.join(self.output_path, os.path.splitext(image_name)[0] + IMAGE_OUTPUT_EXTN)

        return image_name, image_path, image_output_path

import torchvision.transforms as transforms

class DatasetImage(DatasetOD):
    def __init__(self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def __getitem__(self, idx):
        image_name, image_path, image_output_path = self._getImageDetails(idx)
        image = cv2.imread(image_path)
        image_width = image.shape[1]
        image_height = image.shape[0]
        image = preProcess(image, {'height':self.resize_height, 'width':self.resize_width})
        image = image[..., np.newaxis]
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image = transform(image)

        return [(image, {}, image_name)]
