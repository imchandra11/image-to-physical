# dataset.py
import os
import math
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
import pandas as pd

class OCRDataset(Dataset):
    """
    A Dataset that reads image files and labels from a CSV file.
    Expects a CSV with columns 'filename' and 'words'. Images are loaded
    from a subdirectory under root_dir (e.g. 'images/').
    """
    def __init__(self, root_dir, csv_file="labels.csv", img_dir="images", grayscale=True):
        """
        Args:
            root_dir (str): Path to the dataset directory.
            csv_file (str): Name of the CSV file (relative to root_dir).
            img_dir (str): Subdirectory under root_dir where images are stored.
            grayscale (bool): If True, convert images to grayscale ('L'); else RGB.
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, img_dir)
        csv_path = os.path.join(root_dir, csv_file)
        df = pd.read_csv(csv_path)
        # Check for required columns
        if 'filename' not in df.columns or 'words' not in df.columns:
            raise ValueError("CSV file must contain 'filename' and 'words' columns.")
        # Load filenames and labels
        self.filenames = df['filename'].tolist()
        self.labels = df['words'].astype(str).tolist()
        self.grayscale = grayscale

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Load image
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        # Convert color mode
        if self.grayscale:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        # Load label
        label = self.labels[idx]
        return image, label

class AlignCollate(object):
    """
    Collate function that resizes images to (imgW, imgH). If keep_ratio_with_pad
    is True, maintains aspect ratio and pads the rest of width with black pixels.
    Returns a batch of image tensors and a list of labels.
    """
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        """
        Args:
            imgH (int): Desired fixed image height.
            imgW (int): Desired fixed image width.
            keep_ratio_with_pad (bool): If True, keep aspect ratio and pad width;
                                        otherwise, resize directly to (imgW, imgH).
        """
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio_with_pad
        self.toTensor = ToTensor()

    def __call__(self, batch):
        # batch is a list of (PIL.Image, label) tuples
        images, labels = zip(*batch)
        processed_images = []
        for img in images:
            if self.keep_ratio:
                # Compute proportional width
                w, h = img.size
                ratio = w / float(h)
                new_w = math.floor(self.imgH * ratio)
                if new_w > self.imgW:
                    new_w = self.imgW
                # Resize with aspect ratio
                img_resized = img.resize((new_w, self.imgH), Image.BICUBIC)
                # Create a new blank image and paste the resized image
                if img.mode == 'L':
                    new_img = Image.new('L', (self.imgW, self.imgH), color=0)  # black pad
                else:
                    new_img = Image.new('RGB', (self.imgW, self.imgH), color=(0,0,0))
                new_img.paste(img_resized, (0, 0))
            else:
                # Direct resize (may distort)
                img_resized = img.resize((self.imgW, self.imgH), Image.BICUBIC)
                new_img = img_resized
            # Convert to tensor
            img_tensor = self.toTensor(new_img)
            processed_images.append(img_tensor)
        # Stack into (batch, C, H, W)
        image_batch = torch.stack(processed_images, dim=0)
        return image_batch, list(labels)
