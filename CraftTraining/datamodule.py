import os
from typing import Optional, Dict, Any
import torch
from torch.utils.data import DataLoader
import lightning as L
from CraftTraining.dataset import CraftDataset
from utils.craftImageaugmentation import get_transform


class CraftDataModule(L.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 1,
                 num_workers: int = 0,
                 resize: int = 512,
                 pin_memory: bool = True,
                 persistent_workers: bool = False,
                 gauss_cfg: Optional[Dict[str, Any]] = None,
                 data_cfg: Optional[Dict[str, Any]] = None,
                 test_cfg: Optional[Dict[str, Any]] = None):
        
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize = resize
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.gauss_cfg = gauss_cfg or {}
        self.data_cfg = data_cfg or {}
        self.test_cfg = test_cfg or {}

        self.train_transform = get_transform(self.data_cfg, is_train=True)
        self.no_transform = get_transform(self.data_cfg, is_train=False)

    def _buildDataset(self, split_dir: str, transforms, is_train: bool):
        return CraftDataset(
            data_dir=split_dir,
            resize=self.resize,
            transforms=transforms,
            gauss_cfg=self.gauss_cfg,
            is_train=is_train,
            data_cfg=self.data_cfg,
        )

    def _collate_fn(self, batch):
        images, targets, names = zip(*batch)
        heights = [img.shape[1] for img in images]
        widths = [img.shape[2] for img in images]
        max_h, max_w = max(heights), max(widths)

        padded_images, padded_targets = [], []
        for img, t in zip(images, targets):
            c, h, w = img.shape
            pad_img = torch.zeros((c, max_h, max_w), dtype=img.dtype)
            pad_img[:, :h, :w] = img
            padded_images.append(pad_img)

            region, affinity = t["region"], t["affinity"]
            pad_region = torch.zeros((1, max_h, max_w), dtype=region.dtype)
            pad_aff = torch.zeros((1, max_h, max_w), dtype=affinity.dtype)
            pad_region[:, :region.shape[1], :region.shape[2]] = region
            pad_aff[:, :affinity.shape[1], :affinity.shape[2]] = affinity

            t2 = dict(t)
            t2["region"] = pad_region
            t2["affinity"] = pad_aff
            t2["curr_size"] = (max_w, max_h)
            padded_targets.append(t2)

        return torch.stack(padded_images, dim=0), padded_targets, list(names)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            train_dir = os.path.join(self.data_dir, "Training")
            val_dir = os.path.join(self.data_dir, "Validation")
            self.dataset_train = self._buildDataset(train_dir, transforms=self.train_transform, is_train=True)
            self.dataset_val = self._buildDataset(val_dir, transforms=self.no_transform, is_train=False)

        elif stage == "validate":
            val_dir = os.path.join(self.data_dir, "Validation")
            self.dataset_val = self._buildDataset(val_dir, transforms=self.no_transform, is_train=False)

        elif stage == "test":
            test_dir = os.path.join(self.data_dir, "Testing")
            self.dataset_test = self._buildDataset(test_dir, transforms=self.no_transform, is_train=False)

        elif stage == "predict":
            pred_dir = os.path.join(self.data_dir, "Testing")
            self.dataset_predict = self._buildDataset(pred_dir, transforms=self.no_transform, is_train=False)

    def _buildDataLoader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn, 
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

    def train_dataloader(self):
        return self._buildDataLoader(self.dataset_train, shuffle=True)

    def val_dataloader(self):
        return self._buildDataLoader(self.dataset_val)

    def test_dataloader(self):
        return self._buildDataLoader(self.dataset_test)

    def predict_dataloader(self):
        return self._buildDataLoader(self.dataset_predict)

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        images, targets, names = batch
        if isinstance(images, torch.Tensor):
            images = images.to(device, non_blocking=True)
        else:
            images = torch.stack([img.to(device, non_blocking=True) for img in images], dim=0)
        new_targets = []
        for t in targets:
            moved = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in t.items()
            }
            new_targets.append(moved)
        return images, new_targets, names
