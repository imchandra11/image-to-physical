import os
import lightning as L
from typing import Optional
from torch.utils.data import DataLoader
from EasyOCRTraining.dataset import OCRDataset, AlignCollate


class EasyOCRDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        imgH: int = 32,
        imgW: int = 100,
        PAD: bool = True,
        contrast_adjust: float = 0.0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = PAD
        self.contrast = contrast_adjust

        self.train_data_root = None
        self.valid_data_root = None
        self.test_data_root = None
        self.predict_data_root = None

    def _buildDataset(self, data_root: str) -> OCRDataset:
        return OCRDataset(data_root)

    def _buildDataLoader(self, dataset: OCRDataset, shuffle: bool = False, persistent_workers: bool = True) -> DataLoader:
        collate_fn = AlignCollate(
            imgH=self.imgH,
            imgW=self.imgW,
            keep_ratio_with_pad=self.keep_ratio
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=persistent_workers
        )

    def setup(self, stage: Optional[str] = None):
        self.train_data_root = os.path.join(self.data_dir, "Training")
        self.valid_data_root = os.path.join(self.data_dir, "Validation")
        self.test_data_root = os.path.join(self.data_dir, "Testing")
        self.predict_data_root = os.path.join(self.data_dir, "Testing")

        if stage in ("fit", None):
            self.train_dataset = self._buildDataset(self.train_data_root)
            self.valid_dataset = self._buildDataset(self.valid_data_root)

        elif stage == "validate":
            self.valid_dataset = self._buildDataset(self.valid_data_root)

        elif stage == "test":
            self.test_dataset = self._buildDataset(self.test_data_root)

        elif stage == "predict":
            self.predict_dataset = self._buildDataset(self.predict_data_root)

    def train_dataloader(self):
        return self._buildDataLoader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._buildDataLoader(self.valid_dataset)

    def test_dataloader(self):
        return self._buildDataLoader(self.test_dataset, persistent_workers=False)

    def predict_dataloader(self):
        return self._buildDataLoader(self.predict_dataset)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        images, labels = batch
        images = images.to(device)
        return images, labels
