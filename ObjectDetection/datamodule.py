import os
import lightning as L
import utils.lib
from typing import Optional
from ObjectDetection.dataset import DatasetOD, DatasetImage
from torch.utils.data import DataLoader
from utils.dataset import collate_fn
from utils.imageaugmentation import getTransform, getNoTransform

class DataModuleOD(L.LightningDataModule):

    def __init__(self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        resize: int,
        classes: dict,
        dataset: Optional[dict] = None,
        dataset_factory: Optional[dict] = None,
    ) -> None:
        r"""Data Module for Object Detection.

        Args:
            data_dir: The data directory
            batch_size: The batch size
            num_workers: The number of workers to use
            resize: The resize dimension for the image
            classes: The classes to be recognised
            dataset: The dataset to use
            dataset_factory: Function to run to get the dataset to use
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize = resize
        self.classes = classes
        self.dataset_class, self.dataset_args = utils.lib.getCallableAndArgs(dataset, dataset_factory)

    def _buildDataset(self,
        data_dir: str, transforms
    ) -> DatasetOD:
        return self.dataset_class(
            data_dir=data_dir,
            resize=self.resize,
            classes=self.classes,
            transforms=transforms,
            **self.dataset_args if self.dataset_args else {}
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.dataset_train =  self._buildDataset(os.path.join(self.data_dir, "Training"), getTransform())

        if stage in ("fit", "validate"):
            self.dataset_validate = self._buildDataset(os.path.join(self.data_dir, "Validation"), getTransform())

        if stage == "test":
            self.dataset_test = self._buildDataset(os.path.join(self.data_dir, "Testing"), getNoTransform())

        if stage == "predict":
            self.dataset_predict = DatasetImage(data_dir = self.data_dir,
            resize = self.resize,
            classes = self.classes,
            transforms = getNoTransform())

    def _buildDataLoader(self, dataset: DatasetOD, 
        persistent_workers: bool = True, shuffle: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn)

    def train_dataloader(self) -> DataLoader:
        return self._buildDataLoader(self.dataset_train, shuffle = True)

    def val_dataloader(self) -> DataLoader:
        return self._buildDataLoader(self.dataset_validate)

    def test_dataloader(self) -> DataLoader:
        return self._buildDataLoader(self.dataset_test, persistent_workers = False)
    
    def predict_dataloader(self) -> DataLoader:
        return self._buildDataLoader(self.dataset_predict)
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        images, targets, names = batch
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        return (images, targets, names)
    
