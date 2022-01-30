from typing import Optional

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from dataset.mvcnn_dataset import MVCNNDataset
from settings import data_settings


class MVCNNDataModule(LightningDataModule):
    """
    Manages the model datasets
    """

    def __init__(self, num_classes: Optional[int] = None, batch_size: int=1):
        super().__init__()
        self._batch_size = batch_size
        self._train_dataset = MVCNNDataset(data_settings.TRAIN_FOLDER_NAME, num_classes)
        self._val_dataset = MVCNNDataset(data_settings.VALIDATION_FOLDER_NAME, num_classes)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=data_settings.NUM_WORKERS)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val_dataset, batch_size=self._batch_size, shuffle=False, num_workers=data_settings.NUM_WORKERS)

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def get_dataset(self, dataset_name: str) -> MVCNNDataset:
        """
        Returns selected dataset
        """
        if dataset_name == 'train':
            return self._train_dataset
        if dataset_name == 'val':
            return self._val_dataset
        raise NotImplementedError
