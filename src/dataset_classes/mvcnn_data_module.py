from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from dataset_classes.mvcnn_dataset import MVCNNDataset
from settings import consts


class MVCNNDataModule(LightningDataModule):
    """
    Manages the model datasets
    """

    def __init__(self, num_classes=None, batch_size=1):
        super().__init__()
        self._batch_size = batch_size
        self._train_dataset = MVCNNDataset(consts.TRAIN_FOLDER_NAME, num_classes)
        self._val_dataset = MVCNNDataset(consts.VALIDATION_FOLDER_NAME, num_classes)

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=consts.NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self._val_dataset, batch_size=self._batch_size, shuffle=False, num_workers=consts.NUM_WORKERS)

    def test_dataloader(self):
        raise NotImplementedError