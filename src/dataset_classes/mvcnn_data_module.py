from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule


class MVCNNDataModule(LightningDataModule):
    """
    
    """

    def __init__(self):
        super().__init__()
        pass

    def train_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def val_dataloader(self):
        pass