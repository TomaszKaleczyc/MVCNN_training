import torch
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl

class MVCNN(pl.LightningModule):
    """
    
    """

    def __init__(self, 
                 learning_rate=1e-3,
                 feature_extractor=None,
                 num_epochs_freeze_pretrained=1,
                 dropout_rate=0.3
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self._setup_feature_extractor(feature_extractor)
        self._setup_image_vector_creator()
        self._setup_predictor()

    
    def _setup_feature_extractor(self, featire_extractor):
        """
        
        """
        pass

    def _setup_image_vector_creator(self):
        """
        
        """
        pass

    def _setup_predictor(self):
        """
        
        """
        pass