import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from utilities import consts


class UnfreezePretrainedWeights(Callback):
    """
    Unfreezing the feature extractor weights and reducing the learning rate
    """

    def on_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
        switch_trigger1 = not pl_module.feature_extractor.weights_frozen() and pl_module.current_epoch == pl_module.num_epochs_freeze_pretrained
        switch_trigger2 = pl_module.feature_extractor.weights_frozen() and pl_module.current_epoch < pl_module.num_epochs_freeze_pretrained
        if switch_trigger1 or switch_trigger2:
            pl_module.feature_extractor.switch_freeze()
            curr_lr = pl_module.learning_rate
            new_lr = curr_lr / consts.LEARNING_RATE_REDUCTION_FACTOR
            pl_module.learning_rate = new_lr
            pl_module.configure_optimizers()
            print(f'Learning rate reduced from {curr_lr} to {new_lr}')

class ResetEvalResults(Callback):
    """
    Resets the evaluation objects
    """
    def __init__(self, num_classes):
        super().__init__()
        self._num_classes = num_classes

    def on_validation_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
        self._reset_results(pl_module)
    
    def on_test_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
        self._reset_results(pl_module)
    
    def _reset_results(self, pl_module):
        pl_module.avg_metric = torchmetrics.AverageMeter().to(pl_module.device)
        pl_module.eval_metric = torchmetrics.F1(
            threshold=consts.CLASSIFICATION_THRESHOLD,
            num_classes=self._num_classes, 
            average=consts.F1_AVERAGE
            ).to(pl_module.device)
        pl_module.secondary_eval_metric = torchmetrics.Accuracy(
            threshold=consts.CLASSIFICATION_THRESHOLD,
            num_classes=self._num_classes, 
        ).to(pl_module.device)
