import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class UnfreezePretrainedWeights(Callback):
    """
    Unfreezing the feature extractor weights 
    and reducing the learning rate by given factor
    """
    def __init__(self, learning_rate_reduction_factor):
        super().__init__()
        self._learning_rate_reduction_factor = learning_rate_reduction_factor

    def on_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
        switch_trigger1 = not pl_module.feature_extractor.weights_frozen() and pl_module.current_epoch == pl_module.num_epochs_freeze_pretrained
        switch_trigger2 = pl_module.feature_extractor.weights_frozen() and pl_module.current_epoch < pl_module.num_epochs_freeze_pretrained
        if switch_trigger1 or switch_trigger2:
            pl_module.feature_extractor.switch_freeze()
            curr_lr = pl_module.learning_rate
            new_lr = curr_lr / self._learning_rate_reduction_factor
            pl_module.learning_rate = new_lr
            pl_module.configure_optimizers()
            print(f'Learning rate reduced from {curr_lr} to {new_lr}')
