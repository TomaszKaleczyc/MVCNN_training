import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl

from model_classes.feature_extractor import FeatureExtractor
from settings import utils


class MVCNNClassifier(pl.LightningModule):
    """
    Main MVCNN model class
    """
    
    def __init__(self,
                 num_classes,
                 learning_rate=1e-3, 
                 feature_extractor=None,
                 num_epochs_freeze_pretrained=1, 
                 dropout_rate=0.5):
        super().__init__()
        self.save_hyperparameters()
        self._num_classes = utils.get_num_classes(num_classes)
        self.dropout_rate = dropout_rate
        self.num_epochs_freeze_pretrained = num_epochs_freeze_pretrained
        self.learning_rate = learning_rate
        
        # architecture:
        self._setup_feature_extractor(feature_extractor)
        self._setup_image_vector_creator()
        self._setup_predictor()

        # evaluation:
        self.avg_metric = None
        self.eval_metric = None
        self.secondary_eval_metric = None
        
    def _setup_feature_extractor(self, feature_extractor):
        """
        Defines the model feature extractor
        """
        if feature_extractor is None:
            self.feature_extractor = FeatureExtractor()
        else:
            self.feature_extractor =  feature_extractor
        if self.feature_extractor.weights_frozen():
            self.feature_extractor.switch_freeze()
    
    def _setup_image_vector_creator(self, feature_extractor_out_dim=1000, 
                                    image_vector_creator_out_dim=512):
        """
        Defines the layers creating per image vectors
        """
        self.image_vector_creator = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(feature_extractor_out_dim, image_vector_creator_out_dim),
            nn.ReLU(),
        )
        
    def _setup_predictor(self, global_vector_input_dim=512):
        """
        Defines the layers responsible for the final prediction
        """
        self.predictor = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(global_vector_input_dim, 144),
            nn.ReLU(),
            nn.Linear(144, self._num_classes),
        )
        
    def forward(self, x):
        x = x[0]  # TODO: batch processing
        
        image_features = self.feature_extractor(x)
        image_vectors = self.image_vector_creator(image_features)
        
        # combining information from individual image vectors using a symmetrical function:
        global_vector = image_vectors.mean(axis=0)
        
        output_vector = self.predictor(global_vector)
        return output_vector.view(1, -1)
    
    def _loss_step(self, batch, batch_idx, eval=False, criterion=F.cross_entropy):
        """
        Base training/validation loop step
        """
        x, y, _ = batch
        y = y[0]
        target_class = y.argmax(dim=1)
        y_hat = self(x)
        loss = criterion(y_hat, target_class)
        y_hat_smax = torch.softmax(y_hat, dim=1)
        if self.avg_metric is not None:
            self.avg_metric.update(loss)
        if self.eval_metric is not None:
            self.eval_metric.update(y_hat_smax, target_class)           
        if self.secondary_eval_metric is not None:
            self.secondary_eval_metric.update(y_hat_smax, y.int())
        return loss
  
    def training_step(self, batch, batch_idx):
        """
        Training loop step
        """
        loss = self._loss_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation loop step
        """
        loss = self._loss_step(batch, batch_idx, eval=True)
        self.log('val_loss', loss)

    def validation_epoch_end(self, outputs):
        """
        End of validation step
        """
        avg_loss, eval_metric_score, secondary_eval_metric_score = self._eval_epoch_end()

        self.log('avg_val_loss', avg_loss)
        print('Avg Loss val', avg_loss)
        
        self.log('val_f1', eval_metric_score)
        print('F1 val', eval_metric_score)
        
        self.log('val_accuracy', secondary_eval_metric_score)
        print('Accuracy VAL', secondary_eval_metric_score)

    def test_step(self, batch, batch_idx):
        """
        Test loop step
        """
        loss = self._loss_step(batch, batch_idx, eval=True)
        self.log('test_loss', loss)

    def test_epoch_end(self, outputs):
        """
        End of test step
        """
        avg_loss, eval_metric_score, secondary_eval_metric_score = self._eval_epoch_end()

        self.log('avg_test_loss', avg_loss)
        print('Avg Loss na test', avg_loss)

        self.log('test_f1', eval_metric_score)
        print('F1 na test', eval_metric_score)

        self.log('test_accuracy', secondary_eval_metric_score)
        print('Accuracy na TEST', secondary_eval_metric_score)

    def _eval_epoch_end(self):
        """
        Standard evaluation epoch end step
        """
        self._print_stats()
        avg_loss = self.avg_metric.compute()
        eval_metric_score = self.eval_metric.compute()
        secondary_eval_metric_score = self.secondary_eval_metric.compute()
        return avg_loss, eval_metric_score, secondary_eval_metric_score

    def _print_stats(self):
        """
        Displays the evaluation results per class
        """
        print('\nEval stats:')
        tp = self.eval_metric.tp.cpu().numpy()
        tn = self.eval_metric.tn.cpu().numpy()
        fp = self.eval_metric.fp.cpu().numpy()
        fn = self.eval_metric.fn.cpu().numpy()
        epsilon = np.finfo(float).eps
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)

        metrics = ['tp', 'tn', 'fp', 'fn', 'accuracy', 
                   'precision', 'recall', 'f1']
        for metric in metrics:
            print(f'{metric.upper()} per class: {eval(metric)} Average: {eval(metric).mean():.2f}')

    def configure_optimizers(self):
        """
        Configuring the net optimization methods
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)