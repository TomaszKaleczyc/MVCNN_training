from typing import Optional, Tuple
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F

import torchmetrics

import pytorch_lightning as pl

from model.feature_extractor import FeatureExtractor
from settings import data_settings, model_settings
from utilities import utils


class MVCNNClassifier(pl.LightningModule):
    """
    Main MVCNN model class
    """
    
    def __init__(self,
                 num_classes: int,
                 learning_rate: float = 1e-3, 
                 feature_extractor: Optional[nn.Module]=None,
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
        self.avg_metric = torchmetrics.AverageMeter().to(self.device)
        self.eval_metric = torchmetrics.F1(
            threshold=model_settings.CLASSIFICATION_THRESHOLD,
            num_classes=self._num_classes, 
            average=model_settings.F1_AVERAGE
            ).to(self.device)
        self.secondary_eval_metric = torchmetrics.Accuracy(
            threshold=model_settings.CLASSIFICATION_THRESHOLD,
            num_classes=self._num_classes, 
        ).to(self.device)

        
    def _setup_feature_extractor(self, feature_extractor: nn.Module):
        """
        Defines the model feature extractor
        """
        if feature_extractor is None:
            self.feature_extractor = FeatureExtractor()
        else:
            self.feature_extractor =  feature_extractor
        if self.feature_extractor.weights_frozen():
            self.feature_extractor.switch_freeze()
    
    def _setup_image_vector_creator(
            self, 
            feature_extractor_out_dim: int = 1000,
            image_vector_creator_out_dim: int = 512
            ):
        """
        Defines the layers creating per image vectors
        """
        self.image_vector_creator = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(feature_extractor_out_dim, image_vector_creator_out_dim),
            nn.ReLU(),
        )
        
    def _setup_predictor(self, global_vector_input_dim: int = 512):
        """
        Defines the layers responsible for the final prediction
        """
        self.predictor = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(global_vector_input_dim, 144),
            nn.ReLU(),
            nn.Linear(144, self._num_classes),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x = x[0]  # TODO: batch processing
        
        image_features = self.feature_extractor(x)
        image_vectors = self.image_vector_creator(image_features)
        
        # combining information from individual image vectors using a symmetrical function:
        global_vector = image_vectors.mean(axis=0)
        
        output_vector = self.predictor(global_vector)
        return output_vector.view(1, -1)
    
    def _loss_step(
            self, 
            batch: Tuple[Tensor, Tensor], 
            batch_idx: int, 
            dataset_name: str, 
            criterion=F.cross_entropy
        ) -> Tensor:
        """
        Base training/validation loop step
        """
        x, y, _ = batch
        y = y[0]
        target_class = y.argmax(dim=1)
        y_hat = self(x)
        loss = criterion(y_hat, target_class)
        y_hat_smax = torch.softmax(y_hat, dim=1)
        self.avg_metric.update(loss)
        self.eval_metric.update(y_hat_smax, target_class)           
        self.secondary_eval_metric.update(y_hat_smax, y.int())
        self.log(f'{dataset_name}/loss', loss)
        return loss
  
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._loss_step(batch, batch_idx, dataset_name='train')

    def training_epoch_end(self, outputs):
        self._epoch_end('train')
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._loss_step(batch, batch_idx, dataset_name='validation')

    def validation_epoch_end(self, outputs):
        self._epoch_end('validation')

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._loss_step(batch, batch_idx, dataset_name='test')

    def test_epoch_end(self, outputs):
        self._epoch_end('validation')

    def _epoch_end(self, dataset_name: str):
        """
        Standard epoch end step
        """
        print('='*60)
        print(f'Eval stats for {dataset_name.upper()}')
        self._print_stats()
        avg_loss = self.avg_metric.compute()
        eval_metric_score = self.eval_metric.compute()
        secondary_eval_metric_score = self.secondary_eval_metric.compute()
        self.log(f'{dataset_name}/loss', avg_loss)
        print(f'Avg Loss na {dataset_name}', avg_loss)

        self.log(f'{dataset_name}/f1', eval_metric_score)
        print(f'F1 - {dataset_name}', eval_metric_score)

        self.log(f'{dataset_name}/accuracy', secondary_eval_metric_score)
        print(f'Accuracy - {dataset_name}', secondary_eval_metric_score)

        for metric in [self.avg_metric, self.eval_metric, self.secondary_eval_metric]:
            metric.reset()

    def _print_stats(self):
        """
        Displays the evaluation results per class
        """
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
        parameter_groups = [
            {'params': self.feature_extractor.parameters(), 'weight_decay': model_settings.FEATURE_EXTRACTOR_WEIGHT_DECAY},
            {'params': self.image_vector_creator.parameters(), 'weight_decay': model_settings.IMAGE_VECTOR_CREATOR_WEIGHT_DECAY},
            {'params': self.predictor.parameters(), 'weight_decay': model_settings.PREDICTOR_WEIGHT_DECAY}
        ]
        return torch.optim.Adam(parameter_groups, lr=self.learning_rate)

    def predict(self, image_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns the predicted class and class probabilities
        for a given image tensor
        """
        if len(image_tensor.shape) < 5:
            image_tensor = image_tensor.unsqueeze(0)
        output = self.train(False)(image_tensor)
        probabilities = F.softmax(output)
        predicted_class = probabilities.argmax(dim=1)
        return predicted_class, probabilities
