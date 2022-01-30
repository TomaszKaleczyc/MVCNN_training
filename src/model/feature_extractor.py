from torch import nn, Tensor
from torchvision import models


class FeatureExtractor(nn.Module):
    """
    Manages the pretrained image feature extractor
    """

    def __init__(self):
        super().__init__()
        self._feature_extractor = models.resnet50(pretrained=True)
        self._requires_grad = True

    def forward(self, x: 'Tensor') -> Tensor:
        return self._feature_extractor(x)

    def switch_freeze(self):
        """
        Switches the feature extractor weight learning state
        """
        new_state = not self._requires_grad
        for param in self._feature_extractor.parameters():
            param.requires_grad = new_state
        self._requires_grad = new_state
        state_name = 'unfrozen' if self._requires_grad else 'frozen'
        print(f'Feature extractor weights {state_name}')

    def weights_frozen(self) -> bool:
        """
        Returns the parameter state of the feature extractor
        """
        return self._requires_grad
