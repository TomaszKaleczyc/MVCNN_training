import numpy as np
from matplotlib import pyplot as plt

import torch

from utilities import consts


class MVCNNObjectClassInstance:
    """
    Contains object class instance metadata and methods
    """

    def __init__(self, 
                 mvcnn_class, 
                 class_instance_id, 
                 class_instance_img_paths):
        self._class_id, self._class_name, self._num_classes = mvcnn_class.get_attributes()
        self._img_paths = class_instance_img_paths
        self._instance_id = class_instance_id

    def __len__(self):
        return len(self._img_paths)

    def get_images(self):
        """
        Returns images as a list of numpy arrays
        """
        image_list = [
            np.expand_dims(plt.imread(img_path), 0) for img_path in self._img_paths
            ]
        return np.concatenate(image_list, axis=0)

    def view_images(self, figsize=(30,10)):
        """
        Shows all instance images in a single row
        """
        print(self._class_name)
        print(self._instance_id)
        _, ax = plt.subplots(1, len(self), figsize=figsize)
        for idx, image in enumerate(self.get_images()):
            axis = ax[idx] if len(self) > 1 else ax
            axis.imshow(image)
        plt.show()

    def get_image_tensor(self):
        """
        Returns all instance images as torch tensor
        expected by the model
        """
        tensor = torch.tensor(self.get_images())
        tensor = torch.transpose(tensor, -1, 1)
        tensor = tensor.float()
        tensor /= 255
        mean_vec = torch.tensor(consts.NORMALIZATION_MEAN).view(1, 3, 1, -1)
        std_vec = torch.tensor(consts.NORMALIZATION_STD).view(1, 3, 1, -1)
        tensor = (tensor - mean_vec) / std_vec
        return tensor

    def get_target_tensor(self):
        """
        Returns tensor of expected target class
        """
        target = torch.zeros((1, self._num_classes))
        target[0, self._class_id] = 1
        return target

    def get_attributes(self):
        """
        Returns additional instance stats
        """
        output = {
            'class_name': self._class_name,
            'class_id': self._class_id,
            'instance_id': self._instance_id,
            'image_count': len(self._img_paths),
        }
        return output
