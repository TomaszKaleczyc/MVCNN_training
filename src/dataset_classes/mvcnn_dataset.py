import numpy as np
from pathlib import Path

from torch.utils.data import Dataset

from utilities import consts

class MVCNNDataset(Dataset):
    """
    
    """

    def __init__(self, dataset_type_name):
        self._dataset_type_name = dataset_type_name
        self._instances_list = self._get_instances_list()

    def _get_instances_list(self):
        """
        
        """
        class_paths = Path(consts.DATA_DIR).iterdir()
        for class_path in class_paths:
            self._get_class_instances(class_path)

    def _get_class_instances(self, class_path):
        """
        
        """
        class_dataset_path = class_path/self._dataset_type_name
        class_dataset_img_paths = self._get_dataset_img_paths(class_dataset_path)
        class_dataset_instances = self._get_class_dataset_instances(class_dataset_img_paths)
        for class_dataset_instance in class_dataset_instances:
            self._append_class_instance(class_path.name, class_dataset_img_paths)

    def _get_dataset_img_paths(self, class_dataset_path):
        """
        
        """
        class_image_paths = [
            path for path in class_dataset_path.iterdir() if path.suffix.lower() in consts.IMG_SUFFIX_LIST
            ]
        return class_image_paths

    def _get_class_dataset_instances(self, class_dataset_img_paths):
        """
        
        """
        class_dataset_instances = np.unique(
            [path.name.split('_')[-2] for path in class_dataset_img_paths]
        )
        return class_dataset_instances

    def _append_class_instance(self, class_dataset_instance, class_dataset_img_paths):
        """
        
        """
        class_instance_img_paths = self._get_class_instance_img_paths(class_dataset_instance, class_dataset_img_paths)
        self._instances_list.append(ClassInstance(class_dataset_instance, class_instance_img_paths))

    def _get_class_instance_img_paths(self, class_dataset_instance, class_dataset_img_paths):
        """
        
        """
        class_instance_img_paths = [
            img_path for img_path in class_dataset_img_paths if img_path.name.split('_')[-2] == class_dataset_instance
        ]
        return class_instance_img_paths


    def __len__(self):
        return len(self._instances_list)

    def __getitem__(self, idx):
        pass


class ClassInstance:

    def __init__(self, class_name, class_instance_img_paths):
        self._class_name = class_name
        self._class_instance_img_paths = class_instance_img_paths