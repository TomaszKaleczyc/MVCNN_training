from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path, PosixPath

from torch import Tensor
from torch.utils.data import Dataset

from dataset.mvcnn_object_class import MVCNNObjectClass
from dataset.mvcnn_object_class_instance import MVCNNObjectClassInstance
from settings import data_settings
from utilities import utils


class MVCNNDataset(Dataset):
    """
    Manages all dataset instances
    """

    def __init__(
            self, 
            dataset_type_name: str, 
            num_classes: Optional[int] = None, 
            verbose: bool = True
            ):   
        self._type_name: str = dataset_type_name
        self._num_classes: int = utils.get_num_classes(num_classes)
        self._classes_list: List[MVCNNObjectClass] = self._get_classes_list()
        self._instances_list: List[MVCNNObjectClassInstance] = []
        self._identify_instances()
        if verbose:
            self._print_dataset_summary()

    def _get_classes_list(self) -> List['MVCNNObjectClass']:
        """
        Creates object class list to store class metadata
        """
        iterator = list(enumerate(Path(data_settings.DATA_DIR).iterdir()))
        classes_list = [
            MVCNNObjectClass(class_num, class_path, self._num_classes) for class_num, class_path in iterator if class_num < self._num_classes
            ]
        return classes_list

    def _identify_instances(self):
        """
        Identifies all dataset instances
        """
        for mvcnn_class in self._classes_list:
            self._get_class_instances(mvcnn_class)

    def _get_class_instances(self, mvcnn_class: 'MVCNNObjectClass'):
        """
        Identifies all instances of given class
        """
        class_dataset_path = mvcnn_class.get_path()/self._type_name
        class_dataset_img_paths = self._get_dataset_img_paths(class_dataset_path)
        class_dataset_instances = self._get_class_dataset_instances(class_dataset_img_paths)
        for class_dataset_instance_id in class_dataset_instances:
            class_instance_img_paths = self._get_class_instance_img_paths(class_dataset_instance_id, class_dataset_img_paths)
            class_instance = MVCNNObjectClassInstance(
                mvcnn_class, class_dataset_instance_id, class_instance_img_paths
                )
            self._instances_list.append(class_instance)
            mvcnn_class.update_summary(class_instance)

    def _get_dataset_img_paths(self, class_dataset_path: PosixPath) -> List[PosixPath]:
        """
        Returns image paths of a given dataset path
        """
        class_image_paths = [
            path for path in class_dataset_path.iterdir() if path.suffix.lower() in data_settings.IMG_SUFFIX_LIST
            ]
        return class_image_paths

    def _get_class_dataset_instances(self, class_dataset_img_paths: List[PosixPath]) -> List['MVCNNObjectClassInstance']:
        """
        Returns instance id's of all given image paths
        """
        class_dataset_instances = np.unique(
            [path.name.split('_')[-2] for path in class_dataset_img_paths]
        )
        return class_dataset_instances

    def _get_class_instance_img_paths(
            self, 
            class_dataset_instance: MVCNNObjectClassInstance, 
            class_dataset_img_paths: List[PosixPath]
            ) -> List[PosixPath]:
        """
        Returns all image paths pertaining to a given class instance
        """
        class_instance_img_paths = [
            img_path for img_path in class_dataset_img_paths if img_path.name.split('_')[-2] == class_dataset_instance
        ]
        return class_instance_img_paths

    def _print_dataset_summary(self):
        """
        Displays the summary of the dataset
        """
        print('='*60)
        print('Dataset type:', self._type_name.upper())
        for class_object in self._classes_list:
            class_object.print_summary()

    def get_summary_df(self):
        """
        Returns DataFrame with summary
        """
        summary_dict = {
            'class_id': [],
            'class_name': [],
            'num_instances': [],
            'num_images': [],
        }
        for class_object in self._classes_list:
            for key, value in class_object.get_summary().items():
                summary_dict[key].append(value)
        return pd.DataFrame(summary_dict)

    def __len__(self):
        return len(self._instances_list)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, dict]:
        object_instance = self.get_instance(idx)
        image_tensor = object_instance.get_image_tensor()
        target_tensor = object_instance.get_target_tensor()
        instance_attributes = object_instance.get_attributes()
        return image_tensor, target_tensor, instance_attributes

    def view_random_instances(self, class_id: int=None, num_instances: int=1):
        """
        Displays random class instances
        """
        if class_id is None:
            class_id = self._get_random_class_id()
        class_instances = [instance for instance in self._instances_list if instance.belongs_to_class(class_id)]
        random_instances = np.random.choice(class_instances, size=num_instances, replace=False)
        for instance in random_instances:
            instance.view_images()

    def _get_random_class_id(self) -> int:
        """
        Returns random class id
        """
        random_class = np.random.choice(self._classes_list, 1)[0]
        class_id, *_ = random_class.get_attributes()
        return class_id

    def get_instance(self, idx: int) -> 'MVCNNObjectClassInstance':
        """
        Returns the object instance
        """
        return self._instances_list[idx]

    def get_random_instance(self) -> 'MVCNNObjectClassInstance':
        """
        Returns a random instance
        """
        idx = np.random.randint(0, len(self))
        return self.get_instance(idx)

    def __repr__(self):
        return f'Dataset: {self._type_name}: {self._num_classes} classes, {len(self)} instances'
