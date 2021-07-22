import numpy as np
from pathlib import Path

from torch.utils.data import Dataset

from dataset_classes.mvcnn_object_class import MVCNNObjectClass
from dataset_classes.mvcnn_object_class_instance import MVCNNObjectClassInstance
from utilities import consts


class MVCNNDataset(Dataset):
    """
    Manages all dataset instances
    """

    def __init__(self, dataset_type_name, num_classes):
        self._num_classes = num_classes
        self._type_name = dataset_type_name
        self._classes_list = self._get_classes_dict()
        self._instances_list = []
        self._identify_instances()
        self._print_dataset_summary()

    def _get_classes_dict(self):
        """
        Creates object class list to store class metadata
        """
        iterator = enumerate(Path(consts.DATA_DIR).iterdir())
        classes_dict = [
            MVCNNObjectClass(class_num, class_path, self._num_classes) for class_num, class_path in iterator if class_num < self._num_classes
            ]
        return classes_dict
        

    def _identify_instances(self):
        """
        Identifies all dataset instances
        """
        for mvcnn_class in self._classes_list:
            self._get_class_instances(mvcnn_class)

    def _get_class_instances(self, mvcnn_class):
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

    def _get_dataset_img_paths(self, class_dataset_path):
        """
        Returns image paths of a given dataset path
        """
        class_image_paths = [
            path for path in class_dataset_path.iterdir() if path.suffix.lower() in consts.IMG_SUFFIX_LIST
            ]
        return class_image_paths

    def _get_class_dataset_instances(self, class_dataset_img_paths):
        """
        Returns instance id's of all given image paths
        """
        class_dataset_instances = np.unique(
            [path.name.split('_')[-2] for path in class_dataset_img_paths]
        )
        return class_dataset_instances

    def _get_class_instance_img_paths(self, class_dataset_instance, class_dataset_img_paths):
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

    def __len__(self):
        return len(self._instances_list)

    def __getitem__(self, idx):
        object_instance = self._instances_list[idx]
        image_tensor = object_instance.get_image_tensor()
        target_tensor = object_instance.get_target_tensor()
        instance_attributes = object_instance.get_attributes()
        return image_tensor, target_tensor, instance_attributes




