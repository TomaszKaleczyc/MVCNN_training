class MVCNNObjectClass:
    """
    Contains object class metadata
    """

    def __init__(self, idx: int, class_path: str, num_classes: int):
        self._id = idx
        self._path = class_path
        self._name = class_path.name
        self._num_classes = num_classes
        self._instance_id_list = []
        self._instance_img_count = 0

    def get_path(self):
        """
        Returns path to class data
        """
        return self._path

    def get_attributes(self):
        """
        Returns basic class attributes
        """
        return self._id, self._name, self._num_classes

    def get_summary(self) -> dict:
        """
        Returns attributes for the summary DataFrame
        """
        summary_dict = {
            'class_id': self._id,
            'class_name': self._name,
            'num_instances': len(self._instance_id_list),
            'num_images': self._instance_img_count,
        }
        return summary_dict

    def update_summary(self, class_instance):
        """
        Updates class summary with class instance attributes
        """
        instance_attributes = class_instance.get_attributes()
        self._instance_id_list.append(instance_attributes['instance_id'])
        self._instance_img_count += instance_attributes['image_count']

    def print_summary(self):
        """
        Prints out details of class data
        """
        print('-'*60)
        print('Class name:', self._name.upper())
        print('Total number of instances:', len(self._instance_id_list))
        print('Total number of images:', self._instance_img_count)