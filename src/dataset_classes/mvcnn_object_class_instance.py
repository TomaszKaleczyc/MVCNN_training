from matplotlib import pyplot as plt


class MVCNNObjectClassInstance:
    """
    Contains object class instance metadata and methods
    """

    def __init__(self, mvcnn_class, class_instance_id, class_instance_img_paths):
        self._class_id, self._class_name = mvcnn_class.get_attributes()
        self._img_paths = class_instance_img_paths
        self._instance_id = class_instance_id

    def __len__(self):
        return len(self._img_paths)

    def get_images(self):
        """
        Returns images as a list of numpy arrays
        """
        return [plt.imread(img_path) for img_path in self._img_paths]

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

    def get_tensors(self):
        """
        Returns all instance images as torch tensor
        expected by the model
        """
        raise NotImplementedError
