class MVCNNObjectClass:
    """
    Contains object class metadata
    """

    def __init__(self, idx, class_path):
        self._class_id = idx
        self._class_path = class_path
        self._class_name = class_path.name

    def get_path(self):
        """
        Returns path to class data
        """
        return self._class_path

    def get_attributes(self):
        """
        Returns basic class attributes
        """
        return self._class_id, self._class_name