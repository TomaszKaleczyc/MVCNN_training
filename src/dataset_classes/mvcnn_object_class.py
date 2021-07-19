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
        
        """
        return self._class_path

    def get_attributes(self):
        """
        
        """
        return self._class_id, self._class_name