from pathlib import Path

from settings import consts


def get_num_classes(num_classes):
    """
    Determines the number of classes to be used in training
    """
    if num_classes is None:
        num_classes = len(list(Path(consts.DATA_DIR).iterdir()))
    return num_classes