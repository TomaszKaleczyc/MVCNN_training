from pathlib import Path
from typing import Optional

from settings import data_settings


def get_num_classes(num_classes: Optional[int] = None) -> int:
    """
    Determines the number of classes to be used in training
    """
    if num_classes is None:
        num_classes = len(list(Path(data_settings.DATA_DIR).iterdir()))
    return num_classes
