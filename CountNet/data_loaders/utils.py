"""Data Loading Utilities"""

import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------

def read_jpg(path_to_jpg: str):
    """Read a JPEG image as numpy array"""
    # NOTE `dtype` is not changed (int), could be adjusted here
    return np.array(Image.open(path_to_jpg))