"""Transformations that can be passed to the datasets"""

from Typing import Union, Tuple

import numpy as np
from skimage.transform import downscale_local_mean

# -----------------------------------------------------------------------------

class Downscale_Image_GT():
    def __init__(self, downscaling_factor: Union[int, Tuple[int]]):
        """Transformation that, if applied to an (image, density-map) pair,
        downscales both by the given downscaling-factor. The density-map is
        then rescaled such that it sums up to 1.

        The images are assumed to be of shape (C, H, W), the density-maps are
        assumed to be of shape (H, W). (C: channels, H: height, W: width).

        Args:
            downscaling_factor (Union[int, Tuple[int]]): The downscaling-factor
        """
        if isinstance(downscaling_factor, int):
            downscaling_factor = (1, downscaling_factor, downscaling_factor)

        elif len(downscaling_factor) == 2:
            downscaling_factor = (1, downscaling_factor[0],
                                     downscaling_factor[1])

        self.downscaling_factor = downscaling_factor

    def __call__(self, image, density_map):
        """Applies the transformation to an image and its density map."""
        img_red = downscale_local_mean(image, factors=downscaling_factor)
        density_map_red = downscale_local_mean(density_map,
                                               factors=downscaling_factor[1:])

        density_map_red /= density_map_red.sum()

        return [img_red, density_map_red]
