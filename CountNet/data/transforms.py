"""Transformations that can be passed to the datasets"""

from typing import Union, Tuple

from PIL import Image
import numpy as np
from skimage.transform import downscale_local_mean
from torchvision.transforms import ToTensor

# -----------------------------------------------------------------------------

class RandomCrop_Image_GT():
    """Transformation that, if applied to an (image, density-map) pair,
    randomly crops out and returns equivalent patches of the given size.
    
    Images are assumed to be of type PIL.Image, density-maps are assumed to be
    of type np.ndarray (2d).
    """
    def __init__(self, crop_size: Union[int, Tuple[int]]):
        """
        Args:
            crop_size (Union[int, Tuple[int]]): The size of the cropped patch
                (height, width). If an integer is given, a square patch is
                cropped.
        """
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert len(crop_size) == 2

        self.crop_size = crop_size

    def __call__(self, image, density_map):
        """Applies the transformation to an image and its density map."""
        assert isinstance(image, Image.Image)
        assert isinstance(density_map, np.ndarray)
        assert density_map.ndim == 2

        # Get random box position
        if density_map.shape[0] == self.crop_size[0]:
            x_origin = 0
        else:
            x_origin = np.random.randint(0, density_map.shape[0]
                                            -self.crop_size[0])
        
        if density_map.shape[1] == self.crop_size[1]:
            y_origin = 0
        else:
            y_origin = np.random.randint(0, density_map.shape[1]
                                            -self.crop_size[1])

        # Crop images
        img_cropped = image.crop((x_origin, y_origin,
                                  x_origin+self.crop_size[0],
                                  y_origin+self.crop_size[1]))

        density_map_cropped = density_map[x_origin:x_origin+self.crop_size[0],
                                          y_origin:y_origin+self.crop_size[1]]

        return img_cropped, density_map_cropped


class Downscale_Image_GT():
    """Transformation that, if applied to an (image, density-map) pair,
    downscales both by the given downscaling-factor. The density-map is
    then rescaled such that it sums up to 1.

    Images are assumed to be of type PIL.Image, density-maps are assumed to
    be of type np.ndarray (2d).
    """
    def __init__(self, downscaling_factor: Union[int, Tuple[int]]):
        """
        Args:
            downscaling_factor (Union[int, Tuple[int]]): The downscaling-factor
        """
        if isinstance(downscaling_factor, int):
            downscaling_factor = (downscaling_factor, downscaling_factor, 1)

        elif len(downscaling_factor) == 2:
            downscaling_factor = (downscaling_factor[0],
                                  downscaling_factor[1], 1)

        self.downscaling_factor = downscaling_factor

    def __call__(self, image, density_map):
        """Applies the transformation to an image and its density map."""
        assert isinstance(image, Image.Image)
        assert isinstance(density_map, np.ndarray)
        assert density_map.ndim == 2

        image = np.array(image)
        img_red = downscale_local_mean(image, factors=self.downscaling_factor)
        img_red = Image.fromarray((img_red * 255).astype('uint8'), mode='RGB')

        density_map_red = downscale_local_mean(density_map,
                                        factors=self.downscaling_factor[:2])

        # Rescale the density-map such that it sums to 1
        density_map_red /= density_map_red.sum()

        return img_red, density_map_red


class ToTensor_Image_GT():
    """Converts a (image, density-map) pair to a pair of tensor objects."""
    def __call__(self, image, density_map):
        assert isinstance(image, Image.Image)
        assert isinstance(density_map, np.ndarray)
        assert density_map.ndim == 2

        image = ToTensor(image)
        density_map = torch.from_numpy(density_map[None,...])

        return image, density_map


class Compose():
    """Composes several transforms together."""
    def __init__(self, transforms: list):
        """
        Args:
            transforms (list): list of transforms to compose
        """
        assert all([callable(transform) for transform in transforms])
        self.transforms = transforms

    def __call__(self, image, density_map):
        """Sequentially applies the transformations"""
        for transform in self.transforms:
            image, density_map = transform(image, density_map)

        return image, density_map
