""""""

import os

import numpy as np
from scipy.io import loadmat
from PIL import Image

import torchvision
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from .utils import read_jpg

# -----------------------------------------------------------------------------

class MallDataset(Dataset):
    """The mall dataset"""
    def __init__(self, path_to_folder: str, *, train: bool=True,
                                               transform=None):
        """Initializes the MallDataset
        
        Args:
            path_to_folder (str): Description
            train (bool, optional): Description
            transform (None, optional): Description
        """
        super().__init__()

        if not os.path.exists(path_to_folder):
            raise FileNotFoundError(f"Path '{path_to_folder}' does not exist!")

        if transform is not None and not callable(transform):
            raise TypeError("Invalid 'transform' type. Must be 'None' or "
                            f"'Callable', got: {type(transform)}")

        self.path = path_to_folder
        self.train = train
        self.transform = transform

        # TODO Store the data as HDF5 once and adapt this function to read
        #      from the HDF5 file. This is much quicker and allows for a
        #      uniform datastructure.

        # Assemble the paths to the image data and the ground truth
        frames_path = os.path.join(self.path, "frames/seq_"+"{idx:06d}"+".jpg")
        gt_path = os.path.join(self.path, "mall_gt.mat")

        # Load the images
        # NOTE For now, only import the first 10 images (since loading .jpg
        #      files from disc takes a long time)
        frames = np.empty((10, 480, 640, 3))
        for i in range(10):
            frames[i] = read_jpg(frames_path.format(idx=i+1))

        # Load the groundtruth
        gt = loadmat(gt_path)

        self.data = frames
        self.count = gt['count']
        # The positions are stored in a nested structure of arrays and tuples,
        # hence the somewhat ugly item access...
        self.pos = [gt['frame'][:,i][0][0][0][0] for i in range(2000)]

    def __getitem__(self, idx):
        """Returns tuple of image, total count, and positions at given index"""
        frame, count, pos = self.data[idx], self.count[idx], self.pos[idx]

        if self.transform is not None:
            frame, count, pos = self.transform(frame, count, pos)

        # TODO Find a consistent output format here
        return frame, (count, pos)

    def __len__(self):
        return len(self.data)