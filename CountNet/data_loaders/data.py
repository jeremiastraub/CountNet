""""""

import os

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from torch.utils.data.dataset import Dataset

# -----------------------------------------------------------------------------

class MallDataset(Dataset):
    """The mall dataset"""
    def __init__(self, path_to_folder: str, *, train: bool=True,
                                               transform=None):
        """Initializes the MallDataset
        
        Args:
            path_to_folder (str): Path to the folder containing the mall-data
            train (bool, optional): If true, the training data is loaded. Else,
                the test data is loaded.
            transform (None, optional): Transformations to apply to the data
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

        np.random.seed(0)
        train_idxs, test_idxs = np.split(np.random.permutation(2000), [1800])

        if train:
            idxs = train_idxs
        else:
            idxs = test_idxs

        # Load the images
        # TODO For now, only import the first 10 images. Replace with `idxs`.
        frames = []
        for i in range(10):
            frames.append(plt.imread(frames_path.format(idx=i+1)))

        # Load the groundtruth
        gt = loadmat(gt_path)

        self.data = frames
        self.count = gt['count'][:10]
        # The positions are stored in a nested structure of arrays and tuples,
        # hence the somewhat ugly item access...
        self.pos = [gt['frame'][:,i][0][0][0][0] for i in range(10)]

    def __getitem__(self, idx):
        """Returns tuple of image, total count, and positions at given index"""
        frame, count, pos = self.data[idx], self.count[idx], self.pos[idx]

        if self.transform is not None:
            frame, count, pos = self.transform(frame, count, pos)

        # TODO Find a consistent output format here
        return frame, (count, pos)

    def __len__(self):
        return len(self.data)
