"""This module implements Crowd-Counting datasets."""

import os

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from torch.utils.data.dataset import Dataset

# -----------------------------------------------------------------------------
# Base class

class CrowdCountingDataset(Dataset):
    """Base class for Crowd-counting datasets. Derived classes overwrite the
    ``_extract_data`` method.
    """
    def __init__(self, path_to_folder: str, *, train: bool=True,
                                               transform=None):
        """Initializes the dataset.
        
        Args:
            path_to_folder (str): Path to the folder containing the data
            train (bool, optional): If true, the training data is loaded. Else,
                the test data is loaded.
            transform (Callable, optional): A function/transform that takes the
                image and the target and returns the tranformed versions.
        """
        super().__init__()

        if not os.path.exists(path_to_folder):
            raise FileNotFoundError(f"Path '{path_to_folder}' does not exist!")

        self.path = path_to_folder
        self.train = train
        self.transform = transform

        # TODO Store the data as HDF5 once and adapt this function to read
        #      from the HDF5 file. This is much quicker and allows for a
        #      uniform datastructure.
        image_data, density = self._extract_data()

        self.data = image_data
        self.density = density

    def __getitem__(self, idx):
        """Returns tuple of image, total count, and positions at given index"""
        if self.transform is not None:
            return self.transform(self.data[idx], self.density[idx])

        return self.data[idx], self.density[idx]

    def __len__(self):
        return len(self.data)

    def _extract_data(self):
        """Extracts the image, count, and position data"""
        raise NotImplementedError()

# -----------------------------------------------------------------------------
# Derived classes

# FIXME Generate ground-truth density maps and load the data from the generated
#       ground-truth.

class MallDataset(CrowdCountingDataset):
    """The mall dataset.

    https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html
    """
    def _extract_data(self):
        """Extracts the image, count, and position data"""
        # Assemble the paths to the image data and the ground truth
        frames_path = os.path.join(self.path, "frames/seq_"+"{idx:06d}"+".jpg")
        gt_path = os.path.join(self.path, "mall_gt.mat")

        np.random.seed(0)
        train_idxs, test_idxs = np.split(np.random.permutation(2000), [1800])

        if self.train:
            idxs = train_idxs
        else:
            idxs = test_idxs

        # Load the images
        # TODO For now, only import the first 10 images. Replace with `idxs`.
        frames = []
        for i in range(10):
            img = plt.imread(frames_path.format(idx=i+1)).astype(float)
            img = np.transpose(img, (2,0,1))
            img /= img.max()
            frames.append(img)

        # Load the groundtruth
        gt = loadmat(gt_path)

        images = frames
        count = gt['count'][:10]
        pos = [gt['frame'][:,i][0][0][0][0] for i in range(10)]

        return images, count, pos


class ShanghaiTechDataset(CrowdCountingDataset):
    """The ShanghaiTech (part A/B) dataset.

    https://www.kaggle.com/tthien/shanghaitech
    """
    def _extract_data(self):
        """Extracts the image, count, and position data"""
        if self.train:
            data_path = os.path.join(self.path, "train_data")
        else:
            data_path = os.path.join(self.path, "test_data")

        # Assemble the paths to the image data and the ground truth
        image_path = os.path.join(data_path, "images/IMG_"+"{idx}"+".jpg")
        gt_path = os.path.join(data_path, "ground-truth/GT_IMG_{idx}.mat")

        # Load the images and groundtruth
        # TODO For now, only import the first 10 images.
        images = []
        gt = []
        for i in range(10):
            img = plt.imread(image_path.format(idx=i+1)).astype(float)
            img = np.transpose(img, (2,0,1))
            img /= img.max()
            images.append(img)
            gt.append(loadmat(gt_path.format(idx=i+1)))

        count = [label['image_info'].item().item()[1].astype('uint8')
                      for label in gt]
        pos = [label['image_info'].item().item()[0] for label in gt]

        return images, count, pos


class UFC_CC_50Dataset(CrowdCountingDataset):
    """The UFC_CC_50 dataset.

    https://www.kaggle.com/tthien/ucfcc50/data
    """
    def _extract_data(self):
        """Extracts the image, count, and position data"""
        # Assemble the paths to the image data and the ground truth
        image_path = os.path.join(self.path, "{idx}.jpg")
        gt_path = os.path.join(self.path, "{idx}_ann.mat")

        np.random.seed(0)
        train_idxs, test_idxs = np.split(np.random.permutation(50), [40])

        if self.train:
            idxs = train_idxs
        else:
            idxs = test_idxs

        # Load the images and groundtruth
        images = []
        gt = []
        for i in idxs:
            img = plt.imread(image_path.format(idx=i+1)).astype(float)
            img = np.transpose(img, (2,0,1))
            img /= img.max()
            images.append(img)
            gt.append(loadmat(gt_path.format(idx=i+1)))

        count = [len(label['annPoints']) for label in gt]
        pos = [label['annPoints'] for label in gt]

        return images, count, pos
