"""Provides an interface to all training and test data."""

import os
from typing import Callable

from torch.utils.data.dataset import Dataset
from PIL import Image
import h5py
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------

class CrowdCountingDataset(Dataset):
    def __init__(self, *, data_path: str,
                          mode: str,
                          main_transform: Callable=None,
                          img_transform: Callable=None,
                          gt_transform: Callable=None):
        """
        Args:
            data_path (str): Path to the dataset
            mode (str): Can be either 'train' or 'test'
            main_transform (Callable, optional): Transformation that takes both
                image and density map and returns the transformed versions.
            img_transform (Callable, optional): Transformation that is applied
                only to the image.
            gt_transform (Callable, optional): Transformation that is applied
                only to the ground-truth (density map).
        """
        super().__init__()

        # First, catch potential errors
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Path '{data_path}' does not exist!")

        if not (mode=='train' or mode=='test'):
            raise ValueError(f"Invalid mode '{mode}'. Must be either 'train'"
                             "or 'test'.")

        assert main_transform is None or callable(main_transform)
        assert gt_transform is None or callable(gt_transform)
        assert img_transform is None or callable(img_transform)

        # Load the data
        self.data_path = data_path
        self.mode = mode
        self.images, self.gt = self._load_data()

        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

        print(f"Initialized '{os.path.split(self.data_path)[1]}' dataset in "
              f"{self.mode}ing mode (size: {len(self)})")

    def __getitem__(self, idx):
        """Provides access via index."""
        img, den = self.images[idx], self.gt[idx]

        if self.main_transform is not None:
            img, den = self.main_transform(img, den)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.gt_transform is not None:
            den = self.gt_transform(den)

        return img, den

    def __len__(self):
        return len(self.images)

    def _load_data(self):
        raise NotImplementedError()

class MallDataset(CrowdCountingDataset):
    """The mall dataset.

    https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html
    """
    def _load_data(self):
        """Load image data and ground-truth density maps."""
        img_path = os.path.join(self.data_path , "frames", "seq_{idx:06d}.jpg")
        gt_path = os.path.join(self.data_path, "density_maps.h5")
        
        np.random.seed(0)
        train_idxs, test_idxs = np.split(np.random.permutation(2000), [1600])

        if self.mode == 'train':
            idxs = train_idxs
        else:
            idxs = test_idxs

        images, gt = [], []

        for i in tqdm(idxs,
                      desc=f"Loading '{os.path.split(self.data_path)[1]}' "
                           f"{self.mode}ing data..."):
            # Load image
            img = Image.open(img_path.format(idx=i+1))
            if img.mode == 'L':
                img = img.convert('RGB')

            # Load density map
            with h5py.File(gt_path, 'r') as f:
                dm = f[str(i+1)][()]
            
            # Store both
            images.append(img)
            gt.append(dm)

        return images, gt

class ShanghaiTechDataset(CrowdCountingDataset):
    """The ShanghaiTech (part A/B) dataset.

    https://www.kaggle.com/tthien/shanghaitech
    """
    def __init__(self, *, part: str, **kwargs):
        """"""
        super().__init__(**kwargs)
        self.part = part

    def _load_data(self):
        """Load image data and ground-truth density maps."""
        if self.mode == 'train':
            data_path = os.path.join(self.data_path, "train_data")
        else:
            data_path = os.path.join(self.data_path, "test_data")

        img_path = os.path.join(data_path , "images", "IMG_{idx}.jpg")
        gt_path = os.path.join(data_path, "density_maps.h5")

        images, gt = [], []

        if self.part=='A' and self.mode=='test':
            num_images = 182
        elif self.part=='A' and self.mode=='train':
            num_images = 300
        elif self.part=='B' and self.mode=='test':
            num_images = 316
        elif self.part=='B' and self.mode=='train':
            num_images = 400

        for i in tqdm(range(num_images),
                      desc=f"Loading '{os.path.split(self.data_path)[1]}' "
                           f"{self.mode}ing data..."):
            # Load image
            img = Image.open(img_path.format(idx=i+1))
            if img.mode == 'L':
                img = img.convert('RGB')

            # Load density map
            with h5py.File(gt_path, 'r') as f:
                dm = f[str(i+1)][()]
            
            # Store both
            images.append(img)
            gt.append(dm)

        return images, gt

class UCF_CC_50Dataset(CrowdCountingDataset):
    """The UCF_CC_50 dataset.

    https://www.kaggle.com/tthien/ucfcc50/data
    """
    def _load_data(self):
        """Load image data and ground-truth density maps."""
        img_path = os.path.join(self.data_path , "UCF_CC_50", "{idx}.jpg")
        gt_path = os.path.join(self.data_path, "density_maps.h5")
        
        np.random.seed(0)
        train_idxs, test_idxs = np.split(np.random.permutation(50), [30])

        if self.mode == 'train':
            idxs = train_idxs
        else:
            idxs = test_idxs

        images, gt = [], []

        for i in tqdm(idxs,
                      desc=f"Loading '{os.path.split(self.data_path)[1]}' "
                           f"{self.mode}ing data..."):
            # Load image
            img = Image.open(img_path.format(idx=i+1))
            if img.mode == 'L':
                img = img.convert('RGB')

            # Load density map
            with h5py.File(gt_path, 'r') as f:
                dm = f[str(i+1)][()]
            
            # Store both
            images.append(img)
            gt.append(dm)

        return images, gt
