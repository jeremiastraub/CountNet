"""Provides an interface to all training and test data."""

import os
from typing import Callable

from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------

class CrowdCountingDataset(Dataset):
    def __init__(self, data_path: str,
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

        if main_transform is not None and not callable(main_transform):
            raise TypeError("main_transform must be callable! "
                            f"Received type: {type(main_transform)}")

        if img_transform is not None and not callable(img_transform):
            raise TypeError("img_transform must be callable! "
                            f"Received type: {type(img_transform)}")

        if gt_transform is not None and not callable(gt_transform):
            raise TypeError("gt_transform must be callable! "
                            f"Received type: {type(gt_transform)}")

        # Load the data
        self.mode = mode
        self.images, self.gt = self._load_data(data_path, mode)

        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

        print(f"Initialized '{os.path.split(data_path)[1]}' dataset in "
              f"{mode}ing mode (size: {len(self)}.")

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

    def _load_data(self, data_path, mode):
        """Load image data and ground-truth density maps."""
        # TODO If the stored data has a uniform structure, do it as it is.
        #      If *not*, use this as base class and derive from it (overwriting
        #      the `_load_data` method in the derived Dataset classes.
        # TODO Change depending on the storage format of train and test data.
        #      Currently, expects train data at ".../img/train/" and
        #      ".../den/train/".
        img_path = os.path.join(data_path , 'img', mode)
        gt_path = os.path.join(data_path, 'den', mode)
        
        images, gt = [], []

        # FIXME `os.listdir` does not guarantee for correct ordering! This has
        #       to be handled explicitly! (or sorted, if filenames allow for
        #       that)
        for fname in [f for f in os.listdir(img_path)
                      if os.path.isfile(os.path.join(img_path, f))]:

            img = Image.open(os.path.join(img_path, fname))
            if img.mode == 'L':
                img = img.convert('RGB')

            den = pd.read_csv(os.path.join(gt_path,
                                           os.path.splitext(fname)[0]+'.csv'),
                              sep=',', header=None).values
            den = den.astype(np.float32, copy=False)
            den = Image.fromarray(den)
            
            images.append(img)
            gt.append(den)

        images = np.stack(images)
        gt = np.stack(gt)

        return images, gt
