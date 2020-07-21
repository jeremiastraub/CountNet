"""Helper script for generating the ground-truth density maps"""

import numpy as np
import os
import h5py
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from scipy.io import loadmat
from PIL import Image

# -----------------------------------------------------------------------------

def density_map_from_annotations(annotations, img_shape, sigma):
    """Generates a heatmap from annotations.
    
    Args:
        annotations (array): (2d) Annotation coordinates
        img_shape (Tuple[int, int]): The image shape
        sigma: The standard deviation used for the gaussian blurring
    
    Returns:
        array: The density map (normalized to N)
    """
    # Create annotation map with delta-peaks
    dm = np.zeros(img_shape)

    for y, x in annotations:
        try:
            dm[int(x), int(y)] = 1
        except:
            pass

    # Blur the annotations with a gaussian
    dm = gaussian_filter(dm, sigma, mode='constant')
    
    return dm

def generate_mall_density_maps(sigma=None):
    """
    Args:
        sigma (optional): The standard deviation used for the gaussian
            blurring. If None, it is chosen adaptively.
    """
    path_to_images = "Mall/frames/seq_{idx:06d}.jpg"
    path_to_gt = "Mall/mall_gt.mat"
    outpath_dm = "Mall/density_maps.h5"
    
    with h5py.File(outpath_dm, 'w') as f:
        if sigma is not None:
            # Also store the sigma value
            f.create_dataset("sigma", data=[sigma])

        for i in tqdm(range(2000), desc="Generating Mall density maps"):
            # Load image to get image shape
            img = Image.open(path_to_images.format(idx=i+1))
            img_shape = (img.size[1], img.size[0])
            if sigma is None:
                    sigma = img_shape[0]/150.
            # Get annotation points
            gt = loadmat(path_to_gt)
            pos = gt['frame'][0][i][0][0][0]
            # Create and store density map
            dm = density_map_from_annotations(pos, img_shape, sigma)
            f.create_dataset(f"{i+1}", data=dm, compression='gzip',
                                                compression_opts=9)

def generate_ufc_density_maps(sigma=None):
    """
    Args:
        sigma (optional): The standard deviation used for the gaussian
            blurring. If None, it is chosen adaptively.
    """
    path_to_images = "UCF_CC_50/UCF_CC_50/{idx}.jpg"
    path_to_gt = "UCF_CC_50/UCF_CC_50/{idx}_ann.mat"
    outpath_dm = "UCF_CC_50/density_maps.h5"
    
    with h5py.File(outpath_dm, 'w') as f:
        if sigma is not None:
            # Also store the sigma value
            f.create_dataset("sigma", data=[sigma])

        for i in tqdm(range(50), desc="Generating UFC_CC_50 density maps"):
            # Load image to get image shape
            img = Image.open(path_to_images.format(idx=i+1))
            img_shape = (img.size[1], img.size[0])
            if sigma is None:
                    sigma = img_shape[0]/150.
            # Get annotation points
            gt = loadmat(path_to_gt.format(idx=i+1))
            pos = gt['annPoints']
            # Create and store density map
            dm = density_map_from_annotations(pos, img_shape, sigma)
            f.create_dataset(f"{i+1}", data=dm, compression='gzip',
                                                compression_opts=9)

def generate_shanghaitech_density_maps(part: str, sigma=None):
    """
    Args:
        part (str): Which part of the dataset to use. Either 'A' or 'B'.
        sigma (optional): The standard deviation used for the gaussian
            blurring. If None, it is chosen adaptively.
    """
    path_to_images = ("ShanghaiTech_" + f"{part}"
                      + "/{mode}_data/images/IMG_{idx}.jpg")
    path_to_gt = ("ShanghaiTech_" + f"{part}"
                  + "/{mode}_data/ground-truth/GT_IMG_{idx}.mat")
    outpath_dm = "ShanghaiTech_" + f"{part}" + "/{mode}_data/density_maps.h5"
    
    for mode in ['test', 'train']:
        with h5py.File(outpath_dm.format(mode=mode), 'w') as f:
            if sigma is not None:
                # Also store the sigma value
                f.create_dataset("sigma", data=[sigma])

            if part=='A' and mode=='test':
                num_images = 182
            elif part=='A' and mode=='train':
                num_images = 300
            elif part=='B' and mode=='test':
                num_images = 316
            elif part=='B' and mode=='train':
                num_images = 400

            for i in tqdm(range(num_images), desc="Generating "
                          f"ShanghaiTech_{part} {mode} density maps"):
                # Load image to get image shape
                img = Image.open(path_to_images.format(mode=mode, idx=i+1))
                img_shape = (img.size[1], img.size[0])
                if sigma is None:
                    sigma = img_shape[0]/150.
                # Get annotation points
                gt = loadmat(path_to_gt.format(mode=mode, idx=i+1))
                pos = gt['image_info'].item().item()[0]
                # Create and store density map
                dm = density_map_from_annotations(pos, img_shape, sigma)
                f.create_dataset(f"{i+1}", data=dm, compression='gzip',
                                                    compression_opts=9)


if __name__ == '__main__':
    # Uncomment below to run the script
    pass
    # generate_mall_density_maps()#sigma=5
    # generate_ufc_density_maps()#sigma=3
    # generate_shanghaitech_density_maps(part='A')#sigma=8
    # generate_shanghaitech_density_maps(part='B')#sigma=8