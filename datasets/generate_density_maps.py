"""Helper script for generating the ground-truth density maps"""

import numpy as np
import os
import h5py
from tqdm import tqdm
from scipy.spatial import KDTree
from scipy.ndimage.filters import gaussian_filter
from scipy.io import loadmat
from PIL import Image

# -----------------------------------------------------------------------------

def density_map_from_annotations(annotations, img_shape, sigma=None):
    """Generates a heatmap from annotations.
    
    Args:
        annotations (array): (2d) Annotation coordinates
        img_shape (Tuple[int, int]): The image shape
        sigma (optional): The standard deviation used for the gaussian
            blurring. If None, sigma is chosen adaptively.
    
    Returns:
        array: The density map (normalized to N)
    """
    dm = np.zeros(img_shape)

    n_points = len(annotations)
    if n_points == 0:
        return dm

    # Create density map with fixed sigma
    if sigma is not None:
        for y, x in annotations:
            try:
                dm[int(x), int(y)] = 1.
            except:
                pass

        # Blur the annotations with a gaussian
        dm = gaussian_filter(dm, sigma, mode='constant')
        
        return dm

    # Determine sigma based on distance to nearest neighbors
    else:
        tree = KDTree(annotations.copy(), leafsize=2048)
        distances, _ = tree.query(annotations, k=3)

        for i, pt in enumerate(annotations):
            pt2d = np.zeros(img_shape, dtype=np.float32)
            try:
                pt2d[int(pt[1]),int(pt[0])] = 1.
            except:
                continue

            if n_points > 1:
                sigma = (distances[i][0]+distances[i][1]+distances[i][2])*0.15
            else:
                sigma = np.average(np.array(img_shape))/4. # single point
                
            dm += gaussian_filter(pt2d, sigma, mode='constant')
        
        return dm

def generate_mall_density_maps(sigma=None, adaptive: bool=False):
    """
    Args:
        sigma (optional): The standard deviation used for the gaussian
            blurring. If None, it is chosen adaptively.
        adaptive (bool, optional): Whether to choose sigma adaptively. If True,
            the `sigma` argument is ignored.
    """
    path_to_images = "Mall/frames/seq_{idx:06d}.jpg"
    path_to_gt = "Mall/mall_gt.mat"
    outpath_dm = "Mall/density_maps.h5"
    
    with h5py.File(outpath_dm, 'w') as f:
        if sigma is not None and not adaptive:
            # Also store the sigma value
            f.create_dataset("sigma", data=[sigma])

        for i in tqdm(range(2000), desc="Generating Mall density maps"):
            # Load image to get image shape
            img = Image.open(path_to_images.format(idx=i+1))
            img_shape = (img.size[1], img.size[0])

            if adaptive:
                sigma = None

            elif sigma is None:
                sigma = img_shape[0]/150.

            # Get annotation points
            gt = loadmat(path_to_gt)
            pos = gt['frame'][0][i][0][0][0]
            # Create and store density map
            dm = density_map_from_annotations(pos, img_shape, sigma)
            f.create_dataset(f"{i+1}", data=dm, compression='gzip',
                                                compression_opts=9)

def generate_ufc_density_maps(sigma=None, adaptive: bool=False):
    """
    Args:
        sigma (optional): The standard deviation used for the gaussian
            blurring. If None, it is chosen adaptively.
        adaptive (bool, optional): Whether to choose sigma adaptively. If True,
            the `sigma` argument is ignored.
    """
    path_to_images = "UCF_CC_50/UCF_CC_50/{idx}.jpg"
    path_to_gt = "UCF_CC_50/UCF_CC_50/{idx}_ann.mat"
    outpath_dm = "UCF_CC_50/density_maps.h5"
    
    with h5py.File(outpath_dm, 'w') as f:
        if sigma is not None and not adaptive:
            # Also store the sigma value
            f.create_dataset("sigma", data=[sigma])

        for i in tqdm(range(50), desc="Generating UFC_CC_50 density maps"):
            # Load image to get image shape
            img = Image.open(path_to_images.format(idx=i+1))
            img_shape = (img.size[1], img.size[0])
            
            if adaptive:
                sigma = None

            elif sigma is None:
                sigma = img_shape[0]/150.

            # Get annotation points
            gt = loadmat(path_to_gt.format(idx=i+1))
            pos = gt['annPoints']
            # Create and store density map
            dm = density_map_from_annotations(pos, img_shape, sigma)
            f.create_dataset(f"{i+1}", data=dm, compression='gzip',
                                                compression_opts=9)

def generate_shanghaitech_density_maps(part: str, sigma=None,
                                       adaptive: bool=False):
    """
    Args:
        part (str): Which part of the dataset to use. Either 'A' or 'B'.
        sigma (optional): The standard deviation used for the gaussian
            blurring. If None, it is chosen adaptively.
        adaptive (bool, optional): Whether to choose sigma adaptively. If True,
            the `sigma` argument is ignored.
    """
    path_to_images = ("ShanghaiTech_" + f"{part}"
                      + "/{mode}_data/images/IMG_{idx}.jpg")
    path_to_gt = ("ShanghaiTech_" + f"{part}"
                  + "/{mode}_data/ground-truth/GT_IMG_{idx}.mat")
    outpath_dm = "ShanghaiTech_" + f"{part}" + "/{mode}_data/density_maps.h5"
    
    for mode in ['test', 'train']:
        with h5py.File(outpath_dm.format(mode=mode), 'w') as f:
            if sigma is not None and not adaptive:
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
                
                if adaptive:
                    sigma = None

                elif sigma is None:
                    sigma = img_shape[0]/150.

                # Get annotation points
                gt = loadmat(path_to_gt.format(mode=mode, idx=i+1))
                pos = gt['image_info'].item().item()[0]
                # Create and store density map
                dm = density_map_from_annotations(pos, img_shape, sigma)
                f.create_dataset(f"{i+1}", data=dm, compression='gzip',
                                                    compression_opts=9)


if __name__ == '__main__':
    generate_mall_density_maps(sigma=6, adaptive=False)
    generate_ufc_density_maps(adaptive=True)
    generate_shanghaitech_density_maps(part='A', adaptive=True)
    generate_shanghaitech_density_maps(part='B', adaptive=True)