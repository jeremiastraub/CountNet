"""Visualizing the model ouput"""

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .plot_utils import load_trainer

# -----------------------------------------------------------------------------

def plot_example_output(tag: str,
                        datapath: str,
                        path_run_cfg: str,
                        path_dataset_cfg: str,
                        outpath:str=None):
    """Visualize some exemplary model output.
    
    Args:
        tag (str): Description
        datapath (str): Path to the folder containing the output
        path_run_cfg (str): Path to the run configuration file
        path_dataset_cfg (str): Path to the datasets configuration file
        outpath (str, optional): Path to where the figures are saved
    """
    trainer = load_trainer(path_run_cfg, path_dataset_cfg)

    # Visualize a few predictions
    trainer.model.to(device='cpu')

    fig, axes = plt.subplots(2,3, gridspec_kw={'hspace': 0.1})

    for i, (img, dm) in zip(range(3), trainer.loader_test):
        pred = trainer.model(img).detach().numpy().squeeze()
        img = img[0,0,...].numpy()
        dm = dm[0,0,...].numpy()

        axes[0,i].imshow(img, cmap='Greys_r')
        axes[0,i].imshow(dm, alpha=0.6)
        axes[1,i].imshow(pred, vmin=0)

        axes[0,i].axis('off')
        axes[1,i].axis('off')
        axes[0,i].set_title(f"N = {np.sum(dm):.0f}")
        axes[1,i].set_title(f"N = {np.sum(pred):.0f}")

    if outpath is not None:
        fig.savefig(outpath, bbox_inches='tight')

    return fig, axes
