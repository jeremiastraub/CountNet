"""Visualizing the loss development"""

from typing import List
import matplotlib.pyplot as plt
from .plot_utils import extract_loss_information

# -----------------------------------------------------------------------------

def plot_loss_single(tag: str, datapath: str, outpath: str=None):
    """Plot loss over iterations.
    
    Args:
        tag (str): The tag of the checkpoint
        datapath (str): Path to stored results
        outpath (str, optional): Path to where the figure is saved
    
    Returns:
        Tuple (fig, ax) containing the created plot
    """
    loss, steps, epoch = extract_loss_information(tag, datapath)

    fig, ax = plt.subplots()
    ax.plot(steps, loss)

    if outpath is not None:
        fig.savefig(outpath, bbox_inches='tight')

    else:
        plt.show()

    return fig, ax

def plot_loss_multiple(tags: List[str],
                       datapath: str,
                       labels: List[str]=None,
                       outpath:str=None):
    """Plot loss over iterations for multiple tags in a single plot.
    
    Args:
        tags (List[str]): List of checkpoint/run tags
        datapath (str): Path to stored results
        labels (List[str], optional): List of legend labels
        outpath (str, optional): Path to where the figure is saved
    
    Returns:
        Tuple (fig, ax) containing the created plot
    """
    labels_exist = labels is not None
    if labels_exist:
        assert len(tags) == len(labels)

    fig, ax = plt.subplots()

    for i, tag in enumerate(tags):
        loss, steps, epoch = extract_loss_information(tag, datapath)
        ax.plot(steps, loss, label=(labels[i] if labels_exist else None))

    if labels_exist:
        ax.legend()

    if outpath is not None:
        fig.savefig(outpath, bbox_inches='tight')

    else:
        plt.show()

    return fig, ax
