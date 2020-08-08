"""Visualizing the validation metric development"""

from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from .plot_utils import extract_loss_information, extract_metric_information

# -----------------------------------------------------------------------------

def plot_loss_single(tag: str, datapath: str, outpath: str=None, skip: int=0):
    """Plot loss over iterations.
    
    Args:
        tag (str): The tag of the checkpoint
        datapath (str): Path to stored results
        outpath (str, optional): Path to where the figure is saved
        skip (int, optional): Skip initial steps
    
    Returns:
        Tuple (fig, ax) containing the created plot
    """
    loss, steps, epoch = extract_loss_information(tag, datapath)

    sns.set()
    sns.set(font_scale=1.2)
    # sns.set_style('ticks')

    fig, ax = plt.subplots()
    ax.plot(steps[skip:], loss[skip:])
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE Loss")
    sns.despine()

    if outpath is not None:
        fig.savefig(outpath, bbox_inches='tight')

    return fig, ax

def plot_loss_multiple(tags: List[str],
                       datapath: str,
                       labels: List[str]=None,
                       outpath:str=None,
                       skip: int=0):
    """Plot loss over iterations for multiple tags in a single plot.
    
    Args:
        tags (List[str]): List of checkpoint/run tags
        datapath (str): Path to stored results
        labels (List[str], optional): List of legend labels
        outpath (str, optional): Path to where the figure is saved
        skip (int, optional): Skip initial steps
    
    Returns:
        Tuple (fig, ax) containing the created plot
    """
    labels_exist = labels is not None
    if labels_exist:
        assert len(tags) == len(labels)

    sns.set()
    sns.set(font_scale=1.2)
    fig, ax = plt.subplots()

    for i, tag in enumerate(tags):
        loss, steps, epoch = extract_loss_information(tag, datapath)
        ax.plot(steps[skip:], loss[skip:],
                label=(labels[i] if labels_exist else None))

    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE Loss")
    if labels_exist:
        ax.legend()
    sns.despine()

    if outpath is not None:
        fig.savefig(outpath, bbox_inches='tight')

    return fig, ax

def plot_metric_single(tag: str,
                       metric: str,
                       datapath: str,
                       outpath: str=None):
    """Plot a validation metric over iterations.
    
    Args:
        tag (str): The tag of the checkpoint
        metric (str): The metric to be plotted
        datapath (str): Path to stored results
        outpath (str, optional): Path to where the figure is saved
    
    Returns:
        Tuple (fig, ax) containing the created plot
    """
    values, epochs = extract_metric_information(tag, metric, datapath)

    sns.set()
    sns.set(font_scale=1.2)

    fig, ax = plt.subplots()
    ax.plot(epochs, values)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    sns.despine()

    if outpath is not None:
        fig.savefig(outpath, bbox_inches='tight')

    return fig, ax

def plot_metric_multiple(tags: List[str],
                         metric: str,
                         datapath: str,
                         labels: List[str]=None,
                         outpath: str=None):
    """Plot a validation metric over iterations for multiple tags.
    
    Args:
        tags (List[str]): List of checkpoint/run tags
        metric (str): The metric to be plotted
        datapath (str): Path to stored results
        labels (List[str], optional): Description
        outpath (str, optional): Path to where the figure is saved
    
    Returns:
        Tuple (fig, ax) containing the created plot
    """
    labels_exist = labels is not None
    if labels_exist:
        assert len(tags) == len(labels)

    sns.set()
    sns.set(font_scale=1.2)
    fig, ax = plt.subplots()

    for i, tag in enumerate(tags):
        values, epochs = extract_metric_information(tag, metric, datapath)
        ax.plot(epochs, values, label=(labels[i] if labels_exist else None))

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    if labels_exist:
        ax.legend()
    sns.despine()

    if outpath is not None:
        fig.savefig(outpath, bbox_inches='tight')

    return fig, ax
