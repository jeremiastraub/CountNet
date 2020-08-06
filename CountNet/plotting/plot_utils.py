"""Utility functions for plotting"""

import os
import torch
import numpy as np
from CountNet.utils import load_yml

# -----------------------------------------------------------------------------

def extract_loss_information(tag: str, datapath: str):
    """Extracts the loss information from a checkpoint. If needed, recursively
    extracts information from parent checkpoints.

    Args:
        tag (str): The tag of the checkpoint
        datapath (str): Path to stored results
    
    Returns:
        Tuple containing a list of losses, a list of the iteration steps
        (scaled to epochs), and the total number of epochs.
    """
    path = os.path.join(datapath, tag)

    checkpoint = torch.load(os.path.join(path, "checkpoint.pt"))
    loss = checkpoint['losses']
    epoch = checkpoint['epoch'] - 1
    steps = list(np.linspace(0, epoch, len(loss)))

    config = load_yml(os.path.join(path, "configuration.yml"))
    load_from = config.get('load_from', None)
    if load_from is not None:
        load_from = os.path.split(load_from)[1]

    # If the model was loaded from another checkpoint, recursively load the
    # parent checkpoints.
    if load_from is not None:
        l_prev, _, e_prev = extract_loss_information(load_from, datapath)
        steps = (  list(np.linspace(0, e_prev, len(l_prev)))
                 + list(np.linspace(e_prev, epoch, len(loss))))
        loss = l_prev + loss

    return loss, steps, epoch
