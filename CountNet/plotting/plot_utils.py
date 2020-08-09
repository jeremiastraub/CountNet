"""Utility functions for plotting"""

import os
import torch
import numpy as np
from CountNet.utils import load_yml, initialize_trainer

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

    checkpoint = torch.load(os.path.join(path, "checkpoint.pt"),
                            map_location=torch.device('cpu'))
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

def extract_metric_information(tag: str, metric: str, datapath: str):
    """Extracts the metric information from a checkpoint. If needed,
    recursively extracts information from parent checkpoints.
    
    Args:
        tag (str): The tag of the checkpoint
        metric (str): The metric to be loaded
        datapath (str): Path to stored results
    
    Returns:
        Tuple containing the loaded results, a list of the iteration steps
        (scaled to epochs), and the total number of epochs.
    """
    path = os.path.join(datapath, tag)

    checkpoint = torch.load(os.path.join(path, "checkpoint.pt"),
                            map_location=torch.device('cpu'))
    validations = checkpoint['validations']
    epochs = list(validations.keys())
    values = [validations[i][metric] for i in validations.keys()]

    config = load_yml(os.path.join(path, "configuration.yml"))
    load_from = config.get('load_from', None)
    if load_from is not None:
        load_from = os.path.split(load_from)[1]

    # If the model was loaded from another checkpoint, recursively load the
    # parent checkpoints.
    if load_from is not None:
        v_prev, e_prev = extract_metric_information(load_from,
                                                    metric,
                                                    datapath)
        epochs = e_prev + epochs
        values = v_prev + values

    return values, epochs

def load_trainer(path_run_cfg: str,
                 path_dataset_cfg: str,
                 load_train: bool=False):
    """Initializes and returns a `Trainer` with the specified configuration.
    
    Args:
        path_run_cfg (str): Path to the run configuration file
        path_dataset_cfg (str): Path to the datasets configuration file
        load_train (bool, optional): Whether to load the training data
    
    Returns:
        The Trainer
    
    Raises:
        ValueError: Description
    """
    # Get the configurations
    datasets_cfg = load_yml(path_dataset_cfg)
    run_cfg = load_yml(path_run_cfg)
    validation_cfg = run_cfg.get('validation', None)

    assert validation_cfg is not None, "No validation configuration found!"

    model_cfg = run_cfg['CountNet']
    trainer_cfg = run_cfg['Trainer']

    if not load_train:
        # Set 'loader_train' entry to 'None' such that the training data is not
        # loaded.
        trainer_cfg['loader_train'] = None

    if not 'validate_run' in trainer_cfg:
        raise ValueError("No tag found at 'Trainer.validate_run'! The tag "
                         "specifies the run to be validated.")

    trainer = initialize_trainer(trainer_cfg, model_cfg=model_cfg,
                                              dset_cfg=datasets_cfg)

    return trainer
