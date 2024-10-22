"""Utility functions"""

import copy

import yaml
import torch
from torch.utils.data import DataLoader

from .data import (MallDataset, ShanghaiTechDataset, UCF_CC_50Dataset,
                    transforms)
from .network import CountNet
from .trainer import Trainer
import CountNet.metrics as metrics

# -----------------------------------------------------------------------------

def load_yml(path: str, *, mode: str='r'):
    """Loads a YAML file.

    Args:
        path (str): The path to the YAML file that should be loaded.
        mode (str, optional): Read mode

    Returns:
        The result of the data loading.
    """
    with open(path, mode) as yaml_file:
        return yaml.safe_load(yaml_file)

def initialize_dataset(name: str, cfg: dict, mode: str):
    """Creates a CrowdCountingDataset."""

    def initialize_transform(transform):
        """Creates a transform object."""
        if isinstance(transform, str):
            return getattr(transforms, transform)()

        if len(transform) > 1:
            raise ValueError("Only a single transform object can be "
                             f"passed (received: {len(transform)}. Use "
                             "'Compose' to aplly multiple transforms.")

        transform, kwargs = next(iter(transform.items()))
        if transform == 'Compose':
            t_list = []
            for t in kwargs:
                t_list.append(initialize_transform(t))
            return getattr(transforms, 'Compose')(t_list)

        else:
            return getattr(transforms, transform)(**kwargs)

    dset_kwargs = cfg[name]

    if name.startswith("Mall"):
        DatasetType = MallDataset

    elif name.startswith("ShanghaiTech"):
        DatasetType = ShanghaiTechDataset

    elif name.startswith("UCF_CC_50"):
        DatasetType = UCF_CC_50Dataset

    else:
        raise ValueError(f"Invalid Dataset Type: {name}")

    # Parse transforms
    for k in list(dset_kwargs.keys()):
        if 'transform' in k:
            dset_kwargs[k] = initialize_transform(dset_kwargs[k])

    return DatasetType(mode=mode, **dset_kwargs)

def initialize_data_loader(cfg: dict, dset_cfg: dict, mode: str):
    """Creates a DataLoader."""
    if cfg is None:
        return None
    
    dset_cfg = copy.deepcopy(dset_cfg)
    dataset = initialize_dataset(cfg.pop('dataset'), dset_cfg, mode)
    data_loader = DataLoader(dataset, **cfg)
    return data_loader

def initialize_trainer(cfg: dict, model_cfg: dict, dset_cfg: dict):
    """Creates a Trainer."""
    loader_train_cfg = cfg.pop('loader_train', None)
    loader_test_cfg = cfg.pop('loader_test', None)

    # Initialize the data loaders
    loader_train = initialize_data_loader(loader_train_cfg, dset_cfg,
                                          mode='train')
    loader_test = initialize_data_loader(loader_test_cfg, dset_cfg,
                                         mode='test')

    # Initialize the model
    model = CountNet(**model_cfg)

    # Parse the optimizer
    optimizer = cfg['optimizer']
    if isinstance(optimizer, str):
        cfg['optimizer'] = getattr(torch.optim, optimizer)(model.parameters())

    else:
        assert len(optimizer) == 1
        optimizer, kwargs = next(iter(optimizer.items()))
        cfg['optimizer'] = getattr(torch.optim, optimizer)(model.parameters(),
                                                           **kwargs)

    # Parse the loss metric
    # TODO Allow using custom metrics from metrics.py (try-except block)
    loss_metric = cfg['loss_metric']
    if isinstance(loss_metric, str):
        cfg['loss_metric'] = getattr(torch.nn, loss_metric)()

    else:
        assert len(loss_metric) == 1
        loss_metric, kwargs = next(iter(loss_metric.items()))
        cfg['loss_metric'] = getattr(torch.nn, loss_metric)(**kwargs)

    trainer = Trainer(model=model,
                      loader_train=loader_train,
                      loader_test=loader_test, **cfg)

    return trainer

def parse_metrics(metric_names: list):
    """Returns parsed metric list"""
    metrics_list = []

    for m in metric_names:
        if isinstance(m, str):
            try:
                metrics_list.append(getattr(metrics, m))
            except:
                metrics_list.append(getattr(torch.nn, m)())

        else:
            assert len(m) == 1
            m, m_kwargs = next(iter(m.items()))
            metrics_list.append(getattr(torch.nn, m)(**m_kwargs))

    return metrics_list

def parse_training_kwargs(kwargs: dict):
    """Returns the parsed training kwargs"""
    kwargs = copy.deepcopy(kwargs)
    if 'validation_metrics' in kwargs:
        kwargs['validation_metrics'] = parse_metrics(
                                                kwargs['validation_metrics'])
    return kwargs

def parse_validation_kwargs(kwargs: dict):
    """Returns the parsed validation kwargs"""
    kwargs = copy.deepcopy(kwargs)
    kwargs['metrics'] = parse_metrics(kwargs['metrics'])
    return kwargs
