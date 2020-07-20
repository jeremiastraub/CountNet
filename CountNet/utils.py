"""Utility functions"""

import yaml
import torch
from torch.utils.data import DataLoader

from ..data import CrowdCountingDataset, transforms
from ..network import CountNet, Trainer

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

def initialize_dataset(name: Union[str, dict], cfg: dict):
    """Creates a CrowdCountingDataset."""
    def initialize_transform(transform):
        """Creates a transform object."""
        if transform is isinstance(str):
            return getattr(transforms, transform)()

        if len(transform) > 1:
            raise ValueError("Only a single transform object can be "
                             f"passed (received: {len(transform)}. Use "
                             "'Concatenate' to aplly multiple transforms.")

        transform, kwargs = next(iter(transform.items()))
        if transform == 'Concatenate':
            t_list = []
            for t in kwargs:
                t_list.append(initialize_transform(t))
            return getattr(transforms, 'Concatenate')(t_list)

        else:
            return getattr(transforms, transform)(**kwargs)

    dset_kwargs = cfg[name]

    # Parse transforms
    for k in list(dset_kwargs.keys()):
        if 'transform' in k:
            dset_kwargs[k] = initialize_transform(dset_kwargs[k])

    return CrowdCountingDataset(**dset_kwargs)

def initialize_data_loader(cfg: dict, dset_cfg: dict):
    """Creates a DataLoader."""
    dataset = initialize_dataset(cfg.pop('dataset'), dset_cfg)
    data_loader = DataLoader(dataset, **cfg)
    return data_loader

def initialize_trainer(cfg: dict, dset_cfg: dict):
    """Creates a Trainer."""
    loader_train_cfg = cfg.pop('loader_train')
    loader_test_cfg = cfg.pop('loader_test')

    # Initialize the data loaders
    loader_train = parse_data_loader(loader_train_cfg, dset_cfg)
    loader_test = parse_data_loader(loader_test_cfg, dset_cfg)

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
