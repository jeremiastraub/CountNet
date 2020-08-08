"""Implements the Trainer class for training and validating the model."""

import os
from datetime import datetime

import yaml
import numpy as np
from tqdm import tqdm

import torch

# -----------------------------------------------------------------------------

class Trainer(object):
    """The Trainer takes care of training and validation of a model."""
    OUTPUT_PATH = "output"

    def __init__(self, *, model,
                          optimizer,
                          loss_metric,
                          loader_train,
                          loader_test,
                          validate_run: str=None,
                          load_from: str=None,
                          tag_ext: str=None,
                          save_checkpoint=True):
        """Initializes a Trainer.
        
        Args:
            model: The model to train/validate
            optimizer: The (initialized) optimizer object
            loss_metric: The loss function that takes the predicted
                density-map and the target map and returns the loss.
            loader_train: The data loader for the training set
            loader_test: The data loader for the test/validation set
            validate_run (str, optional): Tag of a previous to be validated.
                If given, *no* new output tag is created (`load_from` must
                match or be `None`).
            load_from (str, optional): Tag of a previous run to be loaded
            tag_ext (str, optional): Suffix for output folder tag
            save_checkpoint (bool, optional): Whether to save the current model
                and training configuration as a checkpoint
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_metric = loss_metric
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.save_checkpoint = save_checkpoint
        self.epoch = None

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            print("\nNo GPU available. Using CPU...")
            self.device = torch.device('cpu')

        # Get output directory
        if validate_run is not None:
            out_dir = os.path.join(self.OUTPUT_PATH, validate_run)

        else:
            tag = datetime.now().strftime("%y%m%d-%H%M%S")
            if tag_ext is not None:
                tag += "-" + tag_ext

            out_dir = os.path.join(self.OUTPUT_PATH, tag)
            os.makedirs(out_dir)

        self.out_dir = out_dir
        assert os.path.isdir(self.out_dir)

        # Assemble configuration dictionary which is written after training
        model_cfg = {
            'depth': self.model.depth,
            'fmaps': self.model.fmaps,
            'final_activation': self.model.final_activation
        }
        optimizer_cfg = {
            'name': type(self.optimizer).__name__,
            'learning_rate': self.optimizer.defaults['lr']
        }
        self.cfg = {
            'model': model_cfg,
            'optimizer': optimizer_cfg,
            'loss_metric': type(self.loss_metric).__name__,
            'load_from': load_from,
            'train_dataset': (type(self.loader_train.dataset).__name__
                              if self.loader_train is not None else None),
            'test_dataset': type(self.loader_test.dataset).__name__,
            'train_batch_size': (self.loader_train.batch_size
                                 if self.loader_train is not None else None),
            'test_batch_size': self.loader_test.batch_size
        }

        # Load checkpoint if needed
        if load_from is not None or validate_run is not None:
            if load_from is not None and validate_run is not None:
                assert load_from == validate_run

            load_path = (self.out_dir if validate_run is not None
                         else os.path.join(self.OUTPUT_PATH, load_from))

            if not os.path.splitext(load_path)[1]:
                load_path = os.path.join(load_path, "checkpoint.pt")

            if self.device == torch.device('cpu'):
                checkpoint = torch.load(load_path,
                                        map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(load_path)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.device == torch.device('cuda'):
                # NOTE Workaround if loading a checkpoint while using a GPU.
                #      In that case, the optimizer needs to be re-initialized
                #      after the model was moved to the 'cuda' device.
                self.model = self.model.to(device=self.device)
                lr = self.optimizer.defaults['lr']
                self.optimizer = type(self.optimizer)(
                                        params=self.model.parameters(), lr=lr)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            print(f"\nLoaded checkpoint at '{load_path}'.")

    def train_model(self, *, epochs: int=1,
                             write_every: int=None,
                             validate_every_epoch: bool=False,
                             validation_metrics: list=None):
        """Trains the model.
        
        Args:
            epochs (int, optional): The number of epochs to train for
            write_every (int, optional): If not None, store and return losses
                in equidistant intervals.
            validate_every_epoch (bool, optional): Whether to validate the
                model after every epoch.
            validation_metrics (list, optional): The metrics used for
                validation if validate_every_epoch=True.
        
        Returns:
            Tuple[list, dict]: Losses for different training iterations, dict
            of validation scores keyed by epoch (if validate_every_epoch=True).
        """
        self.model = self.model.to(device=self.device)
        validations = dict()
        losses = []

        if self.epoch is None:
            self.epoch = 1
        
        for e in tqdm(range(epochs), desc=f"Training {epochs} Epoch(s)"):
            for t, (image, target) in enumerate(tqdm(self.loader_train,
                                                     leave=False,
                                                     desc="Epoch "
                                                          f"{self.epoch}")):
                self.model.train()
                image = image.to(device=self.device)
                target = target.to(device=self.device)

                prediction = self.model(image)
                loss = self.loss_metric(prediction, target)

                if write_every is not None and t%write_every == 0:
                    losses.append(loss.item())

                # Perform one optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                del loss, prediction

            print("\nCurrent loss: ", losses[-1])

            if validate_every_epoch:
                validations[self.epoch] = self.validate_model(
                                                    metrics=validation_metrics)
            self.epoch += 1

        # Save checkpoint and the configurations
        self.cfg['write_every'] = write_every
        self.cfg['current_epoch'] = self.epoch - 1
        cfg_path = os.path.join(self.out_dir, 'configuration.yml')
        with open(cfg_path, 'w') as file:
            yaml.dump(self.cfg, file, default_flow_style=False)

        if self.save_checkpoint:
            checkpoint_path = os.path.join(self.out_dir, "checkpoint.pt")
            torch.save({
                'epoch': self.epoch,
                'losses': losses,
                'validations': validations,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
                }, checkpoint_path)

        return losses, validations

    def validate_model(self, metrics: list):
        """Validates the model.

        Args:
            metrics (list): List of metrics to compute
        
        Returns:
            dict: metric scores evaluated on the test data keyed by name
        """
        self.model = self.model.to(device=self.device)
        self.model.eval()
        # Dont't need the gradient information
        with torch.no_grad():
            # Accumulate the metric scores here and average across the data
            scores = np.zeros(len(metrics))

            for t, (image, target) in enumerate(tqdm(self.loader_test,
                                                     desc="Validation:")):
                image = image.to(device=self.device)
                target = target.to(device=self.device)
                
                prediction = self.model(image)

                for i, metric in enumerate(metrics):
                    scores[i] += metric(prediction, target).item()

                del prediction
            
            scores /= t
            validations = {(metric.__name__ if hasattr(metric, '__name__')
                            else type(metric).__name__): float(s)
                           for metric, s in zip(metrics, scores)}

        validation_folder_path = os.path.join(self.out_dir, "validation")
        os.makedirs(validation_folder_path, exist_ok=True)
        tag = datetime.now().strftime("%y%m%d-%H%M%S")
        validation_path = os.path.join(validation_folder_path, tag)
        validation_path += ".yml"

        validation_output = {k: v for k, v in validations.items()}
        validation_output['test_dataset'] = type(self.loader_test.dataset).__name__
        with open(validation_path, 'w') as file:
            yaml.dump(validation_output, file, default_flow_style=False)

        return validations
