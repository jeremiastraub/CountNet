"""Implements the Trainer class for training and validating the model."""

import os
from datetime import datetime

import numpy as np
from tqdm import tqdm

import torch

# -----------------------------------------------------------------------------

class Trainer(object):
    """The Trainer takes care of training and validation of a model."""
    def __init__(self, *, model,
                          optimizer,
                          loss_metric,
                          loader_train,
                          loader_test,
                          write_to: str=None,
                          name_ext: str=None):
        """
        Args:
            model: The model to train/validate
            optimizer: The (initialized) optimizer object
            loss_metric: The loss function that takes the predicted
                density-map and the target map and returns the loss.
            loader_train: The data loader for the training set
            loader_test: The data loader for the test/validation set
            write_to (str, optional): Path to the output directory. If it is
                None, no data is written.
            name_ext (str, optional): Suffix for output folder-name
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_metric = loss_metric
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.out_path = None

        # Create output directory
        if write_to is not None:
            name = datetime.now().strftime("%y%m%d-%H%M%S")
            if name_ext is not None:
                name += "-"+name_ext
            out_path = os.path.join(write_to, name)
            os.makedirs(out_path, exist_ok=True)
            self.out_path = out_path

    def train_model(self, epochs: int=1, write_every: int=None):
        """Train the model.
        
        Args:
            epochs (int, optional): The number of epochs to train for
            write_every (int, optional): If not None, store and return losses
                in equidistant intervals.
        
        Returns:
            array: Losses for different training iterations
        """
        losses = []
        
        e_count = 1
        for e in tqdm(range(epochs), desc=f"Training {epochs} Epoch(s)"):
            for t, (image, target) in enumerate(tqdm(self.loader_train,
                                                     leave=False,
                                                     desc=f"Epoch {e_count}")):
                self.model.train()

                prediction = self.model(image)
                loss = self.loss_metric(prediction, target)
                losses.append(loss)

                # Perform one optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # TODO - Compute some metrics (?)
                #      - write output (write out model configuration, too)

            e_count += 1

        return np.array(losses)

    def validate_model(self, metrics: list):
        """
        Args:
            metrics (list): List of metrics to compute
        
        Returns:
            array: metric scores evaluated on the test data
        """
        self.model.eval()

        # Dont't need the gradient information
        with torch.no_grad():
            # Accumulate the metric scores here and average across the data
            scores = np.zeros(len(metrics))

            for t, (image, target) in enumerate(self.loader_test):

                prediction = self.model(image)

                for i, metric in enumerate(metrics):
                    scores[i] += metric(prediction, target)
            
            scores /= t

        return scores
