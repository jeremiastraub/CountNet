""""""

import os
from datetime import datetime

import numpy as np

import torch

# -----------------------------------------------------------------------------

class Trainer(object):
    def __init__(self, *, model,
                          optimizer,
                          loss_metric,
                          loader_train,
                          loader_test,
                          output_path: str,
                          name_ext: str=None):
        """The Trainer takes care of training and validation
        
        Args:
            model (TYPE): Description
            optimizer (TYPE): Description
            loss_metric (TYPE): Description
            loader_train (TYPE): Description
            loader_test (TYPE): Description
            output_path (str): Description
            name_ext (str, optional): Description
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_metric = loss_metric
        self.loader_train = loader_train
        self.loader_test = loader_test

        # Create output directory
        name = datetime.now().strftime("%y%m%d-%H%M%S")
        if name_ext is not None:
            name += "-"+name_ext
        out_path = os.path.join(output_path, name)
        os.makedirs(out_path, exist_ok=True)


    def train_model(self, epochs=1):
        """Train the model.
        
        Args:
            epochs (int, optional): The number of epochs to train for
        """
        for e in range(epochs):
            for t, (img_input, target) in enumerate(self.loader_train):
                model.train()  # put model to training mode

                prediction = model(img_input)
                loss = self.loss_metric(prediction, target)

                # Reset gradients
                self.optimizer.zero_grad()

                # Backward propagation
                loss.backward()

                # Take one optimizer step
                self.optimizer.step()

                # TODO - Compute some metrics (?)
                #      - write output (write out model configuration, too)
                #      - provide some logging messages (use tqm-package)



    def test_model(self, calc_metrics: list):
        """
        Args:
            calc_metrics (list, optional): The metrics to compute
        
        Returns:
            array: metric scores evaluated on the test data
        """
        self.model.eval()

        # Dont't need the gradient information
        with torch.no_grad():
            # Accumulate the metric scores here and average across the data
            scores = np.zeros(len(calc_metrics))

            for t, (img_input, target) in enumerate(self.loader_test):

                prediction = self.model(img_input)
                loss = loss + self.loss_metric(prediction, target)

                for i, metric in enumerate(calc_metrics):
                    scores[i] += metric(prediction, target)
            
            scores /= t

        return scores
