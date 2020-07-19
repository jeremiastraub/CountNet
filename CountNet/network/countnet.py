"""Implements the CountNet neural network"""

import torch.nn as nn
import torch

# -----------------------------------------------------------------------------

class CountNet(nn.Module):
    """"""
    def __init__(self):
        """Initializes a CountNet instance."""
        super().__init__()

        self.features = [32, 32, 32]
        self.layers = self._build_layers(self.features)

    def forward(self, x):
        """Computes the forward path."""
        x = self.layers(x)

        return x

    def _build_layers(self, features):
        """"""
        layers = []
        f_in = 3

        for f in features:
            layers.append(nn.Conv2d(f_in, f, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            f_in = f

        layers.append(nn.Conv2d(f_in, 1, kernel_size=1))

        return nn.Sequential(*layers)
