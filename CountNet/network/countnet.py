"""Implements the CountNet neural network"""

import torch.nn as nn
import torch

# -----------------------------------------------------------------------------

class CountNet(nn.Module):
    """"""
    def __init__(self, *, train_on_count: bool):
        """Initializes a CountNet instance.
        
        Args:
            train_on_count (bool): Whether to train the network on the total
                count. If false, train on the density map directly.
        """
        super().__init__()

        # ... network core

        # NOTE Are we training the network on the total count or on a
        #      (previously generated) ground-truth density map?
        #      The `nn.Conv2d(channels, 1)` layer outputs the density map;
        #      In the case of the former, we need an additional pooling layer
        #      that sums over the whole density map.
        if not train_on_count:
            self.out_layer = nn.Conv2d(channels, 1, kernel_size=1)

        else:
            self.out_layer = nn.Sequential([nn.Conv2d(, 1, kernel_size=1),
                                            nn.AvgPool2d((height, width))])

    def forward(self, x):
        """Computes the forward path."""
        # ...

        x = self.out_layer(x)

        if isinstance(self.out_layer, nn.Sequential):
            x *= height * width

        return x