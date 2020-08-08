"""Implements the CountNet neural network"""

import torch.nn as nn
import torch

from .base import Concatenate, EncoderDecoderSkeleton, Inception

# -----------------------------------------------------------------------------

class CountNet(EncoderDecoderSkeleton):
    def __init__(self, depth,
                       in_channels,
                       out_channels,
                       fmaps,
                       final_activation=None):
        """Initializes a CountNet instance."""
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fmaps = fmaps
        self.merged_fmaps = [2*f for f in fmaps]
        self.final_activation = final_activation
        super().__init__(depth)

    def construct_encoder_module(self, depth):
        f_in = self.in_channels if depth==0 else self.fmaps[depth-1]
        f_out = self.fmaps[depth]
        encoder = nn.Sequential(
                nn.Conv2d(f_in, f_out, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(f_out, f_out, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(),
            )
        return encoder

    def construct_decoder_module(self, depth):
        f_in = self.merged_fmaps[depth]
        f_out = self.fmaps[0] if depth==0 else self.fmaps[depth-1]
        decoder = nn.Sequential(
                Inception(f_in, f_out, dilation=1),
                nn.ReLU(),
                Inception(f_out, f_out, dilation=2),
                nn.ReLU(),
                # nn.Conv2d(f_in, f_out, kernel_size=3, padding=1, padding_mode='reflect'),
                # nn.ReLU(),
                # nn.Conv2d(f_out, f_out, kernel_size=3, padding=1, padding_mode='reflect'),
                # nn.ReLU(),
            )
        return decoder 

    def construct_downsampling_module(self, depth):
        return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def construct_upsampling_module(self, depth):
        # if depth=3:
        #     f_in = 
        # f_in = self.fmaps[self.depth]
        # f_out = self.fmaps[self.depth-1]
        # return nn.ConvTranspose2d(f_in, f_out, kernel_size=2, stride=2)
        return nn.Upsample(scale_factor=2)

    def construct_skip_module(self, depth):
        return nn.Identity()

    def construct_merge_module(self, depth):
        return Concatenate()

    def construct_base_module(self):
        f = self.fmaps[self.depth-1]
        base = nn.Sequential(
                nn.Conv2d(f, f, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(f, f, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(),
            )
        return base

    def construct_output_module(self):
        if self.final_activation is not None:
            return nn.Sequential(
                    nn.Conv2d(self.fmaps[0], 1, kernel_size=1),
                    self.final_activation()
                )
        return nn.Conv2d(self.fmaps[0], 1, kernel_size=1)
