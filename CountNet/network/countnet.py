"""Implements the CountNet neural network"""

import torch.nn as nn
import torch

# -----------------------------------------------------------------------------
# Utils

class Concatenate(nn.Module):
    """
    Concatenate input tensors along a specified dimension.
    """
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


class Inception(nn.Module):
    """A single inception module."""
    def __init__(self, f_in, f_out, dim=1):
        super().__init__()

        f_out = f_out//4

        self.conv1x1 = nn.Conv2d(f_in, f_out, kernel_size=1)
        self.conv3x3 = nn.Sequential(
                nn.Conv2d(f_in, f_out, kernel_size=1),
                nn.Conv2d(f_out, f_out, kernel_size=3, padding=1)
            )
        self.conv5x5 = nn.Sequential(
                nn.Conv2d(f_in, f_out, kernel_size=1),
                nn.Conv2d(f_out, f_out, kernel_size=3, padding=1),
                nn.Conv2d(f_out, f_out, kernel_size=3, padding=1)
            )
        self.pool = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, padding=1, stride=1),
                nn.Conv2d(f_in, f_out, kernel_size=1)
            )
        self.dim = dim

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        x4 = self.pool(x)
        return torch.cat([x1, x2, x3, x4], dim=self.dim)

# -----------------------------------------------------------------------------

# Adapted from https://github.com/imagirom/ConfNets
class EncoderDecoderSkeleton(nn.Module):
    """Base class for Networks with Encoder Decoder Structure, such as UNet.
    """
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        # construct all the layers
        self.encoder_modules = nn.ModuleList(
            [self.construct_encoder_module(i) for i in range(depth)])
        self.skip_modules = nn.ModuleList(
            [self.construct_skip_module(i) for i in range(depth)])
        self.downsampling_modules = nn.ModuleList(
            [self.construct_downsampling_module(i) for i in range(depth)])
        self.upsampling_modules = nn.ModuleList(
            [self.construct_upsampling_module(i) for i in range(depth)])
        self.decoder_modules = nn.ModuleList(
            [self.construct_decoder_module(i) for i in range(depth)])
        self.merge_modules = nn.ModuleList(
            [self.construct_merge_module(i) for i in range(depth)])
        self.base_module = self.construct_base_module()
        self.final_module = self.construct_output_module()

    def forward(self, x):
        """Computes the forward path."""
        encoded_states = []
        for encode, downsample in zip(self.encoder_modules,
                                      self.downsampling_modules):
            x = encode(x)
            encoded_states.append(x)
            x = downsample(x)

        x = self.base_module(x)
        for encoded_state, upsample, skip, merge, decode in reversed(list(zip(
                                                    encoded_states,
                                                    self.upsampling_modules,
                                                    self.skip_modules,
                                                    self.merge_modules,
                                                    self.decoder_modules))
        ):
            x = upsample(x)
            encoded_state = skip(encoded_state)
            x = merge(x, encoded_state)
            x = decode(x)

        x = self.final_module(x)
        return x

    def construct_encoder_module(self, depth):
        return nn.Identity()

    def construct_decoder_module(self, depth):
        return self.construct_encoder_module(depth)

    def construct_downsampling_module(self, depth):
        return nn.Identity()

    def construct_upsampling_module(self, depth):
        return nn.Identity()

    def construct_skip_module(self, depth):
        return nn.Identity()

    def construct_merge_module(self, depth):
        return Concatenate()

    def construct_base_module(self):
        return nn.Identity()

    def construct_output_module(self):
        return nn.Identity()


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
                nn.Conv2d(f_in, f_out, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(f_out, f_out, kernel_size=3, padding=1),
                nn.ReLU()
            )
        return encoder

    # Simple convolution decoder, similar to encoder
    # def construct_decoder_module(self, depth):
    #     f_in = self.merged_fmaps[depth]
    #     f_out = self.out_channels if depth==0 else self.fmaps[depth-1]
    #     decoder = nn.Sequential(
    #             nn.Conv2d(f_in, f_out, kernel_size=3, padding=1),
    #             nn.ReLU(),
    #             nn.Conv2d(f_out, f_out, kernel_size=3, padding=1),
    #             nn.ReLU()
    #         )
    #     return decoder

    # Inception decoder
    def construct_decoder_module(self, depth):
        f_in = self.merged_fmaps[depth]
        f_out = self.fmaps[0] if depth==0 else self.fmaps[depth-1]
        decoder = nn.Sequential(
                Inception(f_in, f_out),
                nn.ReLU()
            )
        return decoder 

    def construct_downsampling_module(self, depth):
        return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def construct_upsampling_module(self, depth):
        return nn.Upsample(scale_factor=2)

    def construct_skip_module(self, depth):
        return nn.Identity()

    def construct_merge_module(self, depth):
        return Concatenate()

    def construct_base_module(self):
        f = self.fmaps[self.depth-1]
        base = nn.Sequential(
                nn.Conv2d(f, f, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(f, f, kernel_size=3, padding=1),
                nn.ReLU()
            )
        return base

    def construct_output_module(self):
        if self.final_activation is not None:
            return nn.Sequential(
                    nn.Conv2d(self.fmaps[0], 1, kernel_size=1),
                    self.final_activation()
                )
        return nn.Conv2d(self.fmaps[0], 1, kernel_size=1)
