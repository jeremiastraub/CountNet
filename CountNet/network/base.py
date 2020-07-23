"""Base classes and utils for the CountNet"""

import torch.nn as nn
import torch

# -----------------------------------------------------------------------------

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
    def __init__(self, f_in, f_out, dim=1, dilation=1):
        super().__init__()
        
        f_out = f_out//4

        self.branch1x1 = nn.Conv2d(f_in, f_out, kernel_size=1)
        self.branch3x3 = nn.Sequential(
                nn.Conv2d(f_in, f_out, kernel_size=1, padding_mode='reflect'),
                nn.Conv2d(f_out, f_out, kernel_size=3,
                                        padding=dilation,
                                        dilation=dilation,
                                        padding_mode='reflect')
            )
        self.branch5x5 = nn.Sequential(
                nn.Conv2d(f_in, f_out, kernel_size=1, padding_mode='reflect'),
                nn.Conv2d(f_out, f_out, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.Conv2d(f_out, f_out, kernel_size=3,
                                        padding=dilation,
                                        dilation=dilation,
                                        padding_mode='reflect')
            )
        self.branch7x7 = nn.Sequential(
                nn.Conv2d(f_in, f_out, kernel_size=1, padding_mode='reflect'),
                nn.Conv2d(f_out, f_out, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.Conv2d(f_out, f_out, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.Conv2d(f_out, f_out, kernel_size=3,
                                        padding=dilation,
                                        dilation=dilation,
                                        padding_mode='reflect')
            )
        self.dim = dim

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        return torch.cat([branch1x1, branch3x3,
                          branch5x5, branch7x7], dim=self.dim)


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
