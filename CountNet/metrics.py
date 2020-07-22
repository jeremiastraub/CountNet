"""Validation metrics"""

import torch
import numpy
from numpy.lib.stride_tricks import as_strided as ast


# -----------------------------------------------------------------------------

def _error(x, y):
    """ Simple error """
    return x - y


# -----------------------------------------------------------------------------
"""Image-level metrics"""


def MSE(x, y):
    """Mean Squared Error"""
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim == 4 and y.ndim == 4

    # Calculate MSE for each image in the batch
    mse = torch.mean(torch.square(_error(x, y)))
    return mse


def MAE(x, y):
    """Mean Absolute Error"""
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim == 4 and y.ndim == 4

    # Calculate MSE for each image in the batch
    mae = torch.mean(torch.abs(_error(x, y)))
    return mae


def RMSE(x, y):
    """Root Mean Squared Error"""
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim == 4 and y.ndim == 4

    # Calculate RMSE for each image in the batch
    mse = torch.mean((_error(x, y)) ** 2, dim=[1, 2, 3])
    rmse = torch.sqrt(mse)

    # Average over the batch
    rmse = torch.mean(rmse, dim=0)
    return rmse


def GAME(x, y):
    """Grid Average Mean absolute Error"""
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim == 4 and y.ndim == 4


def PMAE(x, y):
    pass


def PRMSE(x, y):
    pass


# -----------------------------------------------------------------------------
"""Pixel-level metrics"""

"""Tip: http://stackoverflow.com/a/5078155/1828289"""


def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (A.shape[0] // block[0], A.shape[1] // block[1]) + block
    strides = (block[0] * A.strides[0], block[1] * A.strides[1]) + A.strides
    return ast(A, shape=shape, strides=strides)


def PSNR(x, y):
    """Peak Signal to Noise Ratio"""
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim == 4 and y.ndim == 4

    # Calculate PSNR for each image in the batch
    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    psnr = -10. * torch.log10(mse)

    # Average over the batch
    psnr = torch.mean(psnr, dim=0)
    return psnr


def SSIM(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    """Structural Similarity Index"""
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim == 4 and y.ndim == 4

    bimg1 = block_view(x, (11, 11))
    bimg2 = block_view(y, (11, 11))
    s1 = numpy.sum(bimg1, (-1, -2))
    s2 = numpy.sum(bimg2, (-1, -2))
    ss = numpy.sum(bimg1 * bimg1, (-1, -2)) + numpy.sum(bimg2 * bimg2, (-1, -2))
    s12 = numpy.sum(bimg1 * bimg2, (-1, -2))

    vari = ss - s1 * s1 - s2 * s2
    covar = s12 - s1 * s2

    ssim_map = (2 * s1 * s2 + C1) * (2 * covar + C2) / ((s1 * s1 + s2 * s2 + C1) * (vari + C2))
    return numpy.mean(ssim_map)


# -----------------------------------------------------------------------------
"""Point-level metrics"""


def APK(x, y, k=10):
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim == 4 and y.ndim == 4

    # Calculates the average precision at k.
    if len(y) > k:
        y = y[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(y):
        if p in x and p not in y[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not x:
        return 0.0

    return score / min(len(x), k)
