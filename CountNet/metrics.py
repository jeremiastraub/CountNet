"""Validation metrics"""
from itertools import product
import torch
import numpy
from numpy.lib.stride_tricks import as_strided as ast

# -----------------------------------------------------------------------------

def _error(x, y):
    """ Simple error """
    return x - y

# -----------------------------------------------------------------------------
"""Image-level metrics"""

def MAE(x, y):
    """Mean Absolute Error"""
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim == 4 and y.ndim == 4

    c_x = torch.sum(x, dim=[2,3])
    c_y = torch.sum(y, dim=[2,3])
    mae = torch.mean(torch.abs(c_x - c_y), dim=[0,1])
    return mae

def MSE(x, y):
    """Mean Squared Error"""
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim == 4 and y.ndim == 4

    c_x = torch.sum(x, dim=[2,3])
    c_y = torch.sum(y, dim=[2,3])
    mse = torch.mean(torch.square(c_x - c_y), dim=[0,1])
    return mse

def GAME(x, y, L=4):
    """Grid Average Mean absolute Error"""
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim == 4 and y.ndim == 4
    assert L >= 1

    patch_size = (x.shape[2] // L, x.shape[3] // L)
    c_x = torch.zeros((x.shape[0], x.shape[1], L, L))
    c_y = torch.zeros((x.shape[0], x.shape[1], L, L))

    for i, j in product(range(L), range(L)):
        c_x[...,i,j] = torch.sum(x[...,i*patch_size[0]:(i+1)*patch_size[0],
                                       j*patch_size[1]:(j+1)*patch_size[1]],
                                 dim=[2,3])
        c_y[...,i,j] = torch.sum(y[...,i*patch_size[0]:(i+1)*patch_size[0],
                                       j*patch_size[1]:(j+1)*patch_size[1]],
                                 dim=[2,3])

    game = torch.mean(torch.sum(torch.abs(c_x - c_y), dim=[2,3]))
    return game

def PMAE(x, y):
    raise NotImplementedError()

def PRMSE(x, y):
    raise NotImplementedError()

# -----------------------------------------------------------------------------
"""Pixel-level metrics
Tip: http://stackoverflow.com/a/5078155/1828289
"""

def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (A.shape[0] // block[0], A.shape[1] // block[1]) + block
    strides = (block[0] * A.strides[0], block[1] * A.strides[1]) + A.strides
    return ast(A, shape=shape, strides=strides)

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

def MSE_pixel(x, y):
    """Mean Squared Error"""
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim == 4 and y.ndim == 4

    # Calculate MSE for each image in the batch
    mse = torch.mean(torch.square(_error(x, y)))
    return mse

def MAE_pixel(x, y):
    """Mean Absolute Error"""
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim == 4 and y.ndim == 4

    # Calculate MSE for each image in the batch
    mae = torch.mean(torch.abs(_error(x, y)))
    return mae

def RMSE_pixel(x, y):
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
