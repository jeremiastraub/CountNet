"""Validation metrics"""

import torch

# -----------------------------------------------------------------------------

def RMSE(x, y):
    """Root Mean Squared Error"""
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim==4 and y.ndim==4

    # Calculate RMSE for each image in the batch
    mse = torch.mean((x - y)**2, dim=[1,2,3])
    rmse = torch.sqrt(mse)

    # Average over the batch
    rmse = torch.mean(rmse, dim=0)
    return rmse

def GAME(x, y):
    """Grid Average Mean absolute Error"""
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim==4 and y.ndim==4 

def PSNR(x, y):
    """Peak Signal to Noise Ratio"""
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim==4 and y.ndim==4

    # Calculate PSNR for each image in the batch
    mse = torch.mean((x - y)**2, dim=[1, 2, 3])
    psnr = -10. * torch.log10(mse)

    # Average over the batch
    psnr = torch.mean(psnr, dim=0)
    return psnr

def SSIM(x, y):
    """Structural Similarity Index"""
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim==4 and y.ndim==4 
