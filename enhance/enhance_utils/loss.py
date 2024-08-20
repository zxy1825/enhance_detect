import torch
import torch.nn as nn
import numpy as np
from pytorch_msssim import MS_SSIM


def calculate_psnr(pred, gt):
    if pred.max() > 1:
        pred = pred.float() / 255.0
    if gt.max() > 1:
        gt = gt.float() / 255.0
    
    mse = torch.mean((gt - pred) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    return psnr

# 使用一种L1+MS-SSIM 混合loss当做增强模块的loss, 以平衡高频(MS-SSIM)和低频(L1), 论文来源：Loss Functions for Image Restoration with Neural Networks
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, channel=3, data_range=1.0, size_average=True):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ms_ssim_loss = MS_SSIM(data_range=data_range, size_average=size_average, channel=channel)
    
    def forward(self, pred, gt):
        l1_loss = self.l1_loss(pred, gt)
        ms_ssim_loss = self.ms_ssim_loss(pred, gt)
        loss = self.alpha * ms_ssim_loss + (1 - self.alpha) * l1_loss
        return loss