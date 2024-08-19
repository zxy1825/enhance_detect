#!/usr/bin/env python
# coding=utf-8
'''
FilePath     : /enh_yov5/enhance/val.py
Description  :  
Author       : Zhang Xiuyu
LastEditors  : Zhang Xiuyu
LastEditTime : 2024-08-19 14:39:30
'''
import torch
import torch.nn.functional as F
from tqdm import tqdm

from enhance_utils.loss import calculate_psnr


@torch.inference_mode()
def evaluate(net, dataloader, device, mix_flag):
    net.eval()
    num_val_batches = len(dataloader)
    psnr = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=mix_flag):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, gt = batch['image'], batch['gt']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            gt = gt.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

            # predict the mask
            pred = net(image)
            psnr += calculate_psnr(pred, gt)

    net.train()
    return psnr / max(num_val_batches, 1)
