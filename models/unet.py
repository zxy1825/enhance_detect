#!/usr/bin/env python
# coding=utf-8
'''
FilePath     : /enhance_detect/models/unet.py
Description  : unet network class
Author       : Zhang Xiuyu
LastEditors  : Zhang Xiuyu
LastEditTime : 2024-07-23 00:27:37
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import DoubleConv,Up, Down, OutConv

class UNet(nn.Module):
    def __init__(self, n_channels=3, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, 8)
        self.down2 = Down(128, 256, 8)
        self.down3 = Down(256, 512, 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, 8)
        self.up1 = Up(1024, 512 // factor, 8, bilinear=bilinear)
        self.up2 = Up(512, 256 // factor, 8, bilinear=bilinear)
        self.up3 = Up(256, 128 // factor, 8, bilinear=bilinear)
        self.up4 = Up(128, 64, 8, bilinear=bilinear)
        self.outc = OutConv(64, self.n_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
