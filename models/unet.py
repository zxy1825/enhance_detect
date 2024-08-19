#!/usr/bin/env python
# coding=utf-8
'''
FilePath     : /enh_yov5/models/unet.py
Description  : unet network class
Author       : Zhang Xiuyu
LastEditors  : Zhang Xiuyu
LastEditTime : 2024-08-19 16:25:50
'''

import torch.nn as nn
import torch.utils.checkpoint as cp

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
    

    def use_checkpointing(self):
        self.inc = cp.checkpoint(self.inc)
        self.down1 = cp.checkpoint(self.down1)
        self.down2 = cp.checkpoint(self.down2)
        self.down3 = cp.checkpoint(self.down3)
        self.down4 = cp.checkpoint(self.down4)
        self.up1 = cp.checkpoint(self.up1)
        self.up2 = cp.checkpoint(self.up2)
        self.up3 = cp.checkpoint(self.up3)
        self.up4 = cp.checkpoint(self.up4)
        self.outc = cp.checkpoint(self.outc)