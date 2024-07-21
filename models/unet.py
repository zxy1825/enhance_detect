'''
Author: gw00336465 gw00336465@ifyou.com
Date: 2024-05-09 13:02:17
LastEditors: gw00336465 gw00336465@ifyou.com
LastEditTime: 2024-05-22 16:41:09
FilePath: /UNet/models/unet.py
Description:unet网络结构实现
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import DoubleConv,Encoder, Decoder, OutConv

class UNet(nn.Module):
    def __init__(self, n_channels=3, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        # number_heads怎么取呢？是否需要实验验证？
        self.down1 = Encoder(n_channels, 64, 8)
        self.down2 = Encoder(64, 128, 16)
        self.down3 = Encoder(128, 256, 32)
        self.doubleconv = DoubleConv(256, 512)
        factor = 2 if bilinear else 1
        self.up1 = Decoder(512, 256 // factor, 32, bilinear=bilinear)
        self.up2 = Decoder(256, 128 // factor, 16, bilinear=bilinear)
        self.up3 = Decoder(128, 64 // factor, 8, bilinear=bilinear)
        self.outc = OutConv(64, self.n_channels)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        print(x3.size())
        x = self.doubleconv(x3)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits
