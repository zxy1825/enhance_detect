#!/usr/bin/env python
# coding=utf-8
'''
FilePath     : /enhance_detect/demo.py
Description  : demo python file
Author       : Zhang Xiuyu
LastEditors  : Zhang Xiuyu
LastEditTime : 2024-07-23 00:12:10
'''
from models.unet import UNet
import torch
a = UNet(3, bilinear=True)
a(torch.zeros(1, 3, 640, 640))