#!/usr/bin/env python
# coding=utf-8
'''
FilePath     : /enh_yov5/mnt/work/code/enh_yov5/enhance/enhance_utils/utils.py
Description  : 功能函数
Author       : Zhang Xiuyu
LastEditors  : Zhang Xiuyu
LastEditTime : 2024-08-19 16:32:02
'''

import logging
import cv2
import math
import numpy as np
from PIL import Image
from os.path import splitext, isfile, join
import torch


'''
description: 从文件中读取image数据, RGB & HWC
param filename 文件路径
return Image格式图像数据
'''
def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return np.load(filename)
    elif ext in ['.pt', '.pth']:
        return torch.load(filename).permute(1, 2, 0).cpu().numpy()
    else:
        image = cv2.imread(filename)
        return np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

class GammaGenerator:
    def __init__(self, channel = 3):
        self.channel = channel
    def map_intension_to_gamma(self, intensity):
        return 1 / math.exp(intensity - 0.5)

    def inverse_gamma_transform(self, image, gamma):
        return cv2.pow(image, 1/gamma)

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            image = image.permute(1, 2, 0).cpu().numpy()  # 转换为 HWC 格式
            image = image.astype(np.float64)
        elif isinstance(image, Image.Image):
            image = np.array(image).astype(np.float64)
        elif isinstance(image, np.ndarray):
            image = image.astype(np.float64)
        else:
            logging.error(f'Cannot tell input form in gamma generator, please recheck your dataset')
            return None
        
        if not np.all((image >= 0) & (image <= 1)):
            image /= 255
        image = np.transpose(image, (1, 2, 0)).astype(np.float32)
        assert image.shape[2] == self.channel, 'input channel is not equal with expection'
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        average_value = np.mean(gray_image)
        gamma = self.map_intension_to_gamma(average_value)
        transformed_image = self.inverse_gamma_transform(image, gamma)
        return np.transpose(transformed_image, (2, 0, 1))
