#!/usr/bin/env python
# coding=utf-8
'''
FilePath     : /enh_yov5/enhance/inverse_gamma.py
Description  : 
Author       : Zhang Xiuyu
LastEditors  : Zhang Xiuyu
LastEditTime : 2024-08-14 10:20:54
'''

import cv2
import math
import numpy as np

class GammaGenerator:
    def map_intension_to_gamma(intensity):
        return 1 / math.exp((intensity - 127) / 128)

    def inverse_gamma_transform(image, gamma):
        return cv2.pow(image/255.0, 1/gamma) * 255

    def __call__(self, image):
        if image is not None:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            average_value = np.mean(gray_image)
            gamma = self.map_intension_to_gamma(average_value)
            transformed_image = self.inverse_gamma_transform(image, gamma)
        return transformed_image
