'''
Author: gw00336465 gw00336465@ifyou.com
Date: 2024-05-09 13:53:00
LastEditors: gw00336465 gw00336465@ifyou.com
LastEditTime: 2024-05-10 15:43:51
FilePath: /UNet/models/yo_net.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from torch import nn

class CombinedModel(nn.Module):
    def __init__(self, enhance_model, detect_model):
        super(CombinedModel, self).__init__()
        self.enhance_model = enhance_model
        self.detect_model = detect_model

    def forward(self, images):
        enhanced_output = self.enhance_model(images)
        detection_results = self.detect_model(enhanced_output)
        return detection_results
