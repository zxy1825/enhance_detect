'''
Author                        : Zhang Xiuyu
Date                          : 2024-05-13 17:35:31
LastEditTime                  : 2024-05-13 17:56:10
LastEditors                   : Zhang Xiuyu
Description                   : 测试demo
'''

import torch
from torchvision import transforms
import torch.nn as nn
from models.common import DetectMultiBackend
from PIL import Image

# Assuming you have a path to the official YOLOv5 weights
weights_path = 'yolov5s.pt'
model = DetectMultiBackend()
temp_dict = torch.load(weights_path, map_location='cpu')
model.load_state_dict(temp_dict)
model.eval()

image_path = 'data/bus.jpg'
image = Image.open(image_path)
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])
input_img = transform(image).unsqueeze(0)  # 增加batch维度

# 进行推理
with torch.no_grad():
    output = model(input_img)
    # print(output)