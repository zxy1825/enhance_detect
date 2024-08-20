#!/usr/bin/env python
# coding=utf-8
'''
FilePath     : /enh_yov5/enhance/infer.py
Description  : 
Author       : Zhang Xiuyu
LastEditors  : Zhang Xiuyu
LastEditTime : 2024-08-20 16:34:28
'''
import argparse
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from enhance.enhance_utils.dataloader import EnhanceDataset
from models.unet import UNet

def infer_img(net,
              img,
              device):
    net.eval()
    img = torch.from_numpy(EnhanceDataset.preprocess(img))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        # output = F.interpolate(output, (img.size[1], img.size[0]), mode='bilinear')
        # print(f'output size is {output.shape}')

    return output.numpy()[0]


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='filenames or folder of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='filenames or folder of output images', required=True)
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--channels', '-c', type=int, default=3, help='Number of classes')
    
    return parser.parse_args()


def generate_output_name(input_path, output_dir, i):
    if os.path.isdir(output_dir[0]):
        output_path = os.path.join(output_dir[0], os.path.basename(input_path))
    elif not output_dir:
        output_path = f'{os.path.splitext(input_path)[0]}_OUT.jpg'
    else:
        output_path = output_dir[i]
    return output_path


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if os.path.isdir(args.input[0]):
        file_names = os.listdir(args.input[0])
        in_files = [os.path.join(args.input[0], filename) for filename in file_names]

    else:
        try:
            in_files = []
            for path in args.input:
                in_files.append(path)
        except:
            pass
    

    net = UNet(n_channels=args.channels, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading model {args.model}')
    print(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)

    print('Model loaded!')

    for i, filename in tqdm(enumerate(in_files), desc='Inference images', unit='img'):
        # print(f'Predicting image {filename} ...')
        img = cv2.imread(filename)

        pred = infer_img(net=net,
                         img=img,
                         device=device)
        output_image_path = generate_output_name(filename, args.output, i)
        pred = np.transpose(pred, (1, 2, 0))

        # 如果需要，缩放到 [0, 255] 并转换为 uint8 类型
        pred = (pred * 255).astype(np.uint8)
        # 保存图像到本地
        cv2.imwrite(output_image_path, pred)
