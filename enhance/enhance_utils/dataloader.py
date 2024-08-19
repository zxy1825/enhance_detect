#!/usr/bin/env python
# coding=utf-8
'''
FilePath     : /enh_yov5/enhance/enhance_utils/dataloader.py
Description  :  
Author       : Zhang Xiuyu
LastEditors  : Zhang Xiuyu
LastEditTime : 2024-08-19 14:33:41
'''
import logging
import numpy as np
import torch
from PIL import Image
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from enhance_utils.utils import load_image, GammaGenerator


class EnhanceDataset(Dataset):
    def __init__(self, images_dir: str, channel : int= 3):
        self.images_dir = Path(images_dir)

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        self.generator = GammaGenerator(channel)
        logging.info(f'Creating dataset with {len(self.ids)} examples')
    

    def __len__(self):
        return len(self.ids)


    @staticmethod
    def preprocess(img):
        img = img.transpose((2, 0, 1))
        if (img > 1).any():
            img = img / 255.0
        return img


    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        img = load_image(img_file[0])

        img = self.preprocess(img)

        enhanced_img = self.generator(img)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'gt': torch.as_tensor(enhanced_img.copy()).float().contiguous()
        }

