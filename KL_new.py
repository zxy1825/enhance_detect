import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import pandas as pd
from tabulate import tabulate
from PIL import Image

category_list = ['bike', 'boat', 'bottle', 'bus',
                 'car', 'cat', 'chair', 'cup',
                 'dog', 'motorbike', 'people', 'table']
layer_dic = ['stage0_Conv_features.npy', 'stage1_Conv_features.npy', 'stage2_C3_features.npy',
             'stage3_Conv_features.npy', 'stage4_C3_features.npy', 'stage5_Conv_features.npy', 'stage6_C3_features.npy',
             'stage7_Conv_features.npy', 'stage8_C3_features.npy', 'stage9_SPPF_features.npy', 'stage10_Conv_features.npy',
             'stage11_Upsample_features.npy', 'stage12_Concat_features.npy', 'stage13_C3_features.npy', 'stage14_Conv_features.npy',
             'stage15_Upsample_features.npy', 'stage16_Concat_features.npy', 'stage17_C3_features.npy', 'stage18_Conv_features.npy',
             'stage19_Concat_features.npy', 'stage20_C3_features.npy', 'stage21_Conv_features.npy', 'stage22_Concat_features.npy', 'stage23_C3_features.npy'
             ]

class FeatureMapFile:
    def __init__(self, file_path, bins, range):
        self.file_path = file_path
        self.bins = bins
        self.range = range
        self.hist_list = []
        self.bins_list = []
        self.prob_list = []
        self.analyze_args()
        self.calculate_hist()
        self.convert_hist_to_prob()
    
    def analyze_args(self):
        self.array = np.load(self.file_path)
        self.channels = self.array.shape[0]
        results = self.file_path.split('/')
        self.filename = results[-1]
        self.image = results[-2]
        self.method = results[-3]
        self.stage = ''.join([char for char in self.filename if char.isdigit()])

    # 输入numpy数组，计算通道直方图
    def calculate_hist(self):
        for channel in range(self.channels):
        # 计算每个通道的直方图
            hist, bins = np.histogram(self.array[channel, :, :], bins=self.bins, range=self.range)
            self.hist_list.append(hist)
            self.bins_list.append(bins)
    
    # 将直方图分布转换成概率分布
    def convert_hist_to_prob(self):
        for hist in self.hist_list:
            total = sum(hist)
            if total == 0:  # 避免除以0的错误
                prob = [0] * len(hist)
            else:
                prob = [count / total for count in hist]
            self.prob_list.append(prob)

class FeatureMapGroup:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.stage17_maps = self.build_data('stage17')
        self.stage20_maps = self.build_data('stage20')
        self.stage23_maps = self.build_data('stage23')
        self.stage17_prob_list = []
        self.stage20_prob_list = []
        self.stage23_prob_list = []
        for map in self.stage17_maps:
            self.stage17_prob_list.append(map.prob_list)
        self.stage17_prob_list = column_means(self.stage17_prob_list)
        
        for map in self.stage20_maps:
            self.stage20_prob_list.append(map.prob_list)
        self.stage20_prob_list = column_means(self.stage20_prob_list)

        for map in self.stage23_maps:
            self.stage23_prob_list.append(map.prob_list)
        self.stage23_prob_list = column_means(self.stage23_prob_list)
    
    def build_data(self, keyword):
        path = os.path.join('runs/detect', self.data + '_' + self.model, keyword)
        files = list_direct_files(path)
        feature_map_list = []
        for file in files:
            array = np.load(file)
            feature_map_list.append(FeatureMapFile(array, self.model, '', ''))
        return feature_map_list

def column_means(matrix):
    # 将列表转换为 NumPy 数组
    np_matrix = np.array(matrix)
    # 计算每列的均值
    means = np.mean(np_matrix, axis=0)
    return means.tolist()  # 返回均值列表

def list_direct_files(directory):
    # 获取指定目录下的所有条目
    entries = os.listdir(directory)
    # 过滤出文件，忽略子目录，并直接构建完整路径
    files = [os.path.join(directory, file) for file in entries if os.path.isfile(os.path.join(directory, file))]
    return files
    
def get_single_info(category, layer, threshold):
    map_list = []
    methods = ['Light', 'Dark']
    for method in methods:
        input_path = os.path.join('runs/detect', method,
                                  category, layer+'_C3_features.npy')
        image = np.load(input_path)
        map_list.append(FeatureMapFile(image, method, category, layer))
    for i in range(1): #len(map_list)-1
        for j in range(i+1, len(map_list)):
            count = 0
            kl_list = kl_divergence_2d(map_list[i], map_list[j])
            for channel, kl in enumerate(kl_list):
                if kl > threshold:
                    # plot_feature_map(map_list[i], map_list[j], channel, kl)
                    count += 1
            print(f'KL divergence over:{threshold} between {methods[i]} and {methods[j]} on layer:{layer} and category:{category} counts {count} ')
            labels = list(range(len(kl_list)))
            plt.bar(labels, kl_list, color='blue')
            plt.title(f'KL divergence between {methods[i]} and {methods[j]} on layer:{layer} and category:{category}')
            plt.show()

# 绘图
def plot_feature_map(map_i, map_j, channel, kl):
    
    feature_map1 = map_i.image[channel, :, :]
    feature_map2 = map_j.image[channel, :, :]

    plt.figure(figsize=(16, 9))

    plt.subplot(2, 3, 1)
    plt.bar(map_i.bins_list[channel][:-1], map_i.hist_list[channel], width=0.5, color='grey', alpha=0.75)
    plt.title(f'Channel {channel}')

    plt.subplot(2, 3, 2)
    plt.imshow(feature_map1, cmap='viridis', interpolation='none')  # 使用 'viridis' 颜色映射
    plt.colorbar()  # 添加颜色条以便查看对应的数值
    plt.title(f'{map_i.method} feature image')

    plt.subplot(2, 3, 3)
    origin_image1 = Image.open(map_i.image_path)
    plt.imshow(origin_image1)
    plt.title(f'{map_i.method} detect result')

    plt.subplot(2, 3, 4)
    plt.bar(map_j.bins_list[channel][:-1], map_j.hist_list[channel], width=0.5, color='grey', alpha=0.75)
    plt.title(f'Channel {channel}')

    plt.subplot(2, 3, 5)
    plt.imshow(feature_map2, cmap='viridis', interpolation='none')  # 使用 'viridis' 颜色映射
    plt.colorbar()  # 添加颜色条以便查看对应的数值
    plt.title(f'{map_j.method} feature image')

    plt.subplot(2, 3, 6)
    origin_image2 = Image.open(map_j.image_path)
    plt.imshow(origin_image2)
    plt.title(f'{map_j.method} detect result')

    plt.suptitle(f'category:{map_i.category}    layer:{map_i.layer}    channel:{channel}    KL divergence:{kl}')

    plt.show()

# 计算两张特征图之间的KL散度
def kl_divergence_2d(map_P, map_Q):
    kl_list = []
    
    for i in range(len(map_P.prob_list)):
        layer_p, layer_q = map_P.prob_list[i], map_Q.prob_list[i]
        # 确保除数非零
        epsilon = 1e-10
        # 对q加一个小值
        layer_q = np.maximum(layer_q, epsilon)
        # 计算P中所有概率不为0的KL散度
        kl_div = np.where(layer_p != 0, layer_p * np.log(layer_p / layer_q + epsilon), 0)
        kl_list.append(np.sum(kl_div))
    return kl_list

def kl_divergence(prob_P, prob_Q):
    kl_list = []
    
    for i in range(len(prob_P)):
        layer_p, layer_q = prob_P[i], prob_Q[i]
        # 确保除数非零
        epsilon = 1e-10
        # 对q加一个小值
        layer_q = np.maximum(layer_q, epsilon)
        # 计算P中所有概率不为0的KL散度
        kl_div = np.where(layer_p != 0, layer_p * np.log(layer_p / layer_q + epsilon), 0)
        kl_list.append(np.sum(kl_div))
    return kl_list

# 计算两组特征图之间的KL散度并绘图
def kl_divergence_3d(group_P, group_Q):
    labels = list(range(len(column_means(group_P.stage17_prob_list))))
    plt.subplot(2, 3, 1)
    plt.bar(labels, column_means(group_P.stage17_prob_list), color='blue')
    plt.title('COCO pretrained histogram, stage 17')

    labels = list(range(len(column_means(group_P.stage20_prob_list))))
    plt.subplot(2, 3, 2)
    plt.bar(labels, column_means(group_P.stage20_prob_list), color='blue')
    plt.title('COCO pretrained histogram, stage 20')

    labels = list(range(len(column_means(group_P.stage23_prob_list))))
    plt.subplot(2, 3, 3)
    plt.bar(labels, column_means(group_P.stage23_prob_list), color='blue')
    plt.title('COCO pretrained histogram, stage 23')

    labels = list(range(len(column_means(group_Q.stage17_prob_list))))
    plt.subplot(2, 3, 4)
    plt.bar(labels, column_means(group_Q.stage17_prob_list), color='red')
    plt.title('ExDark pretrained histogram, stage 17')

    labels = list(range(len(column_means(group_Q.stage20_prob_list))))
    plt.subplot(2, 3, 5)
    plt.bar(labels, column_means(group_Q.stage20_prob_list), color='red')
    plt.title('ExDark pretrained histogram, stage 20')

    labels = list(range(len(column_means(group_Q.stage23_prob_list))))
    plt.subplot(2, 3, 6)
    plt.bar(labels, column_means(group_Q.stage23_prob_list), color='red')
    plt.title('ExDark pretrained histogram, stage 23')

    plt.show()
    kl_stage17 = kl_divergence(group_P.stage17_prob_list, group_Q.stage17_prob_list)
    kl_stage20 = kl_divergence(group_P.stage20_prob_list, group_Q.stage20_prob_list)
    kl_stage23 = kl_divergence(group_P.stage23_prob_list, group_Q.stage23_prob_list)
    labels = list(range(len(kl_stage17)))
    plt.subplot(1, 3, 1)
    plt.bar(labels, kl_stage17, color='red')
    plt.title(f'KL divergence between COCO and ExDark on layer:17')
    labels = list(range(len(kl_stage20)))
    plt.subplot(1, 3, 2)
    plt.bar(labels, kl_stage20, color='green')
    plt.title(f'KL divergence between COCO and ExDark on layer:20')
    labels = list(range(len(kl_stage23)))
    plt.subplot(1, 3, 3)
    plt.bar(labels, kl_stage23, color='blue')
    plt.title(f'KL divergence between COCO and ExDark on layer:23')
    plt.show()
    print(f'stage 17 mean kl divergence is {column_means(kl_stage17)}, max is {np.max(kl_stage17)}')
    print(f'stage 20 mean kl divergence is {column_means(kl_stage20)}, max is {np.max(kl_stage20)}')
    print(f'stage 23 mean kl divergence is {column_means(kl_stage23)}, max is {np.max(kl_stage23)}')

def evaluate_single(threshold):
    for category in category_list:
        for layer in ['stage17', 'stage20', 'stage23']:
            get_single_info(category, layer, threshold)

def evaluate_group(threshold):
    coco_group = FeatureMapGroup('Pretrained', 'COCO')
    exdark_group = FeatureMapGroup('Pretrained', 'Source')
    kl_divergence_3d(coco_group, exdark_group)
    
def visual_feature_peak_value(root_path):

    min_2d_list = []
    max_2d_list = []
    for sub_dir in tqdm(os.listdir(root_path)):
        max_list = []
        min_list = []
        for i in range(24):
            array = np.load(os.path.join(root_path, sub_dir, layer_dic[i]))
            max_list.append(np.max(array))
            min_list.append(np.min(array))
        max_2d_list.append(max_list)
        min_2d_list.append(min_list)
    max_2d_list = np.array(max_2d_list)
    min_2d_list = np.array(min_2d_list)
    for i in range(24):
        print(f'Max stage{i} Feature Map Value is {np.max(max_2d_list[:, i])}')
    
    for i in range(24):
        print(f'Min stage{i} Feature Map Value is {np.min(min_2d_list[:, i])}')

def main():
    root_path = '/media/sugar/71621f35-5159-4d8b-837a-18d01bacf880/detect/exdark'
    visual_feature_peak_value(root_path)

if __name__ == '__main__':
    main()