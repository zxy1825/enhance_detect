'''
Author: gw00336465 gw00336465@ifyou.com
Date: 2024-04-17 17:54:47
LastEditors: gw00336465 gw00336465@ifyou.com
LastEditTime: 2024-04-23 17:42:21
FilePath: /YOLOv5/execulate.py
Description: 用于评估单张/两张图片之间的性能指标的模块函数
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

def calculate_mean_fft(directory, stages):
    stage_data = {stage: [] for stage in stages}
    
    # 遍历每个子文件夹
    for subdir, dirs, files in os.walk(directory):
        # 查找指定的stage文件
        for stage in stages:
            filename = f"{stage}_C3_features.npy"
            filepath = os.path.join(subdir, filename)
            
            if os.path.isfile(filepath):
                data = np.load(filepath)
                stage_data[stage].append(data)
    
    results = {}
    
    for stage, data_list in tqdm(stage_data.items(), desc='Resaving data'):
        if data_list:
            fft_results = []
            
            for data in data_list:
                # 对每个通道进行傅里叶变换
                fft_channels = []
                for channel in range(data.shape[0]):
                    channel_data = data[channel]
                    fft_result = scipy.fftpack.fft2(channel_data)
                    fft_magnitude = np.abs(fft_result)
                    fft_channels.append(fft_magnitude)
                
                fft_results.append(np.stack(fft_channels, axis=0))
            
            # 对所有傅里叶变换结果求均值
            all_fft_data = np.stack(fft_results, axis=0)
            mean_fft_data = np.mean(all_fft_data, axis=0)
            
            results[stage] = mean_fft_data
    
    return results

def save_fft_results(fft_results, output_dir="fft_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for stage, fft_data in tqdm(fft_results.items(), desc='Saving results'):
        num_channels = fft_data.shape[0]
        for channel in range(num_channels):
            plt.figure()
            plt.imshow(np.log(fft_data[channel] + 1), cmap='viridis')
            plt.colorbar()
            plt.title(f'FFT Magnitude for {stage} - Channel {channel}')
            plt.xlabel('Frequency')
            plt.ylabel('Frequency')

            # 保存图像而不是显示
            output_path = os.path.join(output_dir, f'{stage}_channel_{channel}.png')
            plt.savefig(output_path)
            plt.close()  # 关闭当前图像，释放内存


def calculate_ssim_between_datasets(fft_results1, fft_results2, stage):
    if stage not in fft_results1 or stage not in fft_results2:
        raise ValueError("Specified stage not found in one of the FFT results.")
    
    fft_data1 = fft_results1[stage][0]  # 仅取第一个通道
    fft_data2 = fft_results2[stage][0]  # 仅取第一个通道
    
    # 计算SSIM
    ssim_value = ssim(fft_data1, fft_data2, data_range=fft_data2.max() - fft_data2.min())
    return ssim_value


def main():
    # 设置文件夹路径和需要处理的stage文件名
    directory = "./dataset/coco"
    stages = ["stage17", "stage20", "stage23"]
    # 计算平均傅里叶变换频谱
    fft_results = calculate_mean_fft(directory, stages)
    # 绘制结果
    save_fft_results(fft_results, './coco_result')

    # 设置文件夹路径和需要处理的stage文件名
    directory = "./dataset/exdark"
    stages = ["stage17", "stage20", "stage23"]
    # 计算平均傅里叶变换频谱
    fft_results = calculate_mean_fft(directory, stages)
    # 绘制结果
    save_fft_results(fft_results, './exdark_result')


if __name__ == '__main__':
    main()