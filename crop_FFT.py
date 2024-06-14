import numpy as np
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
from FourierTransform import init_output_dir
def crop_and_calculate_means(file_path, shape, t, out_path=None):
    # 加载npy文件
    data = np.load(file_path)
    
    channels, height, width = data.shape
    center = (height // 2, width // 2)
    area_ratio = t
    
    def calculate_means(cropped_region, original_region):
        cropped_mean = np.mean(cropped_region)
        original_mean = np.mean(original_region)
        return cropped_mean, original_mean
    
    cropped_means = []
    original_means = []
    
    for c in range(channels):
        channel_data = data[c]
        
        if shape == 'circle':
            radius = int(np.sqrt(area_ratio) * (height / 2))
            mask = np.zeros((height, width), dtype=bool)
            Y, X = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
            mask[dist_from_center <= radius] = True
            
        elif shape == 'square':
            side_length = int(np.sqrt(area_ratio) * height)
            half_side = side_length // 2
            mask = np.zeros((height, width), dtype=bool)
            mask[center[0]-half_side:center[0]+half_side, center[1]-half_side:center[1]+half_side] = True
            
        elif shape == 'diamond':
            side_length = int(np.sqrt(area_ratio) * height)
            half_side = side_length // 2
            mask = np.zeros((height, width), dtype=bool)
            for i in range(-half_side, half_side):
                mask[center[0]+i, center[1]-half_side+abs(i):center[1]+half_side-abs(i)] = True
                
        cropped_region = channel_data[mask]
        original_region = channel_data[~mask]
        
        cropped_mean, original_mean = calculate_means(cropped_region, original_region)
        
        cropped_means.append(cropped_mean)
        original_means.append(original_mean)
        if out_path != None:
            # 保存裁剪区域图像
            plt.imshow(cropped_region, cmap='viridis')
            plt.title(f'Channel {c+1} {shape.capitalize()} Mask')
            plt.axis('off')
            plt.savefig(f'{out_path}/{shape}_{t}/channel_{c+1}.png')
            plt.close()
            # 保存原始图像
            plt.imshow(original_region, cmap='viridis')
            plt.title(f'Channel {c+1} Original')
            plt.axis('off')
            plt.savefig(f'{out_path}/original_{t}/channel_{c+1}_original.png')
            plt.close()
    return np.array(cropped_means), np.array(original_means)


def count_positive_negative(array):
    # 将数组展平成一维数组
    flat_array = array.flatten()
    
    # 计算正数的个数
    positive_count = np.sum(flat_array > 0)
    
    # 计算负数的个数
    negative_count = np.sum(flat_array < 0)
    
    return positive_count, negative_count


def calculate_difference_means_and_max(array1, array2):
    pc, nc = count_positive_negative(array1 - array2)
    difference = np.abs(array1 - array2)
    mean_difference = np.mean(difference)
    max_difference = np.max(difference)
    return pc, nc, mean_difference, max_difference


if __name__ == "__main__":
    for method in ['circle', 'diamond']: # 'square', 
        for t in [0.25, 0.5, 0.75, 0.99]:
            for stage in ['stage17', 'stage20', 'stage23']:
                coco_path = f'coco_result/npy/{stage}_fft.npy'
                coco_mean1, coco_mean2 = crop_and_calculate_means(coco_path, method, t)
                exdark_path = f'exdark_result/npy/{stage}_fft.npy'
                exdark_mean1, exdark_mean2 = crop_and_calculate_means(exdark_path, method, t)
                print(f"\n| method : {method} | ratio : {t} | stage : {stage} |")
                # print("| Datasets | High frequency | Low frenquency | Difference |")
                # print("|   COCO   | {:>14} | {:>13} | {:>10} |".format(np.mean(coco_mean1), np.mean(coco_mean2), np.abs(np.mean(coco_mean1)-np.mean(coco_mean2))))
                # print("|  ExDark  | {:>14} | {:>13} | {:>10} |\n".format(np.mean(coco_mean1), np.mean(coco_mean2), np.abs(np.mean(coco_mean1)-np.mean(coco_mean2))))
                high_pc, high_nc, high_diff_mean, high_diff_max = calculate_difference_means_and_max(coco_mean1, exdark_mean1)
                low_pc, low_nc, low_diff_mean, low_diff_max = calculate_difference_means_and_max(coco_mean2, exdark_mean2)
                print(f"|  Low frequency count : positive:{high_pc} | negative:{high_nc}")
                print(f"| High frequency count : positive:{low_pc} | negative:{low_nc}")
                print(f"|  Low frequency difference : mean:{high_diff_mean} | max:{high_diff_max}")
                print(f"| High frequency difference : mean:{low_diff_mean} | max:{low_diff_max}")