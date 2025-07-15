# coding:utf-8

import os
import random
import shutil
import argparse

parser = argparse.ArgumentParser()
# 图片文件的根目录，图片按类别分文件夹
parser.add_argument('--image_root', default='F:\model\liyolonet\liyolo\my-data\huafen', type=str, help='input image root path')
# 输出的三个文件夹路径
parser.add_argument('--output_dir', default='F:\model\liyolonet\liyolo\my-data\gc10\images', type=str, help='output directory path')
opt = parser.parse_args()

# 数据集划分比例
train_percent = 0.8
val_percent = 0.1
test_percent = 0.1

# 获取输入和输出路径
image_root = opt.image_root
output_dir = opt.output_dir

# 创建输出文件夹
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 获取所有类别文件夹
class_folders = [d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))]

# 初始化图片列表
all_images = []

# 收集所有图片
for class_folder in class_folders:
    class_path = os.path.join(image_root, class_folder)
    image_files = [os.path.join(class_folder, f) for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    all_images.extend(image_files)

num = len(all_images)

if num == 0:
    print("Error: No images found in the image root directory!")
    exit()

# 打乱图片顺序
random.shuffle(all_images)

# 计算划分点
train_end = int(num * train_percent)
val_end = train_end + int(num * val_percent)

# 划分图片到训练集、验证集、测试集
train_images = all_images[:train_end]
val_images = all_images[train_end:val_end]
test_images = all_images[val_end:]

# 定义一个函数来复制图片
def copy_images(img_list, target_dir):
    for img_path in img_list:
        # 解析图片路径
        class_folder, img_name = os.path.split(img_path)
        original_path = os.path.join(image_root, img_path)
        # 复制图片到目标文件夹
        shutil.copy(original_path, os.path.join(target_dir, img_name))

# 复制图片到对应的文件夹
copy_images(train_images, train_dir)
copy_images(val_images, val_dir)
copy_images(test_images, test_dir)

print("Dataset split completed!")