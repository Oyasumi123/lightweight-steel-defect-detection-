# coding:utf-8

import os
import random
import shutil
import argparse

parser = argparse.ArgumentParser()
# 图片文件的根目录，图片按类别分文件夹
parser.add_argument('--image_root', default='F:\model\liyolonet\liyolo\my-data\huafen', type=str, help='input image root path')
# 输出的三个文件夹路径
parser.add_argument('--output_dir', default='F:\model\liyolonet\liyolo\my-data\GC1', type=str, help='output directory path')
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

# 遍历每个类别文件夹
for class_folder in class_folders:
    class_path = os.path.join(image_root, class_folder)
    image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    num = len(image_files)

    if num == 0:
        print(f"Warning: No images found in folder {class_folder}")
        continue

    # 打乱图片顺序
    random.shuffle(image_files)

    # 计算划分点
    train_end = int(num * train_percent)
    val_end = train_end + int(num * val_percent)

    # 划分训练集、验证集、测试集
    train_images = image_files[:train_end]
    val_images = image_files[train_end:val_end]
    test_images = image_files[val_end:]

    # 创建类别子文件夹
    class_train_dir = os.path.join(train_dir, class_folder)
    class_val_dir = os.path.join(val_dir, class_folder)
    class_test_dir = os.path.join(test_dir, class_folder)

    os.makedirs(class_train_dir, exist_ok=True)
    os.makedirs(class_val_dir, exist_ok=True)
    os.makedirs(class_test_dir, exist_ok=True)

    # 将图片复制到对应的文件夹
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(class_train_dir, img))

    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(class_val_dir, img))

    for img in test_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(class_test_dir, img))

print("Dataset split completed!")