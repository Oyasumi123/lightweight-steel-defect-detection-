# coding:utf-8

import os
import shutil
import argparse

parser = argparse.ArgumentParser()
# 图片的train、val、test文件夹路径
parser.add_argument('--image_train_dir', default='F:\model\liyolonet\liyolo\my-data\GC\images\\train', type=str, help='train images directory')
parser.add_argument('--image_val_dir', default='F:\model\liyolonet\liyolo\my-data\GC\images\\val', type=str, help='validation images directory')
parser.add_argument('--image_test_dir', default='F:\model\liyolonet\liyolo\my-data\GC\images\\test', type=str, help='test images directory')
# 标签的原始文件夹路径
parser.add_argument('--label_source_dir', default='F:\model\liyolonet\liyolo\my-data\GC10-DET\labels', type=str, help='source labels directory')
# 输出的标签文件夹路径
parser.add_argument('--label_output_dir', default='F:\model\liyolonet\liyolo\my-data\GC\labels', type=str, help='output labels directory')
opt = parser.parse_args()

# 获取输入和输出路径
image_train_dir = opt.image_train_dir
image_val_dir = opt.image_val_dir
image_test_dir = opt.image_test_dir
label_source_dir = opt.label_source_dir
label_output_dir = opt.label_output_dir

# 创建输出标签文件夹
train_label_dir = os.path.join(label_output_dir, 'train')
val_label_dir = os.path.join(label_output_dir, 'val')
test_label_dir = os.path.join(label_output_dir, 'test')

os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# 获取各个图片文件夹中的图片文件名
def get_image_filenames(image_dir):
    image_files = os.listdir(image_dir)
    # 提取文件名（不带扩展名）
    image_names = [os.path.splitext(f)[0] for f in image_files if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    return image_names

train_image_names = get_image_filenames(image_train_dir)
val_image_names = get_image_filenames(image_val_dir)
test_image_names = get_image_filenames(image_test_dir)

# 将对应的标签文件复制到目标文件夹
def copy_labels(image_names, src_dir, dest_dir):
    for name in image_names:
        label_file = name + '.txt'  # 假设标签文件的扩展名是.txt
        src_path = os.path.join(src_dir, label_file)
        dest_path = os.path.join(dest_dir, label_file)
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
        else:
            print(f"Warning: Label file {label_file} not found for image {name}")

copy_labels(train_image_names, label_source_dir, train_label_dir)
copy_labels(val_image_names, label_source_dir, val_label_dir)
copy_labels(test_image_names, label_source_dir, test_label_dir)

print("Label split completed!")