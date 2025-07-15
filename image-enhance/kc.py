import time
import random
import copy
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from lxml import etree, objectify
import xml.etree.ElementTree as ET
import argparse

from torch.utils.tensorboard import SummaryWriter


# 显示图片
def show_pic(img, bboxes=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 3)
    cv2.namedWindow('pic', 0)  # 1表示原图
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200, 800)  # 可视化的图片大小
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 图像均为cv2读取
class DataAugmentForObjectDetection():
    def __init__(self, rotation_rate=0.5, max_rotation_angle=5,
                 crop_rate=0.5, shift_rate=0.5, change_light_rate=0.5,
                 add_noise_rate=0.5, flip_rate=0.5,
                 cutout_rate=0.5, cut_out_length=50, cut_out_holes=1, cut_out_threshold=0.5,
                 is_addNoise=True, is_changeLight=True, is_cutout=True, is_rotate_img_bbox=True,
                 is_crop_img_bboxes=True, is_shift_pic_bboxes=True, is_filp_pic_bboxes=True):

        # 配置各个操作的属性
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate

        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold

        # 是否使用某种增强方式
        self.is_addNoise = is_addNoise
        self.is_changeLight = is_changeLight
        self.is_cutout = is_cutout
        self.is_rotate_img_bbox = is_rotate_img_bbox
        self.is_crop_img_bboxes = is_crop_img_bboxes
        self.is_shift_pic_bboxes = is_shift_pic_bboxes
        self.is_filp_pic_bboxes = is_filp_pic_bboxes

    # ----1.加噪声---- #
    def _addNoise(self, img):
        '''
        输入:
            img:图像array
        输出:
            加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        '''
        # return cv2.GaussianBlur(img, (11, 11), 0)
        return random_noise(img, mode='gaussian', clip=True) * 255

    # ---2.调整亮度--- #
    def _changeLight(self, img):
        alpha = random.uniform(0.35, 1)
        blank = np.zeros(img.shape, img.dtype)
        return cv2.addWeighted(img, alpha, blank, 1 - alpha, 0)

    # ---3.cutout--- #
    def _cutout(self, img, bboxes, length=100, n_holes=1, threshold=0.5):
        '''
        原版本：https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        Randomly mask out one or more patches from an image.
        Args:
            img : a 3D numpy array,(h,w,c)
            bboxes : 框的坐标
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        '''

        def cal_iou(boxA, boxB):
            '''
            boxA, boxB为两个框，返回iou
            boxB为bouding box
            '''
            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            iou = interArea / float(boxBArea)
            return iou

        # 得到h和w
        if img.ndim == 3:
            h, w, c = img.shape
        else:
            _, h, w, c = img.shape
        mask = np.ones((h, w, c), np.float32)
        for n in range(n_holes):
            chongdie = True  # 看切割的区域是否与box重叠太多
            while chongdie:
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - length // 2, 0,
                             h)  # numpy.clip(a, a_min, a_max, out=None), clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                chongdie = False
                for box in bboxes:
                    if cal_iou([x1, y1, x2, y2], box) > threshold:
                        chongdie = True
                        break
            mask[y1: y2, x1: x2, :] = 0.
        img = img * mask
        return img

    # ---4.旋转--- #
    def _rotate_img_bbox(self, img, bboxes, angle=5, scale=1.):
        w, h = img.shape[1], img.shape[0]
        rangle = np.deg2rad(angle)  # angle in radians
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        rot_bboxes = []
        for bbox in bboxes:
            points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
            new_points = cv2.transform(points[None, :, :], rot_mat)[0]
            rx, ry, rw, rh = cv2.boundingRect(new_points)
            corrected_bbox = [max(0, rx), max(0, ry), min(nw, rx + rw), min(nh, ry + rh)]
            corrected_bbox = [int(val) for val in corrected_bbox]  # Convert to int and correct order if necessary
            rot_bboxes.append(corrected_bbox)
        return rot_img, rot_bboxes

    # ---5.裁剪--- #
    def _crop_img_bboxes(self, img, bboxes):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        '''
        # 裁剪图像
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max  # 包含所有目标框的最小框到右边的距离
        d_to_top = y_min  # 包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max  # 包含所有目标框的最小框到底部的距离

        # 随机扩展这个最小框
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # 随机扩展这个最小框 , 防止别裁的太小
        # crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
        # crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
        # crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
        # crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

        # 确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # 裁剪boundingbox
        # 裁剪后的boundingbox坐标计算
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min, bbox[2] - crop_x_min, bbox[3] - crop_y_min])

        return crop_img, crop_bboxes

    # ---6.平移--- #
    def _shift_pic_bboxes(self, img, bboxes):
        h, w = img.shape[:2]
        x = random.uniform(-w * 0.2, w * 0.2)
        y = random.uniform(-h * 0.2, h * 0.2)
        M = np.float32([[1, 0, x], [0, 1, y]])
        shift_img = cv2.warpAffine(img, M, (w, h))

        shift_bboxes = []
        for bbox in bboxes:
            new_bbox = [bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y]
            corrected_bbox = [max(0, new_bbox[0]), max(0, new_bbox[1]), min(w, new_bbox[2]), min(h, new_bbox[3])]
            corrected_bbox = [int(val) for val in corrected_bbox]  # Convert to int and correct order if necessary
            shift_bboxes.append(corrected_bbox)
        return shift_img, shift_bboxes

    # ---7.镜像--- #
    def _filp_pic_bboxes(self, img, bboxes):
        # Randomly decide the flip method
        flipCode = random.choice([-1, 0, 1])  # -1: both; 0: vertical; 1: horizontal
        flip_img = cv2.flip(img, flipCode)  # Apply the flip
        h, w, _ = img.shape
        flip_bboxes = []

        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            if flipCode == 0:  # Vertical flip
                new_bbox = [x_min, h - y_max, x_max, h - y_min]
            elif flipCode == 1:  # Horizontal flip
                new_bbox = [w - x_max, y_min, w - x_min, y_max]
            else:  # Both flips
                new_bbox = [w - x_max, h - y_max, w - x_min, h - y_min]
            flip_bboxes.append(new_bbox)

        return flip_img, flip_bboxes

    # 图像增强方法
    def dataAugment(self, img, bboxes):
        '''
        图像增强
        输入:
            img:图像array
            bboxes:该图像的所有框坐标
        输出:
            img:增强后的图像
            bboxes:增强后图片对应的box
        '''
        change_num = 0  # 改变的次数
        # print('------')
        while change_num < 1:  # 默认至少有一种数据增强生效

            if self.is_rotate_img_bbox:
                if random.random() > self.rotation_rate:  # 旋转
                    change_num += 1
                    angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                    scale = random.uniform(0.7, 0.8)
                    img, bboxes = self._rotate_img_bbox(img, bboxes, angle, scale)

            if self.is_shift_pic_bboxes:
                if random.random() < self.shift_rate:  # 平移
                    change_num += 1
                    img, bboxes = self._shift_pic_bboxes(img, bboxes)

            if self.is_changeLight:
                if random.random() > self.change_light_rate:  # 改变亮度
                    change_num += 1
                    img = self._changeLight(img)

            if self.is_addNoise:
                if random.random() < self.add_noise_rate:  # 加噪声
                    change_num += 1
                    img = self._addNoise(img)
            if self.is_cutout:
                if random.random() < self.cutout_rate:  # cutout
                    change_num += 1
                    img = self._cutout(img, bboxes, length=self.cut_out_length, n_holes=self.cut_out_holes,
                                       threshold=self.cut_out_threshold)
            if self.is_filp_pic_bboxes:
                if random.random() < self.flip_rate:  # 翻转
                    change_num += 1
                    img, bboxes = self._filp_pic_bboxes(img, bboxes)

        return img, bboxes


class TxtToolHelper():
    def parse_txt(self, path):
        '''
        输入：
            txt_path: txt的文件路径
        输出：
            从txt文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, class_name]]
        '''
        with open(path, 'r') as file:
            lines = file.readlines()
            coords = []
            for line in lines:
                parts = line.strip().split()
                class_name = parts[0]
                x_center, y_center, width, height = map(float, parts[1:])
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)
                coords.append([x_min, y_min, x_max, y_max, class_name])
            return coords

    def save_txt(self, file_name, save_folder, bboxs_info):
        '''
        :param file_name:文件名
        :param save_folder:保存的txt文件的结果
        :param bboxs_info:边界框信息，格式为[[x_center, y_center, width, height, class_name]]
        :return:
        '''
        with open(os.path.join(save_folder, file_name), 'w') as file:
            for box in bboxs_info:
                # 确保 box 包含五个值：x_center, y_center, width, height, class_name
                x_center, y_center, width, height, class_name = box
                file.write(f"{class_name} {x_center} {y_center} {width} {height}\n")

def save_image(file_name, save_folder, img):
    cv2.imwrite(os.path.join(save_folder, file_name), img)

if __name__ == '__main__':
    need_aug_num = 4  # 每张图片需要增强的次数

    dataAug = DataAugmentForObjectDetection()  # 数据增强工具类
    txt_toolhelper = TxtToolHelper()  # txt工具

    # 获取相关参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_img_path', type=str, default='F:\\model\\yolo11\\my-data\\coco128\\images\\train2017')
    parser.add_argument('--source_txt_path', type=str, default='F:\\model\\yolo11\\my-data\\coco128\\labels\\train2017')
    parser.add_argument('--save_img_path', type=str,default='F:\\model\\yolo11\\my-data\\coco128\\images_kc\\train2017')
    parser.add_argument('--save_txt_path', type=str,default='F:\\model\\yolo11\\my-data\\coco128\\labels_kc\\train2017')
    args = parser.parse_args()
    source_img_path = args.source_img_path
    source_txt_path = args.source_txt_path

    save_img_path = args.save_img_path
    save_txt_path = args.save_txt_path

    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)

    if not os.path.exists(save_txt_path):
        os.mkdir(save_txt_path)

    for parent, _, files in os.walk(source_img_path):
        files.sort()
        for file in files:
            cnt = 0
            pic_path = os.path.join(parent, file)
            txt_path = os.path.join(source_txt_path, file[:-4] + '.txt')
            values = txt_toolhelper.parse_txt(txt_path)
            coords = [v[:4] for v in values]
            labels = [v[-1] for v in values]

            img = cv2.imread(pic_path)

            while cnt < need_aug_num:
                auged_img, auged_bboxes = dataAug.dataAugment(img, coords)
                auged_bboxes_int = np.array(auged_bboxes).astype(np.int32)
                height, width, channel = auged_img.shape
                img_name = '{}_{}{}'.format(file[:-4], cnt + 1, file[-4:])

                save_image(img_name, save_img_path, auged_img)

                new_bboxes_info = []
                for bbox in auged_bboxes_int:
                    x_min, y_min, x_max, y_max = bbox
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min
                    new_bboxes_info.append([x_center, y_center, width, height, labels[0]])

                txt_toolhelper.save_txt('{}_{}.txt'.format(file[:-4], cnt + 1), save_txt_path, new_bboxes_info)
                print(img_name)
                cnt += 1

