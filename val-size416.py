import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/size416/GD_original2/weights/best.pt') # 选择训练好的权重路径
    model.val(data='ultralytics/cfg/datasets/gangdai.yaml',
              split='test', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=416,
              batch=8,
              # iou=0.7,
              rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val/size416',
              name='GD_416original2',
              )