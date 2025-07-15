import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('runs/train/conbine416/GD_C3k2_EIEM_DAT_CA_HSFPN_TADDH/weights/best.pt') # select your model.pt path
    model.predict(source='my-data/GD/images/test',
                  imgsz=416,
                  project='runs/keshihua',
                  name='GD-ours',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                )