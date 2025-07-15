import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')
    model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='ultralytics/cfg/datasets/gangdai.yaml',
                # cache=False,
                imgsz=416,
                epochs=200,
                batch=8,
                # close_mosaic=0,
                workers=2, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                lr0=0.01,
                # fraction=0.2,
                project='runs/train200/size416',
                name='GD_C3k2-SHSA-CGLU',
                )

