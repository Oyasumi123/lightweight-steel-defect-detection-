import torch
from thop import profile
from ultralytics import YOLO

# 加载模型
model =YOLO("ultralytics/cfg/models/11/yolo11_dynaconv.yaml")  # 使用模型配置文件

# 创建一个dummy输入
inputs = torch.randn(1, 3, 640, 640)  # 假设输入是3通道640x640的图像

# 使用thop库计算FLOPs
flops, params = profile(model, inputs=(inputs, ))
print(f"FLOPs: {flops}, Params: {params}")