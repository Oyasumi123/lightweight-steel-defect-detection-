import torch
from ultralytics.nn.extra_modules import *

cpca = CPCA(input_channels=256,internal_neurons=64)
x = torch.randn(1, 256, 32, 32)  # 假设输入为 (batch_size, channels, height, width)
y = cpca(x)
print(y.shape)  # 确认输出维度正确
