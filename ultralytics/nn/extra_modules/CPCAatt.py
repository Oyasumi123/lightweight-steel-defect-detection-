import torch
import torch.nn.functional as F
from torch import nn


class CPCAChannelAttention(nn.Module):
    """CPCA 通道注意力模块"""

    def __init__(self, input_channels, internal_neurons):
        """
        Args:
            input_channels (int): 输入特征图的通道数。
            internal_neurons (int): 通道注意力中的中间层通道数（降维后再升维）。
        """
        super(CPCAChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        # 全局平均池化分支
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)

        # 全局最大池化分支
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)

        # 通道注意力融合
        x = x1 + x2
        return inputs * x.view(-1, self.input_channels, 1, 1)


class CPCASpatialAttention(nn.Module):
    """CPCA 空间注意力模块"""

    def __init__(self, input_channels):
        """
        Args:
            input_channels (int): 输入特征图的通道数。
        """
        super(CPCASpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=7, stride=1, padding=3, bias=True)

    def forward(self, inputs):
        # 空间注意力权重计算
        weights = self.conv(inputs)
        weights = torch.sigmoid(weights)
        return inputs * weights


class CPCA(nn.Module):
    """完整的 CPCA 模块，包含通道注意力和空间注意力"""

    def __init__(self, input_channels, internal_neurons=128):
        """
        Args:
            input_channels (int): 输入特征图的通道数。
            internal_neurons (int): 通道注意力中的中间层通道数。
        """
        super(CPCA, self).__init__()
        self.channel_attention = CPCAChannelAttention(input_channels, internal_neurons)
        self.spatial_attention = CPCASpatialAttention(input_channels)

    def forward(self, inputs):
        # 通道注意力
        x = self.channel_attention(inputs)
        # 空间注意力
        x = self.spatial_attention(x)
        return x
