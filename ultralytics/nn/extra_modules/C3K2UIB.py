import torch
import torch.nn as nn
from typing import Optional

__all__ = ['C3k2_UIB']


# 保持原有的make_divisible和autopad函数不变
def make_divisible(value: float, divisor: int, min_value: Optional[float] = None,
                   round_down_protect: bool = True) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class MobileMQA(nn.Module):
    """Mobile Multi-Query Attention"""

    def __init__(self, dim, num_heads=4, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 高效的注意力实现
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, head_dim, bias=False)  # 单个key head
        self.v = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        # 多查询单键值注意力计算
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).unsqueeze(1)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Conv(nn.Module):
    """标准卷积层"""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class EnhancedUIB(nn.Module):
    def __init__(self, inp, oup, expand_ratio=4, use_mqa=False):
        super().__init__()
        hidden_dim = make_divisible(inp * expand_ratio, 8)

        # 深度可分离卷积部分
        self.conv_pw1 = Conv(inp, hidden_dim, k=1)
        self.conv_dw = Conv(hidden_dim, hidden_dim, k=3, g=hidden_dim, act=False)

        # Mobile MQA
        self.use_mqa = use_mqa
        if use_mqa:
            self.mqa = MobileMQA(hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

        # 输出投影
        self.conv_pw2 = Conv(hidden_dim, oup, k=1, act=False)

    def forward(self, x):
        x = self.conv_pw1(x)
        x = self.conv_dw(x)

        # 应用Mobile MQA
        if self.use_mqa:
            B, C, H, W = x.shape
            x_flat = x.flatten(2).transpose(1, 2)  # [B, HW, C]
            x_flat = self.norm(x_flat)
            x_flat = self.mqa(x_flat)
            x = x_flat.transpose(1, 2).reshape(B, C, H, W)

        x = self.conv_pw2(x)
        return x


class C3k2_UIB(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        # 使用增强型UIB
        self.m = nn.ModuleList([
            EnhancedUIB(c_, c_, expand_ratio=2, use_mqa=False)
            for _ in range(n)
        ])
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)

        # 并行处理特征
        y1 = torch.cat([m(y1) for m in self.m], dim=1)

        # 特征融合
        y = self.cv3(torch.cat((y1, y2), dim=1))
        return y + x if self.shortcut else y