
from typing import Tuple
import torch
from torch import nn
from torch import Tensor

from utilities.layers import Layers


class Disc64(nn.Module):
    """Discriminator for 64x64 generated images"""
    def __init__(self, df_dim: int):
        super().__init__()
        self.img_code_s16 = Layers.encode_image_by_16times(df_dim)
        self.outlogits = nn.Sequential(
            nn.Conv2d(df_dim * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid()
            )

    def forward(self, X: Tensor):
        X = self.img_code_s16(X)    # 4 x 4 x 8df
        X = self.outlogits(X)       # batch, 1, 1, 1
        X = X.view(-1)              # batch,
        return X


class Disc128(nn.Module):
    """Discriminator for 128x128 generated images"""
    def __init__(self, df_dim: int):
        super().__init__()
        self.img_code_s16 = Layers.encode_image_by_16times(df_dim)
        self.img_code_s32 = Layers.downBlock(df_dim * 8, df_dim * 16)
        self.img_code_s32_1 = Layers.Block3x3_leakRelu(df_dim * 16, df_dim * 8)
        self.outlogits = nn.Sequential(
            nn.Conv2d(df_dim * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid()
            )

    def forward(self, X: Tensor):
        X = self.img_code_s16(X)   # 8 x 8 x 8df
        X = self.img_code_s32(X)   # 4 x 4 x 16df
        X = self.img_code_s32_1(X)  # 4 x 4 x 8df
        X = self.outlogits(X)       # batch, 1, 1, 1
        X = X.view(-1)              # batch,
        return X


class Disc256(nn.Module):
    """Discriminator for 256x256 generated images"""
    def __init__(self, df_dim: int):
        super().__init__()
        self.img_code_s16 = Layers.encode_image_by_16times(df_dim)
        self.img_code_s32 = Layers.downBlock(df_dim * 8, df_dim * 16)
        self.img_code_s64 = Layers.downBlock(df_dim * 16, df_dim * 32)
        self.img_code_s64_1 = Layers.Block3x3_leakRelu(df_dim * 32, df_dim * 16)
        self.img_code_s64_2 = Layers.Block3x3_leakRelu(df_dim * 16, df_dim * 8)
        self.outlogits = nn.Sequential(
            nn.Conv2d(df_dim * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid()
            )

    def forward(self, X: Tensor):
        X = self.img_code_s16(X)
        X = self.img_code_s32(X)
        X = self.img_code_s64(X)
        X = self.img_code_s64_1(X)
        X = self.img_code_s64_2(X)
        X = self.outlogits(X)       # batch, 1, 1, 1
        X = X.view(-1)              # batch,
        return X