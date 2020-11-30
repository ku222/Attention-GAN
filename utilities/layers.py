
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from math import floor

class Layers:
    """
    Helper class with static methods to help instantiate
    new pytorch layers and blocks
    """
    @staticmethod
    def GLU() -> "Layers.GLU":
        """Instantiates Gated Linear Unit; shrinks filter dims by 2"""
        class GLU(nn.Module):
            def __init__(self):
                super(GLU, self).__init__()
            # Forward prop
            def forward(self, x: Tensor) -> Tensor:
                nc = x.size(1)
                assert nc % 2 == 0, 'channels dont divide 2!'
                nc = int(nc/2)
                return x[:, :nc] * F.sigmoid(x[:, nc:])
        # Return new instance
        return GLU()
    
    @staticmethod
    def conv(in_channels: int, out_channels: int, in_hw: int, out_hw: int, max_kern=4, max_stride=3, max_pad=3) -> nn.Conv2d:
        '''Create KxK conv layer which will scale our output h/w down to the desired h/w'''
        kernels = [k for k in range(1, max_kern+1)]
        strides = [s for s in range(1, max_stride+1)]
        paddings = [p for p in range(max_pad+1)]
        params = [(k,s,p) for k in kernels   for s in strides   for p in paddings]
        valid_params = [(k,s,p) for (k,s,p) in params if Layers.calculate_out_hw(in_hw, k, s, p)==out_hw]
        try: (k,s,p) = max(valid_params, key=lambda x: (x[0], x[2], x[1]))
        except: raise Exception('Could not find valid parameters to produce output hw')
        return nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False)

    @staticmethod
    def calculate_out_hw(hw: int, k: int, s: int, p=0) -> int:
        '''Calculates output hw given input hw, kernel size, stride, padding'''
        return floor(((hw + 2*p - k)/s)+1)

    @staticmethod
    def conv1x1(in_planes: int, out_planes: int, bias=False) -> nn.Conv2d:
        "Instantiates 1x1 convolution with padding"
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

    @staticmethod
    def conv3x3(in_planes: int, out_planes: int) -> nn.Conv2d:
        "Instantiates 3x3 convolution with padding - keeps h/w the same"
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
    
    @staticmethod
    def conv4x4DownSpatial(in_planes: int, out_planes: int, bias=True) -> nn.Conv2d:
        "Instantiates 4x4 convolution with padding - halves h/w"
        return nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=bias)

    @staticmethod
    def upBlock(in_planes: int, out_planes: int) -> nn.Sequential:
        """Instantiates a block that upsales the spatial size by a factor of 2"""
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            Layers.conv3x3(in_planes, out_planes * 2),
            nn.BatchNorm2d(out_planes * 2),
            Layers.GLU()
            )
        return block
    
    @staticmethod
    def upBlockReLU(in_planes: int, out_planes: int) -> nn.Sequential:
        """Instantiates a block that upsales the spatial size by a factor of 2"""
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            Layers.conv3x3(in_planes, out_planes),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
            )
        return block

    @staticmethod
    def downBlockLeakyReLU(in_planes: int, out_planes: int) -> nn.Sequential:
        """Instantiates a block that downsizes the spatial size by a factor of 2"""
        block = nn.Sequential(
            Layers.conv4x4DownSpatial(in_planes, out_planes),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
            )
        return block

    @staticmethod
    def Block3x3_relu(in_planes: int, out_planes: int) -> nn.Sequential:
        """Instantiates a block that contains conv3x3, batchnorm, GLU"""
        block = nn.Sequential(
            Layers.conv3x3(in_planes, out_planes * 2),
            nn.BatchNorm2d(out_planes * 2),
            Layers.GLU()
            )
        return block
    
    @staticmethod
    def Block3x3_leakRelu(in_planes: int, out_planes: int) -> nn.Sequential:
        """
        Instantiates a block that contains conv3x3, batchnorm, LeakyRelu.
        Keeps spatial dimensions unchanged.
        """
        block = nn.Sequential(
            Layers.conv3x3(in_planes, out_planes),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        return block
    
    @staticmethod
    def downBlock(in_planes: int, out_planes: int) -> nn.Sequential:
        """
        Instantiates block that downscales the spatial size by a factor of 2.
        Also contains LeakyRelu activation.
        """
        block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        return block

    @staticmethod
    def encode_image_by_16times(df_dims: int) -> nn.Sequential:
        """
        Instantiates block that downscales the spatial size by a factor of 16.
        Each individual miniblock will downsize spacial factor by 2, using
        kernel=4, padding=2, stride=1.
        df_dims = discriminator feature dims (suggest 64).
        Returns df_dims*8 channels.
        """
        block = nn.Sequential(
            # --> state size. ndf x in_size/2 x in_size/2
            nn.Conv2d(in_channels=3, out_channels=df_dims, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # --> state size 2ndf x x in_size/4 x in_size/4
            nn.Conv2d(in_channels=df_dims, out_channels=df_dims * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(df_dims * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # --> state size 4ndf x in_size/8 x in_size/8
            nn.Conv2d(in_channels=df_dims * 2, out_channels=df_dims * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(df_dims * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # --> state size 8ndf x in_size/16 x in_size/16
            nn.Conv2d(in_channels=df_dims * 4, out_channels=df_dims * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(df_dims * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return block

    @staticmethod
    def ResBlock(channel_num: int) -> "Layers.ResBlock":
        """Instantiates ResidialBlock - channel_num filters in -> same channel_num filters out"""
        class ResBlock(nn.Module):
            def __init__(self, channel_num: int):
                super(ResBlock, self).__init__()
                self.block = nn.Sequential(
                    Layers.conv3x3(channel_num, channel_num * 2),
                    nn.BatchNorm2d(channel_num * 2),
                    Layers.GLU(),
                    Layers.conv3x3(channel_num, channel_num),
                    nn.BatchNorm2d(channel_num)
                    )
            # Forward method
            def forward(self, x: Tensor) -> Tensor:
                residual = x
                out = self.block(x)
                out += residual
                return out
        # Return new instance
        return ResBlock(channel_num=channel_num)