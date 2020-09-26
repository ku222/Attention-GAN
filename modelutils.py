

# Pytorch imports
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from torch.nn.utils import clip_grad_norm_ as clip_gradients

# standard imports
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# file imports
from loader import get_mnist_data


class Utils:
    '''
    Contains helpful methods like making convolutional layers,
    or entire blocks which chain together multiple layers
    '''
    @staticmethod
    def scale_255_to_1(images: Tensor) -> Tensor:
        '''Scale image tensor value range from [0,255] to [-1,1]'''
        return (images - 127.5) / 127.5
    
    @staticmethod
    def scale_1_to_255(images: Tensor) -> Tensor:
        '''Scale image tensor value range from [-1,1] to [0,255]'''
        return (images * 127.5) + 127.5
    
    @staticmethod
    def display_image(image: Tensor) -> None:
        image = image.detach().cpu()
        return plt.imshow(image.permute(1, 2, 0))
    
    @staticmethod
    def calculate_out_hw(hw: int, k: int, s: int, p=0) -> int:
        '''Calculates output hw given input hw, kernel size, stride, padding'''
        return math.floor(((hw + 2*p - k)/s)+1)
        
    @staticmethod
    def conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
        '''Create 3x3 conv layer where input_h/w == output_h/w'''
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    
    @staticmethod
    def conv(in_channels: int, out_channels: int, in_hw: int, out_hw: int, max_kern=4, max_stride=3, max_pad=3) -> nn.Conv2d:
        '''Create KxK conv layer which will scale our output h/w down to the desired h/w'''
        kernels = [k for k in range(1, max_kern+1)]
        strides = [s for s in range(1, max_stride+1)]
        paddings = [p for p in range(max_pad+1)]
        params = [(k,s,p) for k in kernels   for s in strides   for p in paddings]
        valid_params = [(k,s,p) for (k,s,p) in params if Utils.calculate_out_hw(in_hw, k, s, p)==out_hw]
        try: (k,s,p) = max(valid_params, key=lambda x: (x[0], x[2], x[1]))
        except: raise Exception('Could not find valid parameters to produce output hw')
        return nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False)
        
    @staticmethod
    def outpicture_block(in_channels: int, out_channels: int, in_hw: int, out_hw: int) -> nn.Sequential:
        '''Block with conv layer and a Tanh activation'''
        return nn.Sequential(
            Utils.conv(in_channels, out_channels, in_hw, out_hw, max_kern=5),
            nn.Tanh()
        )
        
    @staticmethod
    def upblock(in_channels: int, out_channels: int) -> nn.Sequential:
        '''Block that increases h/w by 2, applies 3x3 convolution, LeakyReLU(0.2), BatchNorm'''
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            Utils.conv3x3(in_channels, out_channels),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features=out_channels)
        )
        
    @staticmethod
    def convdown_block(in_channels: int, out_channels: int, in_hw: int, out_hw: int) -> nn.Sequential:
        '''
        Conv + leakyReLU(0.2) + Dropout(p=0.4)
        '''
        assert out_hw <= 0.5*in_hw
        return nn.Sequential(
            Utils.conv(in_channels, out_channels, in_hw=in_hw, out_hw=out_hw*2, max_kern=3),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features=out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.4)
        )

    @staticmethod
    def noise_vector(n_examples: int, n_hidden: int) -> Tensor:
        '''Creates a random noise vector Z of shape (n_examples, n_hidden)'''
        return torch.randn((n_examples, n_hidden))
    
    @staticmethod
    def plot_history(history: List[float], window_size=100):
        i = 0
        numbers = history
        moving_averages = []
        while i < len(numbers) - window_size + 1:
            this_window = numbers[i : i + window_size]
            window_average = sum(this_window) / window_size
            moving_averages.append(window_average)
            i += 1
        plt.plot(moving_averages)