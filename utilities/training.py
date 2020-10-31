
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


class Training:
    '''
    Contains 'at-runtime' methods to help training process
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
    def noise_vector(n_examples: int, n_hidden: int) -> Tensor:
        '''Creates a random noise vector Z of shape (n_examples, n_hidden)'''
        return torch.randn((n_examples, n_hidden))
    
    @staticmethod
    def plot_history(history: List[float], window_size=100):
        i = 0
        numbers = history
        moving_averages = []
        while i < (len(numbers) - window_size + 1):
            window = numbers[i : i + window_size]
            window_average = sum(window) / window_size
            moving_averages.append(window_average)
            i += 1
        plt.plot(moving_averages)