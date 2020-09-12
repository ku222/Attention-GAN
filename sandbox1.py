

#%% Imports
import torch
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from torchvision import transforms

import numpy as np

#%% Load dataset

def get_mnist_data(batch_size: int=8) -> DataLoader:
    # prepare image transform pipeline
    image_transforms: transforms.Compose = transforms.Compose([transforms.ToTensor()])
    mnist_trainset: Dataset = datasets.MNIST(root='./data', train=True, download=True, transform=image_transforms)
    return DataLoader(dataset=mnist_trainset, batch_size=batch_size, shuffle=True)
    

class GLU(nn.Module):
    '''
    Gated Linear Unit Implementation
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])

def make_conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    '''Create 3x3 conv layer where input dim == output dim'''
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    
def make_upblock(in_channels: int, out_channels: int) -> nn.Sequential:
    '''Block that increases h/w by 2, applies 3x3 convolution, and a GLU'''
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        make_conv3x3(in_channels, out_channels*2),
        nn.BatchNorm2d(out_channels*2),
        GLU()
    )
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=100, out_features=49)
        self.upblock = make_upblock(in_channels=1, out_channels=6)
        
        
    def 
        