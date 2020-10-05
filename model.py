
#%%

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
from matplotlib.figure import Figure
from typing import List
from datetime import datetime

# file imports
from loader import get_mnist_data
from modelutils import Utils
from base_classes import GanDiscriminator, GanGenerator, GanTrainer


class Discriminator(GanDiscriminator):
    def __init__(self):
        super().__init__()
        self.conv1 = Utils.convdown_block(1, 16, in_hw=28, out_hw=14)
        self.conv2 = Utils.convdown_block(16, 32, in_hw=14, out_hw=7)
        self.conv3 = Utils.convdown_block(32, 64, in_hw=7, out_hw=3)
        self.conv4 = Utils.convdown_block(64, 128, in_hw=3, out_hw=1)
        self.fc_1 = nn.Linear(in_features=128*1*1, out_features=128)
        self.fc_discriminate = nn.Linear(in_features=128, out_features=1)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) # (M, 128, 1, 1)
        (M, nc, h, w) = x.shape
        x = x.view(M, nc*h*w) # (M, 128)
        x = F.leaky_relu(self.fc_1(x), 0.2)
        x = self.fc_discriminate(x)
        return x
    
    
class Generator(GanGenerator):
    def __init__(self, latent_dim=6272, n_channels=128):
        super().__init__()
        self.n_channels = n_channels
        self.fc = nn.Linear(in_features=latent_dim, out_features=n_channels*2*2)
        self.up1 = Utils.upblock(in_channels=n_channels, out_channels=n_channels//2) # 4x4
        self.up2 = Utils.upblock(in_channels=n_channels//2, out_channels=n_channels//4) # 8x8
        self.up3 = Utils.upblock(in_channels=n_channels//4, out_channels=n_channels//8) # 16x16
        self.up4 = Utils.upblock(in_channels=n_channels//8, out_channels=n_channels//16) # 32x32
        self.conv_out = Utils.conv(in_channels=n_channels//16, out_channels=1, in_hw=32, out_hw=28, max_kern=7, max_pad=5)
    
    def forward(self, z: Tensor) -> Tensor:
        z = F.leaky_relu(self.fc(z), 0.2)
        M = z.shape[0]
        z = z.view(M, self.n_channels, 2, 2) # (nc, 2, 2)
        z = self.up1(z) # (nc, 4, 4)
        z = self.up2(z) # (nc, 8, 8)
        z = self.up3(z) # (nc, 16, 16)
        z = self.up4(z) # (nc, 32, 32)
        z = self.conv_out(z) # (1, 28, 28)
        return F.sigmoid(z)


class Trainer(GanTrainer):
    def __init__(self, discriminator: GanDiscriminator, generator: GanGenerator):
        super().__init__(discriminator=discriminator, generator=generator)


# Load in Data 
train_dataloader = get_mnist_data(batch_size=16)

# Create Network
gan = Trainer(discriminator=Discriminator(), generator=Generator())

# Pretrain Network
gan.pretrain_discriminator(train_dataloader, n_batches=5, lr=0.002)


#%%
# Train Gan
gan.train_gan(train_dataloader, n_batches=1000, show_images_every_n=200, D_vs_G_updateratio=1, disc_lr=0.0002, gen_lr=0.0004)

#%% Evaluation: Plot Loss History
Utils.plot_history(gan.generator.loss_history, window_size=10)
Utils.plot_history(gan.discriminator.loss_history, window_size=10)

#%%

gan.generator.evaluate_with_image_grid(100)