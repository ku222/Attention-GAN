
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
from typing import List

# file imports
from loader import get_mnist_data
from modelutils import Utils

class GLU(nn.Module):
    '''
    Gated Linear Unit Implementation
    Illustration of logic is here: https://bit.ly/2RlbwqU
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


class Generator(nn.Module):
    def __init__(self, noise_vector_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_features=noise_vector_dim, out_features=256)
        self.upblock_16_8 = Utils.upblock(in_channels=16, out_channels=8) # -> 8x8
        self.upblock_8_4 = Utils.upblock(in_channels=8, out_channels=4) # -> 16x16
        self.upblock_4_2 = Utils.upblock(in_channels=4, out_channels=2) # -> 32x32
        self.outblock_2_1 = Utils.outpicture_block(in_channels=2, out_channels=1, in_hw=32, out_hw=28)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)              # (M, 256)
        # Reshape
        M, h, w = x.shape[0], 4, 4  # (M, nc, h, w)
        x = x.view(M, 16, h, w)     # (M, 16, 4, 4)
        # Upsample                  
        x = self.upblock_16_8(x)    # (M, 8, 8, 8)
        x = self.upblock_8_4(x)     # (M, 4, 16, 16)
        x = self.upblock_4_2(x)     # (M, 2, 32, 32)
        x = self.outblock_2_1(x)    # (M, 1, 28, 28)
        return F.sigmoid(x)
    

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1_8 = Utils.convdown_block(1, 8, in_hw=28, out_hw=14)
        self.conv_8_16 = Utils.convdown_block(8, 16, in_hw=14, out_hw=7)
        self.conv_16_32 = Utils.convdown_block(16, 32, in_hw=7, out_hw=2)
        self.fc_1 = nn.Linear(in_features=32*2*2, out_features=128)
        self.fc_out = nn.Linear(in_features=128, out_features=1)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1_8(x)
        x = self.conv_8_16(x)
        x = self.conv_16_32(x)
        (M, nc, h, w) = x.shape
        x = x.view(M, nc*h*w)
        x = self.fc_1(x)
        x = self.fc_out(x)
        return x
    
    def pretrain(self, train_dataloader: DataLoader, n_batches: int) -> None:
        print('Pretraining Discriminator =================================')
        self.train()
        start_time = time.time()
        criterion = nn.BCEWithLogitsLoss()
        device = torch.device('cuda')
        self.cuda()
        optimizer = optim.AdamW(self.parameters(), lr=0.01)
        self.loss_history = []
        self.accuracy_history = []
        for (i, minibatch) in enumerate(train_dataloader):
            real_images: Tensor = minibatch[0]
            real_labels: Tensor = torch.ones(len(real_images))
            fake_images: Tensor = F.sigmoid(torch.randn((len(real_images), 1, 28, 28)))
            fake_labels: Tensor = torch.zeros(len(fake_images))
            images = torch.cat([real_images, fake_images]).to(device)
            labels = torch.cat([real_labels, fake_labels]).to(device).view(-1, 1)
            optimizer.zero_grad()
            logits = self(images)
            loss = criterion(input=logits, target=labels)
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())
            accuracy = self._compute_accuracy(logits, labels)
            self.accuracy_history.append(accuracy)
            print(f'\t\t Minibatch:{i+1}/{n_batches}')
            print(f'\t\t\t Accuracy = {accuracy}')
            print(f'\t\t\t Loss = {loss.item()}')
            if (i+1) == n_batches:
                print(logits)
                print(labels)
                break
  
    def _compute_accuracy(self, logits: Tensor, labels: Tensor) -> float:
        # logits: (M, 1), labels: (M, 1)
        with torch.no_grad():
            preds = F.sigmoid(logits)
            
            return num_equal/len(labels)
            
            
class Gan:
    def __init__(self, noise_vector_dim: int):
        self.generator = Generator(noise_vector_dim=noise_vector_dim)
        self.discriminator = Discriminator()
        self.generator_losses = []
        self.discriminator_losses = []
        
    def compute_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        criterion = nn.BCEWithLogitsLoss()
        labels = labels.view(-1, 1)
        return criterion(logits, labels)
            
    def pretrain_discriminator(self, batch_size=16, n_batches=50) -> None:
        train_dataloader = get_mnist_data(batch_size=batch_size)
        self.discriminator.pretrain(train_dataloader, n_batches=n_batches)
        
    def train_gan(self, n_epochs: int, disc_steps: int, batch_size=8) -> None:
        self.discriminator.fc_out = self.discriminator.fc_discriminate
        device = torch.device('cuda')
        train_dataloader = get_mnist_data(batch_size=batch_size)
        disc_optimizer = optim.AdamW(self.discriminator.parameters(), lr=0.05)
        gen_optimizer = optim.AdamW(self.generator.parameters())
        self.discriminator.cuda()
        self.generator.cuda()
        for e in range(n_epochs):
            print(f'\t Epoch {e+1}/{n_epochs}')
            for (i, minibatch) in enumerate(train_dataloader):
                print(f'\t\t Minibatch:{i}/{len(train_dataloader)}')
                # Discriminator
                for _ in range(disc_steps):
                    (real_images, _) = minibatch
                    real_images = real_images.to(device)
                    z = Utils.noise_vector(n_examples=batch_size, n_hidden=256).to(device)
                    fake_images = self.generator(z).to(device)
                    real_labels = torch.FloatTensor(batch_size).uniform_(0.9, 0.95).to(device)
                    fake_labels = torch.FloatTensor(batch_size).uniform_(0.05, 0.1).to(device)
                    disc_optimizer.zero_grad()
                    
                    all_images = torch.cat([real_images, fake_images])
                    all_labels = torch.cat([real_labels, fake_labels])
                    logits = self.discriminator(all_images)
                    disc_loss = self.compute_loss(logits=logits, labels=all_labels)
                    
                    disc_loss.backward()
                    disc_optimizer.step()
                    self.discriminator_losses.append(disc_loss.item())
                    print(f'\t\t\t Discriminator Loss = {disc_loss.item()}')
                # Generator
                gen_optimizer.zero_grad()
                z = Utils.noise_vector(n_examples=batch_size, n_hidden=256).to(device)
                fake_images = self.generator(z).to(device)
                logits = self.discriminator(fake_images)
                labels = torch.FloatTensor(batch_size).uniform_(0.9, 0.95).to(device)
                gen_loss = self.compute_loss(logits=logits, labels=labels)
                gen_loss.backward()
                gen_optimizer.step()
                self.generator_losses.append(gen_loss.item())
                print(f'\t\t\t Generator Loss = {gen_loss.item()}')
        
gan = Gan(noise_vector_dim=256)


#%%

gan.pretrain_discriminator(batch_size=8)

#%%
# training

gan.train_gan(n_epochs=1, disc_steps=3, batch_size=8)

#%%

Utils.plot_history(gan.discriminator_losses)

#%%

z = Utils.noise_vector(n_examples=8, n_hidden=256)
gen = Generator(noise_vector_dim=256)
fake_images = gen(z)

disc = Discriminator()
disc(fake_images).shape

#%%

train_dataloader = get_mnist_data(batch_size=32)
image = next(iter(train_dataloader))[0][0, :, :, :]
#Utils.display_image(image)

next(iter(train_dataloader))[1].shape
     
#%%


disc.pretrain(train_dataloader, n_epochs=1)

#%%


torch.FloatTensor(8).uniform_(0.7, 0.9)

#%%

image = next(iter(train_dataloader))[0][0].view(1, 1, 28, 28)
print(torch.argmax(disc(image.to(torch.device('cuda'))), dim=1))
Utils.display_image(image.view(1, 28, 28))


#%%

z = Utils.noise_vector(n_examples=8, n_hidden=256)
fake_images = gen(z)

#%%


with torch.no_grad():
    Utils.display_image(fake_images[0])

#%%

gan = Gan(noise_vector_dim=256)


gan.train_gan(n_epochs=1, batch_size=16)

#%%

z = Utils.noise_vector(n_examples=8, n_hidden=256).to(torch.device('cuda'))
fake_images = gan.generator(z)
fake_images

#%%
with torch.no_grad():
    Utils.display_image(fake_images[0])