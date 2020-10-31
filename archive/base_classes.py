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
from typing import List, Tuple, Union
from datetime import datetime
import random

# file imports
from loader import get_mnist_data
from modelutils import Utils



class GanDiscriminator(nn.Module):
    def __init__(self, device_type='cuda'):
        super().__init__()
        self.device = torch.device(device_type)
        self.cuda() # move weights to GPU
        self.loss_history = []
        self.accuracy_history = []
    
    def compute_accuracy(self, logits: Tensor, labels: Tensor) -> float:
        with torch.no_grad():
            logits = logits.to(self.device)
            labels = labels.to(self.device)
            logits: Tensor = F.sigmoid(logits)
            logits = torch.round(logits.double()).flatten()
            labels = torch.round(labels.double()).flatten()
            num_equal = torch.eq(logits, labels).sum().item()
            return (num_equal / len(labels))
        
    def log_stats(self, loss: Tensor, accuracy: float) -> None:
        print(f'\t\t\t Disc loss = {round(loss.item(), 6)}')
        print(f'\t\t\t Disc acc = {round(accuracy, 6)}')
        self.loss_history.append(loss.item())
        self.accuracy_history.append(accuracy)


class GanGenerator(nn.Module):
    def __init__(self, device_type='cuda'):
        super().__init__()
        self.device = torch.device(device_type)
        self.cuda() # move weights to GPU
        self.loss_history = []
        
    def log_stats(self, loss: Tensor) -> None:
        print(f'\t\t\t Gen loss = {round(loss.item(), 6)}')
        self.loss_history.append(loss.item())
        
    def make_noise(self, n_examples: int) -> Tensor:
        '''Produce a noise vector'''
        (_, fc_in) = next(self.parameters()).shape
        noise = torch.randn((n_examples, fc_in))
        return noise.to(self.device)
    
    def generate_images(self, n_examples: int) -> Tensor:
        '''Produces noise vector and feeds forward through self to generate images'''
        noise = self.make_noise(n_examples=n_examples)
        return self(noise)
    
    def evaluate_with_image_grid(self, n_images: int, epoch: int=None, img_folder_prefix='generated_images') -> None:
        '''Produces an [n_images, n_images] image grid for evaluation purposes, saves to an image folder'''
        self.cuda() # move to GPU
        sqrt = int(math.sqrt(n_images))
        assert sqrt == int(sqrt)
        noise = self.make_noise(n_examples=n_images)
        generated_images = self(noise)
        generated_images = generated_images.detach().cpu()
        # plot images
        (f, axarr) = plt.subplots(sqrt, sqrt)
        counter = 0
        for i in range(sqrt):
            for j in range(sqrt):
                # define subplot
                image = generated_images[counter]
                image = image.permute(1, 2, 0)
                axarr[i,j].axis('off')
                axarr[i,j].imshow(image)
                counter += 1
        # save plot to file
        timenow: str = str(datetime.now()).split('.')[0].replace(':', '-')
        fname = f'{img_folder_prefix}/epoch_{epoch+1}.png' if epoch else f'{img_folder_prefix}/GAN {timenow}.png'
        plt.savefig(fname)
        plt.close()


class GanTrainer:
    def __init__(self, discriminator: GanDiscriminator, generator: GanGenerator, device_type='cuda'):
        self.discriminator = discriminator
        self.discriminator.cuda()
        self.generator = generator
        self.generator.cuda()
        self.device = torch.device(device_type)
        
    def make_real_labels(self, num_labels: int, flattened=False) -> Tensor:
        labels = torch.FloatTensor(num_labels).uniform_(0.9, 0.9)
        labels = labels if flattened else labels.view(num_labels, 1)
        return labels.to(self.device)
    
    def make_fake_labels(self, num_labels: int, flattened=False) -> Tensor:
        labels = torch.FloatTensor(num_labels).uniform_(0.1, 0.1)
        labels = labels if flattened else labels.view(num_labels, 1)
        return labels.to(self.device)
    
    def make_fake_images(self, n_examples: int, also_create_labels=False) -> Union[Tensor, Tuple[Tensor]]:
        fake_images = self.generator.generate_images(n_examples=n_examples)
        if not also_create_labels:
            return fake_images
        fake_labels = self.make_fake_labels(num_labels=n_examples)
        return (fake_images, fake_labels)
        
    def make_real_images(self, train_dataloader: DataLoader, also_create_labels=False) -> Union[Tensor, Tuple[Tensor]]:
        (real_images, _) = next(iter(train_dataloader))
        real_images = real_images.to(self.device)
        if not also_create_labels:
            return real_images
        real_labels = self.make_real_labels(num_labels=train_dataloader.batch_size)
        return (real_images, real_labels)
    
    def bce_loss(self, logits: Tensor, labels: Tensor, apply_sigmoid=True, use_pytorch_loss=False) -> Tensor:
        if not use_pytorch_loss:
            logits = torch.clamp(logits, min=1e-20, max=0.9999)
            logits = F.sigmoid(logits) if apply_sigmoid else logits
            left = labels*torch.log(logits)
            right = (1 - labels)*torch.log(1 - logits)
            return -(left + right).mean()
        return nn.BCEWithLogitsLoss()(input=logits, target=labels)
    
    def stack_real_and_fake(self, real_imgs: Tensor, fake_imgs: Tensor, real_lbls: Tensor, fake_lbls: Tensor) -> Tuple[Tensor]:
        all_images = torch.cat([real_imgs, fake_imgs])
        all_labels = torch.cat([real_lbls, fake_lbls])
        return (all_images, all_labels)
        
    def pretrain_discriminator(self, train_dataloader: DataLoader, n_batches: int, lr=0.0002) -> None:
        optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=lr)
        batch_size = train_dataloader.batch_size
        for n in range(n_batches):
            optimizer.zero_grad()
            (real_images, real_labels) = self.make_real_images(train_dataloader, also_create_labels=True)
            (fake_images, fake_labels) = self.make_fake_images(n_examples=batch_size, also_create_labels=True)
            (images, labels) = self.stack_real_and_fake(real_images, fake_images, real_labels, fake_labels)
            logits = self.discriminator(images)
            loss = self.bce_loss(logits=logits, labels=labels, use_pytorch_loss=True)
            accuracy = self.discriminator.compute_accuracy(logits=logits, labels=labels)
            loss.backward()
            optimizer.step()
            self.discriminator.log_stats(loss=loss, accuracy=accuracy)
    
    def compute_D_vs_G_iterations(self, D_vs_G_updateratio: float) -> int:
        floor, ceil = math.floor(D_vs_G_updateratio), math.ceil(D_vs_G_updateratio)
        decimal_part = D_vs_G_updateratio - floor # e.g. 0.7
        disc_iterations = floor + np.random.choice([1, 0], p=[decimal_part, 1-decimal_part])
        return disc_iterations
    
    def train_gan(self, train_dataloader: DataLoader, n_batches: int, show_images_every_n: int, disc_lr=0.0002, gen_lr=0.001, D_vs_G_updateratio=1.75) -> None:
        disc_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=disc_lr)
        gen_optimizer = torch.optim.AdamW(self.generator.parameters(), lr=1e-3)
        batch_size = train_dataloader.batch_size
        for n in range(n_batches):
            print(f'\t Batch {n+1}/{n_batches}')
            # Discriminator training
            D_vs_G_iterations = self.compute_D_vs_G_iterations(D_vs_G_updateratio)
            for _ in range(D_vs_G_iterations):
                disc_optimizer.zero_grad()
                (real_images, real_labels) = self.make_real_images(train_dataloader, also_create_labels=True)
                (fake_images, fake_labels) = self.make_fake_images(n_examples=batch_size, also_create_labels=True)
                (images, labels) = self.stack_real_and_fake(real_images, fake_images, real_labels, fake_labels)
                logits = self.discriminator(images)
                loss = self.bce_loss(logits=logits, labels=labels, use_pytorch_loss=True)
                accuracy = self.discriminator.compute_accuracy(logits=logits, labels=labels)
                loss.backward()
                disc_optimizer.step()
            self.discriminator.log_stats(loss=loss, accuracy=accuracy)
            
            # Generator training
            gen_optimizer.zero_grad()
            fake_images = self.make_fake_images(n_examples=batch_size)
            real_labels = self.make_real_labels(num_labels=batch_size)
            logits = self.discriminator(fake_images)
            loss = self.bce_loss(logits=logits, labels=real_labels, use_pytorch_loss=True)
            loss.backward()
            gen_optimizer.step()
            self.generator.log_stats(loss=loss)
            
            # Evaluation
            if (n+1) % show_images_every_n == 0:
                self.generator.evaluate_with_image_grid(n_images=100, epoch=n)

