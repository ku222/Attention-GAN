
from typing import List
import torch
from torch.nn import BCELoss, Module
from torch import Tensor


class DiscLoss:
    def __init__(self):
        pass
    
    def make_labels_for_real_imgs(self, num_labels: int, label_smooth=0.8) -> Tensor:
        """Creates labels for real images"""
        labels = torch.FloatTensor(num_labels).uniform_(label_smooth, 1.0)
        return labels.cuda()

    def make_labels_for_fake_imgs(self, num_labels: int) -> Tensor:
        """Creates labels for fake images"""
        labels = torch.FloatTensor(num_labels).uniform_(0.0, 0.0)
        return labels.cuda()
    
    def get_loss(self, discriminator: Module, fake_images: Tensor, real_images: Tensor) -> Tensor:
        raise NotImplementedError


class StandardDiscLoss(DiscLoss):
    def __init__(self):
        super().__init__()
        self.criterion = BCELoss()

    def get_loss(self, discriminator: Module, fake_images: Tensor, real_images: Tensor) -> Tensor:
        """
        Computes loss given a discriminator, fake images and real images
        """
        # Evaluate on fake images
        (batch_size, _, _, _) = fake_images.shape
        logits = discriminator(fake_images)
        labels = self.make_labels_for_fake_imgs(num_labels=batch_size)
        loss_fake: Tensor = self.criterion(logits, labels)
        # Evaluate on real images
        # if self.instance_noise:
        #     gaussnoise = torch.empty_like(real_images).normal_(mean=0.0, std=noise_std)
        logits = discriminator(real_images)
        labels = self.make_labels_for_real_imgs(num_labels=batch_size)
        loss_real: Tensor = self.criterion(logits, labels)
        loss = (loss_fake + loss_real) / 2
        return loss
    
    
class NonSaturatingDiscLoss(DiscLoss):
    def __init__(self):
        super().__init__()
    
    def get_loss(self, discriminator: Module, fake_images: Tensor, real_images: Tensor) -> Tensor:
        DX_score = discriminator(real_images) # D(x)
        DG_score = discriminator(fake_images) # D(G(z))
        loss = torch.sum(
            -torch.mean(
                torch.log(DX_score + 1e-8) + torch.log(1 - DG_score + 1e-8)
                )
            )
        return loss