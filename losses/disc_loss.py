
from typing import List
import torch
from torch.nn import BCELoss, Module
from torch import Tensor


class DiscLoss:
    def __init__(self, device: torch.device):
        self.criterion = BCELoss()
        self.device = device

    def make_labels_for_real_imgs(self, num_labels: int) -> Tensor:
        """Creates labels for real images"""
        labels = torch.FloatTensor(num_labels).uniform_(0.9, 0.9)
        return labels.to(self.device)

    def make_labels_for_fake_imgs(self, num_labels: int) -> Tensor:
        """Creates labels for fake images"""
        labels = torch.FloatTensor(num_labels).uniform_(0.1, 0.1)
        return labels.to(self.device)

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
        logits = discriminator(real_images)
        labels = self.make_labels_for_real_imgs(num_labels=batch_size)
        loss_real: Tensor = self.criterion(logits, labels)
        loss = (loss_fake + loss_real) / 2
        return loss