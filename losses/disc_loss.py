
from typing import List
import torch
from torch.nn import BCELoss, Module
from torch import Tensor


class DiscLoss:
    def __init__(self, device: torch.device, label_smooth=0.9, instance_noise=True):
        self.criterion = BCELoss()
        self.device = device
        self.label_smooth = label_smooth
        self.instance_noise = instance_noise

    def make_labels_for_real_imgs(self, num_labels: int) -> Tensor:
        """Creates labels for real images"""
        labels = torch.FloatTensor(num_labels).uniform_(self.label_smooth, self.label_smooth)
        return labels.to(self.device)

    def make_labels_for_fake_imgs(self, num_labels: int) -> Tensor:
        """Creates labels for fake images"""
        labels = torch.FloatTensor(num_labels).uniform_(0.0, 0.0)
        return labels.to(self.device)

    def get_loss(self, discriminator: Module, fake_images: Tensor, real_images: Tensor, noise_std: float) -> Tensor:
        """
        Computes loss given a discriminator, fake images and real images
        """
        # Evaluate on fake images
        (batch_size, _, _, _) = fake_images.shape
        logits = discriminator(fake_images)
        fake_acc = torch.mean(logits).item()  # Get average fake acc
        labels = self.make_labels_for_fake_imgs(num_labels=batch_size)
        loss_fake: Tensor = self.criterion(logits, labels)
        # Evaluate on real images
        if self.instance_noise:
            gaussnoise = torch.empty_like(real_images).normal_(mean=0.0, std=noise_std)
        logits = discriminator(real_images + gaussnoise)
        real_acc = torch.mean(logits).item()    # Get average real acc
        labels = self.make_labels_for_real_imgs(num_labels=batch_size)
        loss_real: Tensor = self.criterion(logits, labels)
        loss = (loss_fake + loss_real) / 2
        return (loss, real_acc, fake_acc)