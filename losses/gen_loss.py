
from typing import List
import torch
from torch.nn import BCELoss, Module
from torch import Tensor


class GenLoss:
    def __init__(self):
        pass
    
    def make_labels_for_real_imgs(self, num_labels: int) -> Tensor:
        """Creates labels for real images"""
        labels = torch.FloatTensor(num_labels).uniform_(1.0, 1.0)
        return labels.cuda()
    
    def get_loss(self, discriminator: Module, fake_images: Tensor) -> Tensor:
        raise NotImplementedError


class StandardGenLoss(GenLoss):
    def __init__(self):
        super().__init__()
        self.criterion = BCELoss()

    def get_loss(self, discriminator: Module, fake_images: Tensor) -> Tensor:
        """
        Computes loss given a discriminator and fake images
        """
        # Evaluate on fake images
        (batch_size, _, _, _) = fake_images.shape
        logits = discriminator(fake_images)
        labels = self.make_labels_for_real_imgs(num_labels=batch_size)
        loss = self.criterion(logits, labels)
        return loss


class NonSaturatingGenLoss(GenLoss):
    def __init__(self):
        super().__init__()

    def get_loss(self, discriminator: Module, fake_images: Tensor) -> Tensor:
        """Computes loss given a discriminator and fake images"""
        # Evaluate on fake images
        logits = discriminator(fake_images)
        loss = -torch.mean(torch.log(logits + 1e-8))
        return loss