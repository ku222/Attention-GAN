#%%

from typing import Tuple, List
import torch
from torch import nn
from torch import Tensor

from .generator_submodules import GenInitialStage, GenNextStage, GenMakeImage
from utilities.decorators import timer


class Generator(nn.Module):
    def __init__(self, gf_dim: int, emb_dim: int, z_dim: int):
        """
        Params:
            gf_dim: base number of generator features
            emb_dim: Embedding features for text
            z_dim: Noise vector dims
        """
        super().__init__()
        self.gf_dim = gf_dim
        self.emb_dim = emb_dim
        self.z_dim = z_dim
        # First stage
        self.gen1 = GenInitialStage(gf_dim=gf_dim*16, z_dim=z_dim, emb_dim=emb_dim)
        self.img_out1 = GenMakeImage(gf_dim=gf_dim)
        # Second stage
        self.gen2 = GenNextStage(gf_dim=gf_dim, emb_dim=emb_dim, num_residual_blocks=2)
        self.img_out2 = GenMakeImage(gf_dim=gf_dim)
        # Final stage
        self.gen3 = GenNextStage(gf_dim=gf_dim, emb_dim=emb_dim, num_residual_blocks=2)
        self.img_out3 = GenMakeImage(gf_dim=gf_dim)

    @timer
    def forward(self, noise: Tensor, sent_emb: Tensor, word_embs: Tensor, mask: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Params:
            noise:      (batch, z_dim)
            sent_emb:   (batch, emb_dim)
            word_embs:  (batch, emb_dim, seq_len)
            mask:       (batch, seq_len)
            
        Returns a Tuple of Lists each containing 3 Tensors:
            fake_imgs:  3 tensors of fake images (batch, 3, 64/128/256, 64/128/256)
            attn_maps:  3 tensors of attention maps (batch, seq_len, 64/128/256, 64/128/256)
        """
        fake_imgs, attn_maps = [], []
        # First stage
        images = self.gen1(noise, sent_emb)
        fake_img = self.img_out1.forward(images)
        fake_imgs.append(fake_img)
        # Second stage
        (images, attn) = self.gen2(images, word_embs, mask)
        fake_img = self.img_out2(images)
        fake_imgs.append(fake_img)
        attn_maps.append(attn)
        # Final stage
        (images, attn) = self.gen3(images, word_embs, mask)
        fake_img = self.img_out3(images)
        fake_imgs.append(fake_img)
        attn_maps.append(attn)
        # Return tuple
        return (fake_imgs, attn_maps)
    

