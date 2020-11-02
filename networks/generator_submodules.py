#%%

from typing import Tuple
import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable

from utilities.layers import Layers
from .attention import AttentionModule


class GenInitialStage(nn.Module):
    """
    Initial stage network
        - Takes noise vector and sentence vectors
        - Concatenates noise and sentence vectors
        - Pass through Linear layer and reshape
        - Upsample 4 times to a 64x64 image
    """
    def __init__(self, gf_dim: int, z_dim: int, cond_dim: int):
        """
        Params:
            gf_dim: base number of generator features
            z_dim: noise vector dimensions
            cond_dim: VAE embedding dimensions
        """
        super().__init__()
        self.gf_dim = gf_dim
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.define_module()

    def define_module(self):
        ng, nz, nc = self.gf_dim, self.z_dim, self.cond_dim
        self.fc = nn.Sequential(
            nn.Linear(in_features=nz+nc, out_features=ng*4*4*2, bias=False),
            nn.BatchNorm1d(num_features=ng*4*4*2),
            Layers.GLU()
            )
        # Upsample 4 times
        self.upsample1 = Layers.upBlock(ng, ng//2)
        self.upsample2 = Layers.upBlock(ng//2, ng//4)
        self.upsample3 = Layers.upBlock(ng//4, ng//8)
        self.upsample4 = Layers.upBlock(ng//8, ng//16)

    def forward(self, noise: Tensor, condition: Tensor) -> Tensor:
        """
        Parameters:
            noise:      (batch_size, noise_dim)
            condition:   (batch_size, cond_dim)

        Returns:
            Tensor:     (batch_size, gf_dim/16, 64, 64)
        """
        # Concatenate noise and global sentence vectors
        noise_condition = torch.cat((noise, condition), 1)
        # Pass through linear and reshape
        X = self.fc(noise_condition)
        X = X.view(-1, self.gf_dim, 4, 4)       # (batch, gf_dim, 4, 4)
        # Upsample
        X = self.upsample1(X)                   # (batch, gf_dim, 8, 8)
        X = self.upsample2(X)                   # (batch, gf_dim, 16, 16)
        X = self.upsample3(X)                   # (batch, gf_dim, 32, 32)
        X = self.upsample4(X)                   # (batch, gf_dim, 64, 64)
        return X


class GenNextStage(nn.Module):
    """
    Intermediate stage network
        - Takes image vector and word vectors
        - Computes attention alignment between them
        - Concatenates weighted context onto the back
        - Use Residual block to merge their features
        - Upsample to 2 times the original height/width
    """
    def __init__(self, gf_dim: int, emb_dim: int, num_residual_blocks: int):
        """
        Params:
            gf_dim: Number of generator features
            emb_dim: caption word embedding dimensions
            num_residual_blocks: how many Residual blocks to merge attention + images
        """
        super().__init__()
        self.gf_dim = gf_dim
        self.emb_dim = emb_dim
        self.num_residual_blocks = num_residual_blocks
        self.define_module()

    def define_module(self):
        self.attention = AttentionModule(nc_in=self.gf_dim, emb_dim=self.emb_dim)
        self.residual = self._make_layer(block=Layers.ResBlock, channel_num=self.gf_dim*2)
        self.upsample = Layers.upBlock(self.gf_dim*2, self.gf_dim)

    def _make_layer(self, block: Layers.ResBlock, channel_num: int):
        layers = []
        for i in range(self.num_residual_blocks):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def forward(self, images: Tensor, word_embs: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Params:
            images  (query):        (batch, channels, h, w)
            word_embs (context):    (batch, emb_dim, seq_len)
            mask:                   (batch, seq_len)

        Returns a Tuple of Tensor with shape:
            images:                (batch, channels, 2h, 2w)
            attn:                  (batch, seq_len, h, w)
        """
        self.attention.apply_mask(mask=mask)
        (context, attn) = self.attention(images, word_embs)
        # concatenate image and context
        image_context = torch.cat((images, context), 1)
        # Residual and upsample to next level
        X = self.residual(image_context)
        X = self.upsample(X)
        return (X, attn)


class GenMakeImage(nn.Module):
    """
    Makes an image by compressing channels from N -> 3 (RGB)
    Also applies Tanh activation (normalize output to [-1, 1] range)
    """
    def __init__(self, gf_dim: int):
        """
        Params:
            gf_dim: Number of generator features
        """
        super().__init__()
        self.gf_dim = gf_dim
        self.img = nn.Sequential(
            Layers.conv3x3(in_planes=gf_dim, out_planes=3),
            nn.Tanh()
        )

    def forward(self, images: Tensor) -> Tensor:
        X = self.img(images)
        return X


class VarAutoEncoder(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self, emb_dim: int, cond_dim=100):
        super().__init__()
        self.emb_dim = emb_dim
        self.cond_dim = cond_dim
        self.fc = nn.Linear(emb_dim, cond_dim * 4, bias=True)
        self.relu = Layers.GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.cond_dim]
        logvar = x[:, self.cond_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding: Tensor):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar