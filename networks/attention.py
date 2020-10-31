
import random
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from math import sqrt

from utilities.layers import Layers


class AttentionModule(nn.Module):
    def __init__(self, nc_in: int, emb_dim: int):
        super().__init__()
        self.nc_in = nc_in
        self.conv1 = Layers.conv1x1(in_planes=emb_dim, out_planes=nc_in)
        self.mask = None
        
    def apply_mask(self, mask: Tensor) -> None:
        self.mask = mask

    def forward(self, images: Tensor, words: Tensor, scaled=True) -> Tuple[Tensor, Tensor]:
        """
        Applies multiplicative dot-product attention between images pixels and word vectors.
        If scaled==True, then will scale dot-products by the dimensionality of our image/word
        vectors after dot-products are computed, just before the softmax operation.

        Arguments:
            - images with shape [batch, nc_in, h, w]
            - words with shape [batch, emb_dim, seq_len]
            - mask with shape [batch, seq_len]

        Masking Operation:
            Mask is applied prior to softmaxing to compute attention scores.
            Mask should have 0's wherever words are to be ignored by attention.
            For example, the [2, 5] mask below means that we ignore
            the last word in the first observation, and the last 3 words
            in the second observation.
            [[1,1,1,1,0],
             [1,1,0,0,0]]
        """
        (batch, nc_in, h, w) = images.shape
        (batch, emb_dim, seq_len) = words.shape
        (batch, seq_len) = self.mask.shape

        # Reshape words
        words = words.unsqueeze(3) # -> (batch, emb_dim, seq_len, 1)
        words = self.conv1(words) # -> (batch, nc_in, seq_len, 1)
        words = words.squeeze(3) # -> (batch, nc_in, seq_len)

        # Reshape images into one long matrix
        images = images.view(batch, nc_in, h*w)
        images = images.transpose(1, 2).contiguous() # -> (batch, h*w, nc_in)

        # Compute attention
        attn = torch.bmm(images, words) # -> (batch, h*w, seq_len)
        # Scale attention if required
        attn = attn * (1 / sqrt(nc_in)) if scaled else attn
        # Reshape to a long matrix to make softmax op. easier
        attn = attn.view(batch*h*w, seq_len) # -> (batch*h*w, seq_len)
        # Apply mask before softmaxing
        mask = torch.repeat_interleave(input=self.mask, repeats=h*w, dim=0) # -> (batch*h*w, seq_len)
        attn = attn.masked_fill(mask==0, -float('inf')) # -> (batch*h*w, seq_len)
        # Softmax to product scores
        attn = torch.softmax(attn, dim=1)
        attn = attn.view(batch, h*w, seq_len) # revert back
        attn = attn.transpose(1, 2).contiguous() # -> (batch, seq_len, h*w)

        # Compute weighted word vectors
        weighted_words = torch.bmm(words, attn) # -> (batch, nc_in, h*w)

        # Return weighted words and attn
        return (
            weighted_words.view(batch, nc_in, h, w),
            attn.view(batch, seq_len, h, w)
        )

    @staticmethod
    def create_random_mask(batch_size=8, seq_len=15) -> Tensor:
        '''Creates random mask for testing purposes'''
        tensors = [
            torch.LongTensor(
                sorted([1 for _ in range(random.randint(1, seq_len))], reverse=True)
            )
            for i in range(batch_size)
        ]
        return pad_sequence(tensors, batch_first=True, padding_value=0)