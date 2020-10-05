
#%%
import random
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from math import sqrt

from modelutils import Utils

class AttentionModule(nn.Module):
    def __init__(self, nc_in: int, emb_dim: int):
        super().__init__()
        self.nc_in = nc_in
        self.conv1 = Utils.conv1x1(in_channels=emb_dim, out_channels=nc_in)

    def forward(self, images: Tensor, words: Tensor, mask: Tensor, scaled=True) -> Tuple[Tensor, Tensor]:
        """
        Applies multiplicative dot-product attention between images pixels and word vectors.
        If scaled==True, then will scale dot-products by the dimensionality of our image/word
        vectors after dot-products are computed, just before the softmax operation.

        Arguments:
            - images with shape [batch, nc_in, h, w]
            - words with shape [batch, max_words, emb_dim]
            - mask with shape [batch, max_words]

        Masking Operation:
            Mask is applied prior to softmaxing to compute attention scores.
            Mask should have 1's wherever words are to be ignored by attention.
            For example, the [2, 5] mask below means that we ignore
            the last word in the first observation, and the last 3 words
            in the second observation.
            [[0,0,0,0,1],
             [0,0,1,1,1]]
        """
        (batch, nc_in, h, w) = images.shape
        (batch, max_words, emb_dim) = words.shape
        (batch, max_words) = mask.shape

        # Reshape words
        words = words.transpose(1, 2).contiguous().unsqueeze(3) # -> (batch, emb_dim, max_words, 1)
        words = self.conv1(words) # -> (batch, nc_in, max_words, 1)
        words = words.squeeze(3) # -> (batch, nc_in, max_words)

        # Reshape images into one long matrix
        images = images.view(batch, nc_in, h*w)
        images = images.transpose(1, 2).contiguous() # -> (batch, h*w, nc_in)

        # Compute attention
        attn = torch.bmm(images, words) # -> (batch, h*w, max_words)
        # Scale attention if required
        attn = attn * (1 / sqrt(nc_in)) if scaled else attn
        # Reshape to a long matrix to make softmax op. easier
        attn = attn.view(batch*h*w, max_words) # -> (batch*h*w, max_words)
        # Apply mask before softmaxing
        mask = torch.repeat_interleave(input=mask, repeats=h*w, dim=0) # -> (batch*h*w, max_words)
        attn = attn.masked_fill(mask, -float('inf')) # -> (batch*h*w, max_words)
        # Softmax to product scores
        attn = torch.softmax(attn, dim=1)
        attn = attn.view(batch, h*w, max_words) # revert back
        attn = attn.transpose(1, 2).contiguous() # -> (batch, max_words, h*w)

        # Compute weighted word vectors
        weighted_words = torch.bmm(words, attn) # -> (batch, nc_in, h*w)

        # Return weighted words and attn
        return (
            weighted_words.view(batch, nc_in, h, w),
            attn.view(batch, max_words, h, w)
        )

    @staticmethod
    def create_random_mask(batch_size=8, max_words=15) -> Tensor:
        '''Creates random mask for testing purposes'''
        tensors = [
            torch.tensor(
                sorted([0 for _ in range(random.randint(1, max_words))], reverse=True)
            )
            for i in range(batch_size)
        ]
        return pad_sequence(tensors, batch_first=True, padding_value=1)