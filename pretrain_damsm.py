#%%
# Python
from collections import defaultdict
import math
from datetime import datetime
from typing import List
# 3rd Party
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch import Tensor
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
# Modules/networks
from networks.attention import AttentionModule, func_attention
from networks.rnn_encoder import RNNEncoder
from networks.cnn_encoder import CNNEncoder
# Losses
from losses.disc_loss import DiscLoss
from losses.gen_loss import GenLoss
from losses.words_loss import WordsLoss
# Dataloaders
from data.birds import BirdsDataset
from data.preprocessor import DatasetPreprocessor
# utilities
from utilities.decorators import timer
from utilities.training import Training

# Dimensions
GF_DIM = 32
DF_DIM = 64
EMB_DIM = 256
Z_DIM = 100
SEQ_LEN = 15
BATCH_SIZE = 4
LR = 0.002
RNN_GRAD_CLIP = 0.25


#%%
# Datasets
DATASET = BirdsDataset(max_images=100)
PREPROCESSOR = DatasetPreprocessor()
DATALOADER = PREPROCESSOR.preprocess(DATASET, maxlen=SEQ_LEN, batch_size=BATCH_SIZE)

#%%
# Networks/Modules
DEVICE = torch.device('cuda')
RNN = RNNEncoder(vocabsize=PREPROCESSOR.n_words, nhidden=EMB_DIM).to(DEVICE)
CNN = CNNEncoder(out_dim=EMB_DIM).to(DEVICE)
# Training items
params = list(RNN.parameters())
for l in CNN.parameters():
    if l.requires_grad:
        params.append(l)
OPTIMIZER = Adam(params, lr=LR, betas=(0.5, 0.999))


#%%

class DAMSMTrainer:
    def __init__(self):
        self.losses = []

    def _latest_loss(self) -> str:
        g = f"g: {self.genlogs[-1]}"
        d64 = f"d64: {self.disclogs[0][-1]}"
        d128 = f"d128: {self.disclogs[1][-1]}"
        d256 = f"d256: {self.disclogs[2][-1]}"
        losses = [g, d64, d128, d256]
        return ' | '.join(losses)

    def _make_mask(self, lengths: Tensor) -> Tensor:
        maxlen = max(lengths)
        masks = [[1]*leng + [0]*(maxlen-leng)
                 for leng in lengths]
        return torch.LongTensor(masks)

    def _make_match_labels(self, batch_size: int) -> Variable:
        return Variable(torch.LongTensor(range(batch_size)))
    
    def _make_noise(self, batch_size: int, z_dim: int) -> Variable:
        return Variable(torch.FloatTensor(batch_size, z_dim))
                    
    @timer
    def pretrain_damsm(self, epochs=1):
        match_labels = self._make_match_labels(BATCH_SIZE).to(DEVICE)
        for e in range(epochs):
            for (b, batch) in enumerate(DATALOADER):
                batch = [t.to(DEVICE) for t in batch]
                (captions, lengths, class_ids, img64, img128, img256) = batch
                if min(lengths) < 2 or len(captions) < BATCH_SIZE:
                    continue
                # CNN encode image
                class_ids = class_ids.detach().cpu().numpy()
                (words_features, sent_code) = CNN(img256)
                (nef, att_sze) = words_features.size(1), words_features.size(2)
                # RNN
                hiddencell = RNN.init_hidden_cell_states(BATCH_SIZE)
                (word_embs, sent_embs) = RNN(captions, lengths, hiddencell)
                # words Loss
                OPTIMIZER.zero_grad()
                (loss, attn) = WordsLoss(DEVICE).get_loss(words_features, word_embs, match_labels, lengths, class_ids)
                print('\t', len(attn), attn[0].shape)
                loss.backward()
                clip_grad_norm_(RNN.parameters(), RNN_GRAD_CLIP)
                OPTIMIZER.step()
                # Log stats
                print(f"Loss = {loss.item()}")
                self.losses.append(loss.item())
                break
                
trainer = DAMSMTrainer()

#%%
## 44 seconds for 1000 images
trainer.pretrain_damsm(1)

#%%
from torch.utils.data import DataLoader

a = torch.randn(16, 8)
b = torch.randn(4, 8)
len(a)
#%%
import math
num_images = 13


#%%

Training.plot_history(trainer.losses)

#%%

captions=['all-purpose bill, red crown, blue chest']*16

user_dataset = PREPROCESSOR.preprocess_user(captions, batch_size=8)
for batch in user_dataset:
    batch = [t.to(DEVICE) for t in batch]
    (words, lengths) = batch
    masks = trainer._make_mask(lengths).to(DEVICE)
    # Embed words
    hiddencell = EMBEDDER.init_hidden_cell_states(batch_size=BATCH_SIZE)
    (word_embs, sent_embs) = EMBEDDER(words, lengths, hiddencell)
    # Make noise, concatenate with sentence embedding
    noise = torch.randn((len(words), Z_DIM)).to(DEVICE)
    # Make images
    (fake_imgs, attn_maps) = GENERATOR(noise=noise, sent_emb=sent_embs, word_embs=word_embs, mask=masks)

#%%
img = fake_imgs[2][0].detach().cpu()
plt.imshow(img.permute(1, 2, 0))


#%%
from torch.autograd import Variable
import torch
Variable(torch.LongTensor(range(8)))

int('055')