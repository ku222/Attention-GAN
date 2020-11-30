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
from torch.nn import Module
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.nn import Upsample
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
# Modules/networks
from networks.attention import AttentionModule, func_attention
from networks.rnn_encoder import RNNEncoder
from networks.cnn_encoder import CNNEncoder
# Losses
from losses.disc_loss import DiscLoss
from losses.gen_loss import GenLoss
from losses.words_loss import WordsLoss
from losses.sentence_loss import SentenceLoss
# Dataloaders
from data.preprocessor import DatasetPreprocessor, BuildingsDataset
# utilities
from utilities.decorators import timer
from utilities.training import Training
# Trainers
from trainers.trainer import ModelTrainer

from data.bedrooms import Dataset, HierarchicalClusterer
from networks.cnn_embedder import ImageEmbedder

# Dimensions
GF_DIM = 32
DF_DIM = 64
EMB_DIM = 256
Z_DIM = 100
BATCH_SIZE = 64
LR = 0.002
RNN_GRAD_CLIP = 0.25


DATASET = Dataset(rootdir=r'D:\GAN\buildingsDataset', max_images=99999)

def reset_captions_and_class_ids(dataset: Dataset):
    for img in dataset.images:
        img.caption = []
        img.class_id = None

reset_captions_and_class_ids(DATASET)
hc = HierarchicalClusterer()
hc.cluster(DATASET, latent_dims=128, min_clusters=5, max_vocab_size=1000, method='agglomerative_complete')

#%%
DATALOADER = DATASET.make_dataloaders(BATCH_SIZE)

#%%

# Networks/Modules
DEVICE = torch.device('cuda')
RNN = RNNEncoder(vocabsize=DATASET.vocab.n_words, nhidden=EMB_DIM).to(DEVICE)
CNN = CNNEncoder(out_dim=EMB_DIM).to(DEVICE)
# Training items
params = list(RNN.parameters())
for l in CNN.parameters():
    if l.requires_grad:
        params.append(l)
OPTIMIZER = Adam(params, lr=LR, betas=(0.5, 0.999))


#%%

class DAMSMTrainer(ModelTrainer):
    def __init__(self):
        super().__init__()
        self.loss_history = []
        self.modules = [RNN, CNN]
    
    # @timer
    # def populate_attnmaps(self) -> None:
    #     with torch.no_grad():
    #         match_labels = self._make_match_labels(BATCH_SIZE).to(DEVICE)
    #         for (b, batch) in tqdm(enumerate(DATALOADER)):
    #             batch = [t.to(DEVICE) for t in batch]
    #             (img_ids, captions, lengths, class_ids, img64, img128, img256) = batch
    #             if min(lengths) < 2 or len(captions) < BATCH_SIZE:
    #                 continue
    #             # CNN encode image
    #             class_ids = class_ids.detach().cpu().numpy()
    #             (words_features, sent_code) = CNN(img256)
    #             (nef, att_sze) = words_features.size(1), words_features.size(2)
    #             # RNN
    #             hiddencell = RNN.init_hidden_cell_states(BATCH_SIZE)
    #             (word_embs, sent_embs) = RNN(captions, lengths, hiddencell)
    #             # words Loss
    #             (wloss, attnmaps) = WordsLoss(DEVICE).get_loss(words_features, word_embs, match_labels, lengths, class_ids)
    #             # attn = List[1 x seq_len x 17 x 17]
    #             ## Populate Attentionmaps
    #             img_ids = img_ids.tolist()
    #             for (imgid, attn) in zip(img_ids, attnmaps):
    #                 birdImage = DATASET.id2image[imgid]
    #                 birdImage.attnmap = attn.detach().cpu()

    @timer
    def pretrain_damsm(self, epochs=30):
        match_labels = self._make_match_labels(BATCH_SIZE).to(DEVICE)
        for e in range(1, epochs+1):
            print('='*10 + f" Epoch {e} " + '='*10)
            for batch in tqdm(DATALOADER):
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
                (word_embs, sent_embs) = RNN(captions, lengths)
                # words Loss
                OPTIMIZER.zero_grad()
                (wloss, attn) = WordsLoss(DEVICE).get_loss(words_features, word_embs, match_labels, lengths, class_ids)
                sloss = SentenceLoss(DEVICE).get_loss(sent_code, sent_embs, match_labels, class_ids)
                loss = (wloss + sloss)
                loss.backward()
                clip_grad_norm_(RNN.parameters(), RNN_GRAD_CLIP)
                OPTIMIZER.step()
                self.loss_history.append(round(loss.item(), 3))
                
            # Snapshots
            self._save_weights(self.modules)
            self._plot_history(self.loss_history, epoch=e)


trainer = DAMSMTrainer()

#%%

trainer.pretrain_damsm()

#%%
## 44 seconds for 1000 images

trainer.populate_attnmaps()

#%%

img = DATASET.images[25]
img.view_image()

#%%

for (i, token) in enumerate(img.caption.replace(', ', ' , ').split()):
    print(i, token)
    
#%%

img.view_attention_map(1)