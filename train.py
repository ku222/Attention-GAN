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
from tqdm import tqdm
# Modules/networks
from networks.generator import Generator
from networks.discriminators import Disc64, Disc128, Disc256
from networks.attention import AttentionModule, func_attention
from networks.rnn_encoder import RNNEncoder
from networks.cnn_encoder import CNNEncoder
# Losses
from losses.disc_loss import DiscLoss, NonSaturatingDiscLoss, StandardDiscLoss
from losses.gen_loss import GenLoss, NonSaturatingGenLoss, StandardGenLoss
from losses.words_loss import WordsLoss
from losses.sentence_loss import SentenceLoss
from losses.KL_loss import KL_loss
# Dataloaders
from data.bedrooms import Dataset
# utilities
from utilities.decorators import timer
# Trainers
from trainers.trainer import ModelTrainer

# Dimensions
GF_DIM = 32
DF_DIM = 64
EMB_DIM = 256
COND_DIM = 100
Z_DIM = 100
SEQ_LEN = 5
BATCH_SIZE = 16
GEN_LR = 0.0002
DISC_LR = 0.0002

# GAMMA
GAMMA1 = 4.0
GAMMA2 = 5.0
GAMMA3 = 10.0
WLAMBDA = 5.0
SLAMBDA = 5.0

# Datasets
DATASET = Dataset(max_images=8000)
DATASET.load_captions()
DATALOADER = DATASET.make_dataloaders(batch_size=BATCH_SIZE, max_seqlen=8)


#%%
# Networks/Modules
DEVICE = torch.device('cuda')
GENERATOR = Generator(gf_dim=GF_DIM, emb_dim=EMB_DIM, z_dim=Z_DIM, cond_dim=COND_DIM)
GENERATOR.cuda()
DISCRIMINATORS = [Disc64(DF_DIM), Disc128(DF_DIM), Disc256(DF_DIM)]
for d in DISCRIMINATORS:
    d.cuda()
RNN = RNNEncoder(vocabsize=DATASET.vocab.n_words, nhidden=EMB_DIM)
RNN.cuda()
CNN = CNNEncoder(out_dim=EMB_DIM)
CNN.cuda()

# Losses
WORDSLOSS = WordsLoss(DEVICE, GAMMA1, GAMMA2, GAMMA3, WLAMBDA)
SENTLOSS = SentenceLoss(DEVICE, GAMMA3, SLAMBDA)
GENLOSS = NonSaturatingGenLoss()
DISCLOSS = NonSaturatingDiscLoss()

# Training items
GEN_OPTIM = Adam(GENERATOR.parameters(), lr=GEN_LR, betas=(0.5, 0.999))
DISC_OPTIMS = [Adam(disc.parameters(), lr=DISC_LR, betas=(0.5, 0.999))
                   for disc in DISCRIMINATORS]


#%%

class GanTrainer(ModelTrainer):
    def __init__(self):
        super().__init__()
        self._load_weights(modules=[CNN, RNN])
        RNN.freeze_all_weights()
        CNN.freeze_all_weights()
        self.modules = [GENERATOR] + DISCRIMINATORS
        self.g_losses = []
        self.d_losses = []

    def _make_mask(self, lengths: Tensor) -> Tensor:
        maxlen = max(lengths)
        masks = [[1]*leng + [0]*(maxlen-leng)
                 for leng in lengths]
        return torch.LongTensor(masks).cuda()

    @timer
    def train_gan(self, epochs=100):
        match_labels = self._make_match_labels(BATCH_SIZE)
        fixed_input = self._make_noise(batch_size=BATCH_SIZE, z_dim=Z_DIM)
        epbar = tqdm(total=epochs)
        for e in range(1, epochs+1):
            print('='*10 + f" Epoch {e+1} " + '='*10)
            for batch in tqdm(DATALOADER):
                batch = [t.cuda() for t in batch]
                (words, lengths, class_ids, img64, img128, img256) = batch
                if min(lengths) < 2 or len(words) < BATCH_SIZE:
                    continue
                class_ids = class_ids.detach().cpu().numpy()
                masks = self._make_mask(lengths)
                #################### Embed words ####################
                (word_embs, sent_embs) = RNN(words, lengths)
                #################### Make images ####################
                noise = self._make_noise(batch_size=BATCH_SIZE, z_dim=Z_DIM)
                (fake_imgs, attn_maps, mu, logvar) = GENERATOR(noise=noise, sent_emb=sent_embs, word_embs=word_embs, mask=masks)
                real_imgs = [img64, img128, img256]
                #################### Get discriminator loss ####################
                for (i, (disc, optim, fake_img, real_img)) in enumerate(zip(DISCRIMINATORS, DISC_OPTIMS, fake_imgs, real_imgs)):
                    optim.zero_grad()
                    loss = DISCLOSS.get_loss(disc, fake_img, real_img)
                    loss.backward(retain_graph=True)
                    optim.step()
                    # If final resolution, log stats
                    if i == 2:
                        self.d_losses.append(loss.item())
                #################### Get generator loss ####################
                GEN_OPTIM.zero_grad()
                total_genloss = 0
                for (i, (disc, fake_img)) in enumerate(zip(DISCRIMINATORS, fake_imgs)):
                    loss = GENLOSS.get_loss(disc, fake_img)
                    total_genloss += loss
                    #################### DAMSM loss if final resolution ####################
                    if i == 2:
                        (region_feat, global_feat) = CNN(fake_img)
                        (wloss, _) = WORDSLOSS.get_loss(region_feat, word_embs, match_labels, lengths, class_ids)
                        sloss = SENTLOSS.get_loss(global_feat, sent_embs, match_labels, class_ids)
                        total_genloss += (wloss + sloss)
                        self.g_losses.append(loss.item())
                #################### KL Loss ####################
                kl_loss = KL_loss(mu=mu, logvar=logvar)
                total_genloss += kl_loss
                #################### Backprop ####################
                total_genloss.backward()
                GEN_OPTIM.step()
                #################### Log stats ####################
            # After each epoch
            with torch.no_grad():
                (fake_imgs, _, _, _) = GENERATOR(noise=fixed_input, sent_emb=sent_embs, word_embs=word_embs, mask=masks)
                fake_imgs = self._denormalise_multiple(fake_imgs)
                self._plot_image_grid(fake_imgs, epoch=e)
                self._save_weights(self.modules)
                self._plot_history([self.g_losses, self.d_losses], name='loss_hist', epoch=e, window_size=50*e)
                epbar.update()


trainer = GanTrainer()



#%%

trainer.train_gan(epochs=75)

#%%
