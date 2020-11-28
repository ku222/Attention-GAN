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
from losses.disc_loss import DiscLoss
from losses.gen_loss import GenLoss
from losses.words_loss import WordsLoss
from losses.sentence_loss import SentenceLoss
from losses.KL_loss import KL_loss
# Dataloaders
from data.birds import BirdsDataset
from data.preprocessor import DatasetPreprocessor
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
SEQ_LEN = 12
BATCH_SIZE = 16
GEN_LR = 0.0002
DISC_LR = 0.0002
LABEL_SMOOTH = 0.9
INSTANCE_NOISE = True

# GAMMA
GAMMA1 = 4.0
GAMMA2 = 5.0
GAMMA3 = 10.0
WLAMBDA = 5.0
SLAMBDA = 5.0

# Datasets
DATASET = BirdsDataset(max_images=9999)
PREPROCESSOR = DatasetPreprocessor()
DATALOADER = PREPROCESSOR.preprocess(DATASET, maxlen=SEQ_LEN, batch_size=BATCH_SIZE)


#%%
# Networks/Modules
DEVICE = torch.device('cuda')
GENERATOR = Generator(gf_dim=GF_DIM, emb_dim=EMB_DIM, z_dim=Z_DIM, cond_dim=COND_DIM)
GENERATOR.to(DEVICE)
DISCRIMINATORS = [Disc64(DF_DIM), Disc128(DF_DIM), Disc256(DF_DIM)]
for d in DISCRIMINATORS:
    d.to(DEVICE)
RNN = RNNEncoder(vocabsize=PREPROCESSOR.n_words, nhidden=EMB_DIM)
RNN.to(DEVICE)
CNN = CNNEncoder(out_dim=EMB_DIM)
CNN.to(DEVICE)

# Losses
WORDSLOSS = WordsLoss(DEVICE, GAMMA1, GAMMA2, GAMMA3, WLAMBDA)
SENTLOSS = SentenceLoss(DEVICE, GAMMA3, SLAMBDA)
GENLOSS = GenLoss(DEVICE)
DISCLOSS = DiscLoss(DEVICE, LABEL_SMOOTH, INSTANCE_NOISE)

# Training items
GEN_OPTIM = Adam(GENERATOR.parameters(), lr=GEN_LR, betas=(0.5, 0.999))
DISC_OPTIMS = [Adam(disc.parameters(), lr=DISC_LR, betas=(0.5, 0.999))
                   for disc in DISCRIMINATORS]


class GanTrainer(ModelTrainer):
    def __init__(self):
        super().__init__()
        self._load_weights(modules=[CNN, RNN])
        RNN.freeze_all_weights()
        CNN.freeze_all_weights()
        self.modules = [GENERATOR] + DISCRIMINATORS
        self.gen_losses = []
        self.real_logits = []
        self.fake_logits = []

    def _make_mask(self, lengths: Tensor) -> Tensor:
        maxlen = max(lengths)
        masks = [[1]*leng + [0]*(maxlen-leng)
                 for leng in lengths]
        return torch.LongTensor(masks)

    def make_images(self, captions: List[str]) -> None:
        user_dataset = PREPROCESSOR.preprocess_user(captions)
        for batch in user_dataset:
            batch = [t.to(DEVICE) for t in batch]
            (words, lengths) = batch
            masks = self._make_mask(lengths).to(DEVICE)
            # Embed words
            hiddencell = RNN.init_hidden_cell_states(batch_size=BATCH_SIZE)
            (word_embs, sent_embs) = RNN(words, lengths, hiddencell)
            # Make noise, concatenate with sentence embedding
            noise = torch.randn((len(words), Z_DIM)).to(DEVICE)
            # Make images
            (fake_imgs, attn_maps) = GENERATOR(noise=noise, sent_emb=sent_embs, word_embs=word_embs, mask=masks)
            self._evaluate_with_img_grid(fake_imgs)

    @timer
    def train_gan(self, epochs=200, plot_loss_every=1, image_grid_every=1, weights_every=1):
        match_labels = self._make_match_labels(BATCH_SIZE).to(DEVICE)
        epbar = tqdm(total=epochs)
        for e in range(1, epochs+1):
            print('='*10 + f" Epoch {e+1} " + '='*10)
            ### Create progress bar
            bpbar = tqdm(total=len(DATALOADER))
            for (b, batch) in enumerate(DATALOADER):
                batch = [t.to(DEVICE) for t in batch]
                (words, lengths, class_ids, img64, img128, img256) = batch
                if min(lengths) < 2 or len(words) < BATCH_SIZE:
                    continue
                class_ids = class_ids.detach().cpu().numpy()
                masks = self._make_mask(lengths).to(DEVICE)
                #################### Embed words ####################
                hiddencell = RNN.init_hidden_cell_states(batch_size=BATCH_SIZE)
                (word_embs, sent_embs) = RNN(words, lengths, hiddencell)
                #################### Make images ####################
                noise = self._make_noise(batch_size=BATCH_SIZE, z_dim=Z_DIM).to(DEVICE)
                (fake_imgs, attn_maps, mu, logvar) = GENERATOR(noise=noise, sent_emb=sent_embs, word_embs=word_embs, mask=masks)
                real_imgs = [img64, img128, img256]
                #################### Get discriminator loss ####################
                for (i, (disc, optim, fake_img, real_img)) in enumerate(zip(DISCRIMINATORS, DISC_OPTIMS, fake_imgs, real_imgs)):
                    optim.zero_grad()
                    (loss, real_acc, fake_acc) = DISCLOSS.get_loss(disc, fake_img, real_img)
                    loss.backward(retain_graph=True)
                    optim.step()
                    if i == 2:
                        self.real_logits.append(real_acc)
                        self.fake_logits.append(fake_acc)
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
                #################### KL Loss ####################
                kl_loss = KL_loss(mu=mu, logvar=logvar)
                total_genloss += kl_loss
                #################### Backprop ####################
                total_genloss.backward()
                GEN_OPTIM.step()
                #################### Log stats ####################
                self.gen_losses.append(total_genloss.item())
                bpbar.update()
            # After each epoch
            if (e+1) % image_grid_every == 0:
                self._plot_image_grid(fake_imgs, epoch=e)
            if (e+1) % weights_every == 0:
                self._save_weights(self.modules)
            if (e+1) % plot_loss_every == 0:
                self._plot_history(self.gen_losses, name='loss_hist', epoch=e, window_size=100*e)
                self._plot_history(self.real_logits, name='real_img_logits', epoch=e, window_size=100*e)
                self._plot_history(self.fake_logits, name='fake_img_logits', epoch=e, window_size=100*e)
            # Close progress bar
            bpbar.close()
            epbar.update()


trainer = GanTrainer()
trainer.train_gan()


#%%