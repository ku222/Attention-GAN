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
from data.bedrooms import Dataset, CaptionHandler
# utilities
from utilities.decorators import timer
# Trainers
from trainers.trainer import ModelTrainer
from torchvision.utils import make_grid



#%%
# Dimensions
GF_DIM = 32
DF_DIM = 64
EMB_DIM = 256
COND_DIM = 100
Z_DIM = 100


capHandler = CaptionHandler()


DEVICE = torch.device('cuda')
GENERATOR = Generator(gf_dim=GF_DIM, emb_dim=EMB_DIM, z_dim=Z_DIM, cond_dim=COND_DIM)
GENERATOR.cuda()
DISCRIMINATORS = [Disc64(DF_DIM), Disc128(DF_DIM), Disc256(DF_DIM)]
for d in DISCRIMINATORS:
    d.cuda()
RNN = RNNEncoder(vocabsize=capHandler.vocab_size, nhidden=EMB_DIM)
RNN.cuda()
CNN = CNNEncoder(out_dim=EMB_DIM)
CNN.cuda()

#%%
import numpy as np

class GanTester(ModelTrainer):
    def __init__(self, capHandler: capHandler):
        super().__init__()
        self.capHandler = capHandler
        self._load_weights(modules=[CNN, RNN, GENERATOR] + DISCRIMINATORS)
        RNN.freeze_all_weights()
        CNN.freeze_all_weights()

    def _make_mask(self, lengths: Tensor) -> Tensor:
        maxlen = max(lengths)
        masks = [[1]*leng + [0]*(maxlen-leng)
                 for leng in lengths]
        return torch.LongTensor(masks).cuda()

    @timer
    def generate_images(self, captions: List[List[str]]) -> Tensor:
        # Create fixed noise
        fixedNoise = self._make_noise(batch_size=len(captions), z_dim=Z_DIM)
        # Process captions
        (words, lengths) = capHandler.preprocess(captions)
        masks = self._make_mask(lengths)
        #################### Embed words ####################
        (word_embs, sent_embs) = RNN(words.cuda(), lengths.cuda())
        #################### Make images ####################
        (fake_imgs, attn_maps, mu, logvar) = GENERATOR(noise=fixedNoise, sent_emb=sent_embs, word_embs=word_embs, mask=masks)
        return self._denormalise_multiple(fake_imgs[-1])

    def _show(self, img):
        npimg = img.detach().cpu().numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    
    def show_images(self, images: Tensor):
        grid = make_grid(
                images,
                nrow=1,
                normalize=True
                )
        self._show(grid)

gan = GanTester(capHandler)


#%%
captions = capHandler.get_captions(['319_423px-Place_de_Ville_C', '556_640px-Wightman_house_%28Wayne%2C_Nebraska%29_from_W'])
captions = capHandler.swap_captions(captions, 6, reverse=True)
images = gan.generate_images(captions)


gan._display_image(images[1])

#%%

trainer.train_gan(epochs=150)


#%%
