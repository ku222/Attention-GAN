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
# Modules/networks
from networks.generator import Generator
from networks.discriminators import Disc64, Disc128, Disc256
from networks.attention import AttentionModule
from networks.rnn_encoder import RNNEncoder
# Losses
from losses.disc_loss import DiscLoss
from losses.gen_loss import GenLoss
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
BATCH_SIZE = 16
GEN_LR = 0.0002
DISC_LR = 0.0002

# Datasets
DATASET = BirdsDataset(max_images=9999)
PREPROCESSOR = DatasetPreprocessor()
DATALOADER = PREPROCESSOR.preprocess(DATASET, maxlen=SEQ_LEN, batch_size=BATCH_SIZE)



#%%
# Networks/Modules
DEVICE = torch.device('cuda')
GENERATOR = Generator(gf_dim=GF_DIM, emb_dim=EMB_DIM, z_dim=Z_DIM)
GENERATOR.to(DEVICE)
DISCRIMINATORS = [Disc64(DF_DIM), Disc128(DF_DIM), Disc256(DF_DIM)]
for d in DISCRIMINATORS:
    d.to(DEVICE)
EMBEDDER = RNNEncoder(vocabsize=PREPROCESSOR.n_words, nhidden=EMB_DIM)
EMBEDDER.to(DEVICE)

# Training items
GEN_OPTIM = Adam(GENERATOR.parameters(), lr=GEN_LR, betas=(0.5, 0.999))
DISC_OPTIMS = [Adam(disc.parameters(), lr=DISC_LR, betas=(0.5, 0.999))
                   for disc in DISCRIMINATORS]


#%%

class GanTrainer:
    def __init__(self):
        self.disclogs = defaultdict(list)
        self.genlogs = []
        
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
        
    def _evaluate_with_img_grid(self, fake_images: List[Tensor], epoch: int=None, batch: int=None, folder='generated_images') -> None:
        '''Produces an [n_images, n_images] image grid for evaluation purposes, saves to an image folder'''
        # Find largest square number
        num_images = len(fake_images[0])
        square = next(i for i in range(num_images, 1, -1) if math.sqrt(i) == int(math.sqrt(i)))
        sqrt = int(math.sqrt(square))
        for (images, res) in zip(fake_images, ['064', '128', '256']):
            # plot images
            images = images[:square].detach().cpu()
            (f, axarr) = plt.subplots(sqrt, sqrt)
            counter = 0
            for i in range(sqrt):
                for j in range(sqrt):
                    # define subplot
                    image = images[counter]
                    image = image.permute(1, 2, 0)
                    axarr[i,j].axis('off')
                    axarr[i,j].imshow(image)
                    counter += 1
            # save plot to file
            timenow: str = str(datetime.now()).split('.')[0].replace(':', '-')
            fname = f'{folder}/epoch_{epoch}-batch_{batch}-{res}x{res}.png' if batch else f'{folder}/_{res}x{res}-{timenow}.png'
            plt.savefig(fname)
            plt.close()

    def make_images(self, captions: List[str]) -> None:
        user_dataset = PREPROCESSOR.preprocess_user(captions)
        for batch in user_dataset:
            batch = [t.to(DEVICE) for t in batch]
            (words, lengths) = batch
            masks = self._make_mask(lengths).to(DEVICE)
            # Embed words
            hiddencell = EMBEDDER.init_hidden_cell_states(batch_size=BATCH_SIZE)
            (word_embs, sent_embs) = EMBEDDER(words, lengths, hiddencell)
            # Make noise, concatenate with sentence embedding
            noise = torch.randn((len(words), Z_DIM)).to(DEVICE)
            # Make images
            (fake_imgs, attn_maps) = GENERATOR(noise=noise, sent_emb=sent_embs, word_embs=word_embs, mask=masks)
            self._evaluate_with_img_grid(fake_imgs)

    @timer
    def train_gan(self, epochs=50, evaluate_every=50):
        for e in range(epochs):
            for (b, batch) in enumerate(DATALOADER):
                batch = [t.to(DEVICE) for t in batch]
                (words, lengths, img64, img128, img256) = batch
                if min(lengths) < 2 or len(words) < 2:
                    continue
                masks = self._make_mask(lengths).to(DEVICE)
                # Embed words
                hiddencell = EMBEDDER.init_hidden_cell_states(batch_size=BATCH_SIZE)
                (word_embs, sent_embs) = EMBEDDER(words, lengths, hiddencell)
                # Make noise, concatenate with sentence embedding
                noise = torch.randn((len(words), Z_DIM)).to(DEVICE)
                # Make images
                fake_imgs, attn_maps = GENERATOR(noise=noise, sent_emb=sent_embs, word_embs=word_embs, mask=masks)
                real_imgs = [img64, img128, img256]
                # Get discriminator loss
                for (i, (disc, optim, fake_img, real_img)) in enumerate(zip(DISCRIMINATORS, DISC_OPTIMS, fake_imgs, real_imgs)):
                    optim.zero_grad()
                    loss = DiscLoss(DEVICE).get_loss(disc, fake_img, real_img)
                    loss.backward(retain_graph=True)
                    optim.step()
                    self.disclogs[i].append(loss.item())
                # Get generator loss
                GEN_OPTIM.zero_grad()
                total_genloss = 0
                for (disc, fake_img) in zip(DISCRIMINATORS, fake_imgs):
                    loss = GenLoss(DEVICE).get_loss(disc, fake_img)
                    total_genloss += loss
                total_genloss.backward()
                GEN_OPTIM.step()
                self.genlogs.append(total_genloss.item())
                # Log stats
                print(f"\t {self._latest_loss()}")
                if (b+1) % evaluate_every == 0:
                    self._evaluate_with_img_grid(fake_imgs, epoch={e+1}, batch=b+1)


trainer = GanTrainer()

#%%
## 44 seconds for 1000 images
trainer.train_gan()

#%%
from torch.utils.data import DataLoader

a = torch.randn(16, 8)
b = torch.randn(4, 8)
len(a)
#%%
import math
num_images = 13


#%%

Training.plot_history(trainer.genlogs)

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