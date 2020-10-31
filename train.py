#%%
import torch
from torch.optim import Adam
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
DATASET = BirdsDataset(max_images=100)
PREPROCESSOR = DatasetPreprocessor()
DATALOADER = PREPROCESSOR.preprocess(dataset, maxlen=SEQ_LEN, batch_size=BATCH_SIZE)

# Networks/Modules
DEVICE = torch.device('cuda')
GENERATOR = Generator(gf_dim=GF_DIM, emb_dim=EMB_DIM, z_dim=Z_DIM).to(DEVICE)
DISCRIMINATORS = [Disc64(df_dim), Disc128(df_dim), Disc256(df_dim)]
DISCRIMINATORS = [d.to(DEVICE) for d in discriminators]
EMBEDDER = RNNEncoder(vocabsize=preproc.n_words)

# Training items
GEN_OPTIM = Adam(GENERATOR.parameters(), lr=GEN_LR, betas=(0.5, 0.999))

@timer
def train_gan(epochs=1):
    for _ in range(epochs):
        for (words, masks, lengths, img64, img128, img256) in DATALOADER:
            words = words.to(DEVICE)
            masks = masks.to(DEVICE)
            lengths = lengths.to(DEVICE)
            img64 = img64.to(DEVICE)
            img128 = img128.to(DEVICE)
            img256 = img256.to(DEVICE)
            # Embed words
            hiddencell = EMBEDDER.init_hidden_cell_states(batch_size=BATCH_SIZE)
            (word_embs, sent_embs) = EMBEDDER(words, lengths, hiddencell)
            # Generate images
            noise = torch.randn((BATCH_SIZE, Z_DIM))
            # generate
            fake_imgs, attn_maps = generator(noise=noise, sent_emb=sent_emb, word_embs=word_embs, mask=mask)
            

#%%

generator = 
generator = generator.to(DEVICE)



#%%

discriminators = 
discriminators = 
losses = []
for (disc, fake_img) in zip(discriminators, fake_imgs):
    disc_loss = DiscLoss(device).get_loss(disc, fake_img, fake_img)
    losses.append(disc_loss)
    
#%%



dataset = BirdsDataset(max_images=100)

#%%
preproc = DatasetPreprocessor()
loader = preproc.preprocess(dataset)

#%%

(words, masks, lengths, img64, img128, img256) = next(iter(loader))

#%%

embedder = RNNEncoder(vocabsize=preproc.n_words)
hiddencell = embedder.init_hidden_cell_states(batch_size=16)

(word_embs, sent_embs) = embedder.forward(words, lengths, hiddencell)

#%%


