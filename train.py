#%%
import torch
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

gf_dim = 32
df_dim = 64
emb_dim = 256
z_dim = 100
seq_len = 15
batch_size = 8

device = torch.device('cuda')
noise = torch.randn((batch_size, z_dim)).to(device)
sent_emb = torch.randn((batch_size, emb_dim)).to(device)
word_embs = torch.randn((batch_size, emb_dim, seq_len)).to(device)
mask = AttentionModule.create_random_mask(batch_size, seq_len).to(device)


print(mask.shape)
word_embs.shape

#%%

generator = Generator(gf_dim=gf_dim, emb_dim=emb_dim, z_dim=z_dim)
generator = generator.to(device)

fake_imgs, attn_maps = generator(noise=noise, sent_emb=sent_emb, word_embs=word_embs, mask=mask)

#%%

discriminators = [Disc64(df_dim), Disc128(df_dim), Disc256(df_dim)]
discriminators = [d.to(device) for d in discriminators]
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

embedder.forward(words, )

#%%

