#%%

import torch
from torch import nn
from torch import Tensor
from utilities.layers import Layers
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from typing import Tuple


down = Layers.downBlockLeakyReLU
up = Layers.upBlockReLU


class AE_Encoder(nn.Module):
    """
    A simple Convolutional Encoder Model
    """
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            down(3, 8),     #128
            down(8, 16),     #64
            down(16, 32),    #32
            down(32, 64),   #16
            down(64, 128),   #8
            down(128, 256),   #4
            down(256, 512),   #2
            down(512, 1024),   #1
            )
    
    def forward(self, image: Tensor) -> Tensor:
        output = self.blocks(image)            
        return output.squeeze(3).squeeze(2)     # (M, 128)


class AE_Decoder(nn.Module):
    """
    A simple Convolutional Decoder Model
    """
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            up(1024, 512),  #2
            up(512, 256),   #4
            up(256, 128),   #8
            up(128, 64),   #16
            up(64, 32),   #32
            up(32, 16),    #64
            up(16, 8),     #128
            up(8, 3),      #256,
            nn.Tanh()
            )
    
    def forward(self, embedding: Tensor) -> Tensor:
        embedding = embedding.unsqueeze(2).unsqueeze(3) # (M, 1024, 1, 1)
        output = self.blocks(embedding)                 # (M, 3, 256, 256)
        return output


class AutoEncoder(nn.Module):
    def __init__(self, nz=128):
        super().__init__()
        self.encoder = AE_Encoder()
        self.decoder = AE_Decoder()
        # Encoder -> z, mu, logvar
        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, nz)
        self.fc22 = nn.Linear(512, nz)
        # z, mu, logvar -> Decoder
        self.fc3 = nn.Linear(nz, 512)
        self.fc4 = nn.Linear(512, 1024)
        # Relu helpers
        self.relu = nn.ReLU()
        # Cuda self
        self.cuda()

    def loss_function(self, recon_x, x, mu, logvar) -> Tensor:
        BCE = torch.mean((recon_x - x)**2)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
        return (BCE + KLD)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        decoded = self.decode(z)
        return (z, decoded, mu, logvar)

    def encode(self, x):
        conv = self.encoder(x)  # (M, 1024)
        h1 = self.fc1(conv)
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        return self.decoder(deconv_input)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).cuda()
        return eps.mul(std).add_(mu)

    def embed(self, images: Tensor, batch_size: int) -> Tensor:
        """
        Embeds images, returning an (M, 128) embedding tensor
        Input = (M, 3, 256, 256)
        Output = (M, 128) embeddings
        """
        self.eval()
        t_dataset = TensorDataset(images)
        loader = DataLoader(t_dataset, shuffle=False, batch_size=batch_size)
        # Embed inputs
        output = []
        with torch.no_grad():
            for batch in loader:
                batch = batch[0].cuda()
                (embedding, _, _, _) = self(batch)
                output.append(embedding.detach().cpu())
        return torch.cat(output, dim=0)