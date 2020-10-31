
#%%

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch import Tensor
from spacy.lang.en import English
from spacy.lang.de import German
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, Dataset, Example
from typing import Tuple, List
import spacy
import numpy as np
from decorators import timer
import random
import math
import time
import matplotlib.pyplot as plt

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True




class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float):
        """
        Params:
            input_dim:  Source vocabulary size
            emb_dim:    The number of expected embedding features
            hid_dim:    The number of features in the hidden state h
            n_layers:   Number of layers in the LSTM
            dropout:    Dropout between LSTM layers (except last layer)
        """
        super().__init__()
        # params
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        # layers
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=emb_dim)
        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=n_layers, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, src: Tensor) -> Tuple[Tensor]:
        """src = (seq_len, emb_dim)"""
        X = self.embedding(src)                     # -> (seq_len, batch, emb_dim)
        X = self.dropout(X)                         # -> (seq_len, batch, emb_dim)
        (output, (hidden, cell)) = self.rnn(X) 
        # outputs -> (seq_len, batch, hid_dim)
        # hidden -> (n_layers, batch, hid_dim)
        # cell -> (n_layers, batch, hid_dim)
        return (hidden, cell)
    

class Decoder(nn.Module):
    def __init__(self, output_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float):
        """
        Params:
            output_dim: Target vocabulary size
            emb_dim:    The number of expected embedding features
            hid_dim:    The number of features in the hidden state h
            n_layers:   Number of layers in the LSTM
            dropout:    Dropout between LSTM layers (except last layer)
        """
        super().__init__()
        # params
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        # layers
        self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim=emb_dim)
        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=n_layers, dropout=dropout)
        self.fc = nn.Linear(in_features=hid_dim, out_features=output_dim)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, inp: Tensor, hidden: Tensor, cell: Tensor) -> Tuple[Tensor]:
        """
        inp = (batch,)
        hidden = (n_layers, batch, hid_dim)
        cell = (n_layers, batch, hid_dim)
        """
        inp = inp.unsqueeze(0)                      # -> (1, batch)
        inp = self.embedding(inp)                   # -> (1, batch, emb_dim)
        (output, (hidden, cell)) = self.rnn(inp)
        # outputs -> (1, batch, hid_dim)
        # hidden -> (n_layers, batch, hid_dim)
        # cell -> (n_layers, batch, hid_dim)
        output = output.squeeze(0)                  # -> (batch, hid_dim)
        output = self.fc(output)                    # -> (batch, output_dim)
        return (output, hidden, cell)


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        # Checks
        assert encoder.hid_dim == decoder.hid_dim, 'Hidden dims not aligned'
        assert encoder.n_layers == decoder.n_layers, 'N_layers not aligned'
        # Weight initialization
        self.init_weights()
        # Loss history
        self.loss_history = []
    
    def forward(self, src: Tensor, trg: Tensor, teacher_force_ratio=0.5) -> Tensor:
        """
        src: (src_len, batch)
        trg: (trg_len, batch)
        """
        (trg_len, batch) = trg.shape
        trg_vocab_size = self.decoder.output_dim
        # Tensor to store outputs
        outputs = torch.zeros(trg_len, batch, trg_vocab_size).to(self.device)
        # Encoder forward pass
        (hidden, cell) = self.encoder(src)
        # Decoder forward
        inp = trg[0,:]                      # -> (batch,)
        for t in range(1, trg_len):
            (output, hidden, cell) = self.decoder(inp, hidden, cell)
            # store in output tensor
            outputs[t] = output
            # decide teacher force
            teacher_force = random.random() < teacher_force_ratio
            if teacher_force:
                inp = trg[t]              # -> (batch,)
            else:
                inp = output.argmax(dim=1)  # -> (batch,)
        # return outputs
        return outputs
    
    def init_weights(self) -> None:
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
            
    def count_parameters(self) -> None:
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {count:,} trainable parameters')
        

class Trainer:
    """Trains the Seq2Seq model"""
    @staticmethod
    @timer
    def train_step(model: Seq2Seq, iterator: BucketIterator, optimizer: Adam, criterion: CrossEntropyLoss, clip: float) -> float:
        model.train()
        epoch_loss = 0
        for (i, batch) in enumerate(iterator):
            src = batch.src.to(model.device)        # -> (src_len, batch)
            trg = batch.trg.to(model.device)        # -> (trg_len, batch)
            optimizer.zero_grad()
            # forward prop
            output = model(src, trg)                # -> (trg_len, batch, output_dim)
            # reshape output
            output_dim = output.shape[-1]
            output = output[1:]                     # -> (trg_len-1, batch)
            output = output.view(-1, output_dim)    # -> (trg_len-1 * batch, output_dim)
            # reeshape target
            trg = trg[1:]                           # -> (trg_len-1, batch)
            trg = trg.view(-1)                      # -> (trg_len-1 * batch,)
            # compute loss
            loss = criterion(output, trg)
            loss.backward()
            epoch_loss += loss.item()
            model.loss_history.append(loss.item())
            # clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            # update gradients
            optimizer.step()
        # Return average loss
        return epoch_loss / len(iterator)

    @staticmethod
    @timer
    def eval_step(model: Seq2Seq, iterator: BucketIterator, criterion: CrossEntropyLoss):
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for (i, batch) in enumerate(iterator):
                src = batch.src.to(model.device)        # -> (src_len, batch)
                trg = batch.trg.to(model.device)        # -> (trg_len, batch)
                # forward prop 
                output = model(src, trg, 0)             # -> (trg_len, batch, output_dim)
                # reshape output
                output_dim = output.shape[-1]
                output = output[1:]                     # -> (trg_len-1, batch)
                output = output.view(-1, output_dim)    # -> (trg_len-1 * batch, output_dim)
                # reeshape target
                trg = trg[1:]                           # -> (trg_len-1, batch)
                trg = trg.view(-1)                      # -> (trg_len-1 * batch,)
                # compute loss
                loss = criterion(output, trg)
                epoch_loss += loss.item()
        return epoch_loss / len(iterator)

    @staticmethod
    @timer
    def train_model(model: Seq2Seq, preprocessor, epochs=2):
        optimizer = Adam(model.parameters())
        trg_pad_token: str = preprocessor.TRG.pad_token
        english_2_idx: dict = preprocessor.TRG.vocab.stoi
        trg_pad_index: int = english_2_idx.get(trg_pad_token)
        criterion = CrossEntropyLoss(ignore_index=trg_pad_index)
        # training
        for _ in range(epochs):
            avg_train_loss = Trainer.train_step(model, preprocessor.train_iterator, optimizer, criterion, clip=1)
            avg_val_loss = Trainer.eval_step(model, preprocessor.valid_iterator, criterion)
            


from inception import inception_v3

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        pretrained = inception_v3(pretrained=True)
        self.inception = pretrained
        
    def _freeze_weights(self):
        """Freeze weights for all layers"""
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, images: Tensor) -> Tensor:
        return self.inception(images)


import pandas as pd
from pandas import DataFrame
import re
from typing import List


class CSVLoader:
    """Reads from Ginas CSV"""
    @staticmethod
    def load_data(csv_dir='') -> dict:
        """Loads filename/label pairs into a dict"""
        output = dict()
        if not csv_dir:
            csv_dir = r'D:\GAN\DL Data.csv'
        df: DataFrame = pd.read_csv(csv_dir)
        for (i, row) in df.iterrows():
            fname = row['filename']
            labels = row['labels']
            if str(labels) != 'nan':
                output[fname] = labels
        return output


class Lang:
    def __init__(self, name: str):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<UNK>", 3: "<PAD>"}
        self.n_words = len(self.index2word)
        self.vectors = None
        
    def addSentences(self, sentences: List[str]):
        for sentence in sentences:
            self.addSentence(sentence)

    def addSentence(self, sentence: str):
        for word in re.split(pattern=r'[,;\s]', string=sentence):
            word = word.lower().strip()
            if word:
                self.addWord(word)

    def addWord(self, word: str):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    @timer
    def make_vectors(self, emb_file=''):
        if not emb_file:
            emb_file = r'C:\Users\kovid\OneDrive\Documents\GitHub\archi-gan\RNN\.vector_cache\glove.6B.100d.txt'
        glove = {}
        with open(emb_file, encoding="utf8") as f:
            for line in f.readlines():
                line = line.split()
                word = line[0]
                if word in self.word2index:
                    embedding = torch.FloatTensor([float(v) for v in line[1:]])
                    glove[word] = embedding
        emb_dim = len(embedding)
        vectors = []
        for (index, word) in self.index2word.items():
            emb = glove.get(word)
            if emb is None:
                print(f"Word not recognized: {word}")
                emb = torch.zeros(emb_dim)
            vectors.append(emb)
        self.vectors = torch.stack(vectors)
        
    

data = CSVLoader.load_data()
lang = Lang(name='Vocab')
lang.addSentences([sentence for sentence in data.values()])
lang.make_vectors()


from PIL import Image
from torchvision import transforms
import os


class ImageFeeder:
    """"""
    @staticmethod
    @timer
    def load_data(img_folder='', max_images=10000) -> Tensor:
        # Create transformation pipeline
        pipeline = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if not img_folder:
            img_folder = r'D:/GAN/images'
        # Iterate over images
        img_tensors = []
        for (i, fname) in enumerate(os.listdir(img_folder)):
            if (i+1) > max_images:
                break
            img = Image.open(f"{img_folder}/{fname}").convert('RGB')
            img_tensor = pipeline(img)
            img_tensors.append(img_tensor)
        # return stacker tensors
        return torch.stack(img_tensors)


img_tensors = ImageFeeder.load_data(max_images=3)

#%%

img_tensors.shape

#%%

encoder = ImageEncoder()

#%%
device = torch.device('cuda')
test = img_tensors.to(device)
encoder.to(device)
with torch.no_grad():
    outputs = encoder(test)

#%%

outputs[1].shape