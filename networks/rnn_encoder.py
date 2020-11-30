
#%%

import torch
from torch import nn
from torch.autograd import Variable
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from typing import Tuple


class RNNEncoder(nn.Module):
    def __init__(self, vocabsize: int, embdim=300, dropprob=0.5, nhidden=128, nlayers=1, bidirectional=True):
        super().__init__()
        """
        Defines an LSTM encoder
        Params:
            vocabsize:  size of the dictionary
            embdim:     size of each embedding vector
            dropprob:   probability of an element to be zeroed
            nlayers:    number of recurrent layers
        """
        self.vocabsize = vocabsize  
        self.embdim = embdim 
        self.dropprob = dropprob
        self.nlayers = nlayers
        # Bidirectional setup
        self.bidirectional = bidirectional
        self.ndirections = 2 if bidirectional else 1
        self.nhidden = nhidden // self.ndirections
        # Define LSTMs
        self.__define_module()
        # Initialize embeddings
        self.__init_embedding_weights()

    def __define_module(self) -> None:
        """Create embeddings, dropout, LSTM"""
        self.embedding = nn.Embedding(self.vocabsize, self.embdim)
        self.dropout = nn.Dropout(self.dropprob)
        self.rnn = nn.LSTM(
            input_size=self.embdim,
            hidden_size=self.nhidden,
            num_layers=self.nlayers,
            batch_first=True,
            dropout=self.dropprob,
            bidirectional=self.bidirectional
        )

    def __init_embedding_weights(self, bound=0.1) -> None:
        """Initialize embedding weights between neg/pos bounds"""
        self.embedding.weight.data.uniform_(-bound, bound)
    
    def init_hidden_cell_states(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Creates hidden and cell states to feed into LSTM at time zero
        Hidden/cell state shape = (nlayers*ndirections, batch_size, nhidden)
        """
        weight = next(self.parameters()).data
        return (
            Variable(weight.new(self.nlayers * self.ndirections, batch_size, self.nhidden).zero_()),
            Variable(weight.new(self.nlayers * self.ndirections, batch_size, self.nhidden).zero_())
            )
    
    def freeze_all_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, captions: Tensor, caption_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagation - turns captions into word-level and sentence-level embeddings

        Params:
            captions:           torch.LongTensor of shape (batch_size, seq_len)
            caption_lengths:    Any Tensor type of shape (batch_size,)
            hiddencell:         Tuple with two tensors of shape (nlayers*ndirections, batch_size, nhidden)

        Returns a Tuple of:
            word_embeddings:    Tensor of shape (batch_size, nhidden, seq_len)
            sent_embeddings:    Tensor of shape (batch_size, nhidden)
        """
        X = self.embedding(captions)  # -> (batch_size, seq_len, embdim)
        X = self.dropout(X)           # -> (batch_size, seq_len, embdim)
        # Convert caption_lengths to list
        caption_lengths = caption_lengths.data.tolist() # List[int]
        # Pack padded sequence
        X = pack_padded_sequence(input=X, lengths=caption_lengths, batch_first=True, enforce_sorted=False)
        # Pass through RNN
        hiddencell = self.init_hidden_cell_states(batch_size=len(caption_lengths))
        (output, (hidden, cell)) = self.rnn(X, hiddencell)
        # Extract word embeddings from OUTPUT
        output = pad_packed_sequence(sequence=output, batch_first=True)[0]  # -> (batch_size, seq_len, nhidden)
        word_embs = output.transpose(1, 2)                                  # -> (batch_size, nhidden, seq_len)
        # Extract sentence embeddings: hidden shape = (2, batch_size, 64)
        sent_embs = hidden.transpose(0, 1).contiguous()                     # -> (batch_size, 2, 64)
        sent_embs = sent_embs.view(-1, self.ndirections * self.nhidden)     # -> (batch_size, 128)
        return (word_embs, sent_embs)