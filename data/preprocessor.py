
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple

from utilities.decorators import timer
from .birds import BirdsDataset, BirdImage


class DatasetPreprocessor:
    def __init__(self):
        self.word2count = {}
        self.index2word = {0: "[PAD]", 1: "[UNK]"}
        self.word2index = {v: k for (k, v) in self.index2word.items()}
        self.n_words = 2  # Count PAD, UNK
        self.vocab_built = False

    def preprocess_user(self, captions: List[str], maxlen=15, batch_size=8) -> DataLoader:
        """Preprocess user inputs - returns 2-tensor Dataloader of (tokens, lengths)"""
        # Split on spaces and commas
        all_tokens = [cap.replace(', ', ' , ').split() for cap in captions]
        all_lengths = [len(tokens) for tokens in all_tokens]
        # add padding up to maxlen
        all_tokens = [tokens + ['[PAD]']*(maxlen - len(tokens)) for tokens in all_tokens]
        # Word -> Index, and Word -> Mask
        all_indices = [self._make_indices(tokens=tokens) for tokens in all_tokens]
        # Make into TensorDataset
        t_dataset = TensorDataset(
            torch.LongTensor(all_indices),
            torch.LongTensor(all_lengths)
        )
        return DataLoader(dataset=t_dataset, batch_size=batch_size)

    @timer
    def preprocess(self, dataset: BirdsDataset, maxlen=15, batch_size=16) -> DataLoader:
        """Convert caption strings to embedding indices, and makes mask"""
        # Build vocab if needed
        if not self.vocab_built:
            self._buildVocab(dataset)
        # Split on spaces and commas
        all_tokens = [img.caption.replace(', ', ' , ').split() for img in dataset.images]
        all_lengths = [len(tokens) for tokens in all_tokens]
        # add padding up to maxlen
        all_tokens = [tokens + ['[PAD]']*(maxlen - len(tokens)) for tokens in all_tokens]
        # Word -> Index, and Word -> Mask
        all_indices = [self._make_indices(tokens=tokens) for tokens in all_tokens]
        # Make into TensorDataset
        t_dataset = TensorDataset(
            torch.LongTensor(all_indices),
            torch.LongTensor(all_lengths),
            torch.LongTensor(dataset.all_class_ids),
            torch.stack(dataset.all_img64),
            torch.stack(dataset.all_img128),
            torch.stack(dataset.all_img256),
        )
        return DataLoader(dataset=t_dataset, batch_size=batch_size)

    def _make_indices(self, tokens: List[str]) -> List[str]:
        indices = []
        for word in tokens:
            word = word if word in self.word2index else '[UNK]'
            idx = self.word2index[word]
            indices.append(idx)
        return indices
    
    def _buildVocab(self, dataset: BirdsDataset) -> None:
        for image in dataset.images:
            self._addCaption(image.caption)
        self.vocab_built = True

    def _addCaption(self, caption: str):
        tokens = caption.replace(', ', ' , ').split()
        for word in tokens:
            self._addWord(word)

    def _addWord(self, word: str):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1