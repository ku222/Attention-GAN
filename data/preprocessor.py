import torch
from torch import Tensor
import numpy as np
from nptyping import NDArray, Float64
from typing import Dict, List, Tuple
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip, Normalize
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.pyplot import imshow

from utilities.decorators import timer
from networks.rnn_encoder import RNNEncoder
from random import shuffle


class SingleImage:
    """Class storing the image (in 3 resolutions) and caption"""
    def __init__(self, imgtensors: List[Tensor], caption: str, class_id: int = 0):
        (img64, img128, img256) = imgtensors
        self.img64 = img64
        self.img128 = img128
        self.img256 = img256
        self.caption = caption
        self.class_id = class_id

    @property
    def caption_length(self) -> int:
        return len(self.caption.split(','))

    def view_image(self) -> None:
        imshow(self.img256.permute(1,2,0))
        print(self.caption)


class BuildingsDataset:
    def __init__(self, imagedir=r'D:\GAN\buildingsDataset'):
        self.imagedir = imagedir
        self.images = []
        self.caption_lookup = {
            'Achaemenid architecture': 'ancient,rocky,barren,desert',
            'American craftsman style': 'urban,suburb,small',
            'American Foursquare architecture': 'urban,suburb,small',
            'Ancient Egyptian architecture': 'pyramid,desert,barren,sandy,open',
            'Art Deco architecture': 'modern,square,curvy,urban',
            'Art Nouveau architecture': 'religious,castle,urban',
            'Baroque architecture': 'religious,grand,urban',
            'Bauhaus architecture': 'modern,square,simple',
            'Beaux-Arts architecture': 'urban,square',
            'Byzantine architecture': 'rocky,urban,religious',
            'Chicago school architecture': 'square,boring,urban',
            'Colonial architecture': 'religious,suburb',
            'Deconstructivism': 'abstract,urban',
            'Edwardian architecture': 'religious,urban',
            'Georgian architecture': 'square,urban',
            'Gothic architecture': 'religious,spiky',
            'Greek Revival architecture': 'religious,suburb',
            'International style': 'skyscraper,modern,square,urban',
            'Novelty architecture': 'abstact,ugly,colorful',
            'Palladian architecture': 'religious,urban,spacedout',
            'Postmodern architecture': 'modern,square,urban',
            'Queen Anne architecture': 'small,suburb',
            'Romanesque architecture': 'religious,spiky',
            'Russian Revival architecture': 'religious,russian',
            'Tudor Revival architecture': 'tudor,small,suburb'
            }
    
    @timer
    def make_data(self, batch_size: int, max_files=9999):
        counter = 1
        for (i, folder) in enumerate(os.listdir(self.imagedir)):
            for filename in os.listdir(f"{self.imagedir}/{folder}"):
                print(f"Opened File {round((counter/min(4794, max_files))*100, 1)}%")
                counter += 1
                imagename = f"{folder}/{filename}"
                try:
                    imgtensors = self._make_image_tensors(folder, filename)
                    flipped_imgtensors = self._make_image_tensors(folder, filename, flip_horizontal=True)
                    caption = self.caption_lookup[folder]
                    img1 = SingleImage(imgtensors, caption=caption, class_id=i)
                    img2 = SingleImage(flipped_imgtensors, caption=caption, class_id=i)
                    self.images.extend([img1, img2])
                except FileNotFoundError:
                    continue
                if counter >= max_files:
                    return

    def _make_image_tensors(self, folder: str, filename: str, flip_horizontal=False) -> List[Tensor]:
        """Create 64, 128, 256 resolution-tensors for a single image."""
        img = Image.open(f"{self.imagedir}/{folder}/{filename}")
        img = img.convert("RGB")
        output = []
        for res in [64, 128, 256]:
            transforms = [
                Resize(size=(res, res)),
                RandomHorizontalFlip(p=1) if flip_horizontal else None,
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
            pipeline = Compose([t for t in transforms if t is not None])
            output.append(pipeline(img))
        return output


class Dataset:
    """
    Helper class that creates and stores all our images
    """
    def __init__(self, shuffle=True, batch_size=16, make_validation=False, indexdoc=r'D:\GAN\spacesDataset\index.csv', imagedir=r'D:\GAN\spacesDataset\images'):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexdoc = indexdoc
        self.imagedir = imagedir
        self.img2cap = self.load_index()
        self.images = []
        self.validation_images = []
        self.make_images(make_validation=make_validation)
        self.embedder = RNNEncoder()
        self._embed_image_captions()
        self.embedder.move_to_cpu()
        print(f'\t {len(self.images)} images with captions created')

    def load_index(self) -> dict:
        """Loads the CSV that maps each image filename to its captions"""
        output = dict()
        index = pd.read_csv(self.indexdoc)
        for (i, row) in index.iterrows():
            filename = row['filename']
            labels = row['labels']
            # add to output dict
            output[filename] = labels
        return output

    @timer
    def make_images(self, make_validation: bool) -> None:
        """Main method - creates image objects and their captions"""
        all_filenames = os.listdir(self.imagedir)
        train_fnames = all_filenames[:]
        if make_validation:
            shuffle(all_filenames)
            splitpoint = int(0.75*len(all_filenames))
            train_fnames = all_filenames[ :splitpoint]
            val_fnames = all_filenames[splitpoint: ]
            print(len(train_fnames), len(val_fnames))
            # make val images
            val_images = self._make_images(val_fnames) + self._make_images(val_fnames, flip_horizontal=True)
            self.validation_images = self._sort_images(val_images)
        # make train images
        train_images = self._make_images(train_fnames) + self._make_images(train_fnames, flip_horizontal=True)
        self.images = self._sort_images(train_images)

    def _make_images(self, fnames: List[str], flip_horizontal=False, flip_vertical=False) -> List[SingleImage]:
        """Create image objects out of a list of filenames."""
        output = []
        for fname in fnames:
            caption = self.img2cap.get(fname[:-4])
            if not caption:
                continue
            imgtensors = self._make_image_tensors(fname, flip_horizontal, flip_vertical)
            image = SingleImage(imgtensors=imgtensors, caption=caption)
            output.append(image)
        # Sort output by longest caption lengths descending
        return output

    def _make_image_tensors(self, filename: str, flip_horizontal=False, flip_vertical=False) -> List[Tensor]:
        """Create 64, 128, 256 resolution-tensors for a single image."""
        img = Image.open(f"{self.imagedir}/{filename}")
        img = img.convert("RGB")
        output = []
        for res in [64, 128, 256]:
            transforms = [
                Resize(size=(res, res)),
                RandomHorizontalFlip(p=1) if flip_horizontal else None,
                RandomVerticalFlip(p=0.5) if flip_vertical else None,
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
            pipeline = Compose([t for t in transforms if t is not None])
            output.append(pipeline(img))
        return output

    def _sort_images(self, images: List[SingleImage]) -> List[SingleImage]:
        """Sorts images by caption length in descending order of length/"""
        if not self.shuffle:
            images = sorted(images, key=lambda image: image.caption_length, reverse=True)
            return images
        shuffle(images)
        return images


class DatasetPreprocessor:
    """
    Acts as our Vocab model, and holds methods to create Dataloaders out of a Dataset.
    """
    def __init__(self, batch_size=16, max_seqlen=15):
        self.word2count = {}
        self.index2word = {0: "[PAD]", 1: "[UNK]", 2: '[SOS]', 3: '[EOS]'}
        self.word2index = {v: k for (k, v) in self.index2word.items()}
        self.PAD_idx = 0
        self.UNK_idx = 1
        self.SOS_idx = 2
        self.EOS_idx = 3
        self.n_words = 4  # Count PAD, UNK
        self.vocab_built = False
        self.batch_size = batch_size
        self.max_seqlen = max_seqlen

    def make_dataloaders(self, dataset: Dataset, shuffle=True) -> DataLoader:
        """Build a set of DataLoaders from a Dataset"""
        # Build vocab if needed
        if not self.vocab_built:
            self._buildVocab(dataset)
        imagelist = dataset.images
        # Split on spaces and commas
        all_tokens = [img.caption.split(',') for img in imagelist]
        # Get Unpadded lengths
        all_lengths = [len(tokens) for tokens in all_tokens]
        # add padding up to maxlen
        all_tokens = [tokens + ['[PAD]']*(self.max_seqlen - len(tokens)) for tokens in all_tokens]
        # turn tokens into indices
        all_indices = [self.make_indices(tokens=tokens) for tokens in all_tokens]
        # Get images
        all_img64 = [img.img64 for img in imagelist]
        all_img128 = [img.img128 for img in imagelist]
        all_img256 = [img.img256 for img in imagelist]
        # Get class Ids
        class_ids = [img.class_id for img in imagelist]
        # Make into TensorDataset
        t_dataset = TensorDataset(
            torch.LongTensor(all_indices),  # Caption indices
            torch.LongTensor(all_lengths),  # Unpadded Lengths
            torch.LongTensor(class_ids),    # Class IDs
            torch.stack(all_img64),
            torch.stack(all_img128),
            torch.stack(all_img256),
        )
        # Add to output list
        return DataLoader(dataset=t_dataset, batch_size=self.batch_size, shuffle=shuffle)

    def make_indices(self, tokens: List[str]) -> List[str]:
        """Convert a list of words to a list of Indices"""
        indices = []
        for word in tokens:
            word = word if word in self.word2index else '[UNK]'
            idx = self.word2index[word]
            indices.append(idx)
        return indices

    def _buildVocab(self, dataset: Dataset) -> None:
        for image in dataset.images:
            self._addCaption(image.caption)
        self.vocab_built = True

    def _addCaption(self, caption: str):
        tokens = caption.split(',')
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