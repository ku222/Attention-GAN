#%%

import torch
from torch import Tensor
from torch import nn
from torch.optim import Adam
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from typing import Dict, List, Tuple
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, Normalize
import os
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from tqdm import tqdm
from umap import UMAP
import json
from torchvision.utils import make_grid
import numpy as np
from random import shuffle
from utilities.decorators import timer
from networks.cnn_embedder import ImageEmbedder
from trainers.trainer import ModelTrainer
from rapidfuzz.fuzz import ratio


class SingleImage:
    def __init__(self, fpath: str, imgtensors: List[Tensor]):
        self.fpath = fpath
        self.imgtensors = imgtensors if imgtensors else []
        self.caption = []
        self.class_id = None # Must be assigned

    def assign_class_id(self, class_id: int) -> None:
        self.class_id = class_id

    @property
    def img64(self) -> Tensor:
        return self.imgtensors[0] if len(self.imgtensors) >= 1 else None

    @property
    def img128(self) -> Tensor:
        return self.imgtensors[1] if len(self.imgtensors) >= 2 else None
    
    @property
    def img256(self) -> Tensor:
        return self.imgtensors[2] if len(self.imgtensors) >= 3 else None
    
    @property
    def caption_length(self) -> int:
        return len(self.caption)

    def view_image(self) -> None:
        imshow(self.img256.permute(1,2,0))
        print(self.caption)
    

class Vocab:
    """
    Language model mapping words to indices
    """
    def __init__(self):
        self.word2count = {}
        self.index2word = {}
        self.word2index = {}
        self.n_words = 0
        self.vocab_built = False

    def process(self, tokens: List[str]) -> List[str]:
        """Convert a list of words to a list of Indices"""
        indices = []
        for word in tokens:
            word = word if word in self.word2index else '[UNK]'
            idx = self.word2index[word]
            indices.append(idx)
        return indices

    def buildVocab(self, dataset: "Dataset") -> None:
        """Builds the vocabulary"""
        for image in dataset.images:
            self._addCaption(image.caption)
        self.vocab_built = True
        
    def buildVocabFromMapping(self, mapping: dict):
        for (path, (caption, class_id)) in mapping.items():
            self._addCaption(caption)
        self.vocab_built = True

    def _addCaption(self, caption: List[str]):
        for word in caption:
            self._addWord(word)

    def _addWord(self, word: str):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            

class Dataset:
    def __init__(self, rootdir=r'D:\GAN\bedroom', max_images=100):
        self.rootdir = rootdir
        self.max_images = max_images
        self.images = self.make_data()
        self.paths_to_images = {img.fpath: img 
                                for img in self.images}
        self.vocab = Vocab()
        
    @property
    def max_seqlen(self) -> int:
        caption_lengths = [len(img.caption) for img in self.images]
        return max(caption_lengths)

    @timer
    def make_data(self) -> List[SingleImage]:
        print('Processing Photos', '='*20)
        # Define recurse find
        image_paths = []
        def _recurseFindPhotos(path: List[str]) -> None:
            nonlocal image_paths
            fPath, f = '/'.join(path), path[-1]
            if f.lower().endswith('.jpg'):
                image_paths.append(fPath)
            else:
                for file in os.listdir(fPath):
                    _recurseFindPhotos(path + [file])
        # Pull out photo paths
        _recurseFindPhotos(path=[self.rootdir])
        maximages = min(self.max_images, len(image_paths)*2)
        # Process images
        counter = 0
        images = []
        for fPath in tqdm(image_paths):
            try:
                normal = SingleImage(fpath=fPath, imgtensors=Dataset.makeImage(fPath))
                reverse = SingleImage(fpath=f'{fPath}_r', imgtensors=Dataset.makeImage(fPath, flip_horizontal=True))
                images.extend([normal, reverse])
                counter += 2
            except FileNotFoundError:
                continue
            if counter > maximages:
                return images
        return images

    @staticmethod
    def makeImage(filename, flip_horizontal=False) -> List[Tensor]:
        """Create 64, 128, 256 resolution-tensors for a single image."""
        img = Image.open(filename)
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

    def save_captions_and_class_ids(self, path=r'D:\GAN\bedroomProcessed\captionsAndClassIDs.json') -> None:
        mapping = {
            img.fpath: [img.caption, img.class_id]
            for img in self.images
            }
        with open(path, 'w') as file:
            json.dump(mapping, file)

    def load_captions_and_class_ids(self, path=r'D:\GAN\bedroomProcessed\captionsAndClassIDs.json') -> None:
        with open(path) as file:
            mapping = json.load(file)
        for (path, (caption, class_id)) in mapping.items():
            img = self.paths_to_images[path]
            img.caption = caption
            img.class_id = class_id

    def _show(self, img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

    def evaluate_clustering(self, idx: int, max_images=50, nrow=10, folder='images_testing'):
        # Get image
        if isinstance(idx, int):
            image = self.images[idx]
        else:
            image = self.paths_to_images[idx]
        # Plot
        for (i, cap) in enumerate(reversed(image.caption), 1):
            (thisK, thisCluster) = cap.split('c')
            members = [img for img in self.images if img.caption[-i].endswith(thisCluster)]
            print(f'When k={thisK}, members={len(members)}')
            shuffle(members)
            members = members[ :max_images]
            members_images = [img.img256 for img in members]
            grid = make_grid(
                torch.stack(members_images),
                nrow=nrow,
                normalize=True
                )
            self._show(grid)
            plt.savefig(f"{folder}/k-{thisK}")
            plt.close()

    @timer
    def make_dataloaders(self, batch_size=16, shuffle=True) -> DataLoader:
        """Build a set of DataLoaders from a Dataset"""
        # Build vocab if needed
        if not self.vocab.vocab_built:
            self.vocab.buildVocab(self)
        imagelist = self.images
        # Split on spaces and commas
        all_tokens = [img.caption for img in imagelist]
        # Get Unpadded lengths
        all_lengths = [len(tokens) for tokens in all_tokens]
        # turn tokens into indices
        all_indices = [self.vocab.process(tokens=tokens) for tokens in all_tokens]
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
        return DataLoader(dataset=t_dataset, batch_size=batch_size, shuffle=shuffle)


class HierarchicalClusterer(ModelTrainer):
    """
    Clusters our images using various values of K
    """
    def __init__(self):
        self.embedder = ImageEmbedder()

    def cluster(self, dataset: Dataset, latent_dims=512, max_vocab_size=600, min_clusters=5, batch_size=32, method='agglomerative') -> None:
        """
        Clusters the images in the given dataset.
        Images will be assigned clusters and class_ids (this is an in-place operation).
        Total number of clusters will not overshoot the maximum vocab size.
        Method must be in ['kmeans', 'agglomerative_single_linkage, agglomerative_complete'].
        """
        all_images = torch.stack([img.img256 for img in dataset.images])
        embeddings = self.embedder.embed(all_images, batch_size=batch_size)
        X = embeddings.numpy()
        if latent_dims < 512:
            X = self._reduce_dimensionality(X, outdims=latent_dims)
        k_values = self._determine_k_values(dataset, max_vocab_size=max_vocab_size, min_k=min_clusters)
        # Assign clusters based on each increasing value of k
        for k in k_values:
            clusterLabels = self._makeClusterLabels(X=X, k=k, method=method)
            for (img, label) in zip(dataset.images, clusterLabels):
                img.caption.append(label)
        # Assign class IDs based on final & most granular clusters
        smallest_clusters = set(clusterLabels)
        id_map = {label: i for (i, label) in enumerate(smallest_clusters)}
        for (img, label) in zip(dataset.images, clusterLabels):
            class_id = id_map[label]
            img.assign_class_id(class_id)

    @timer
    def _reduce_dimensionality(self, X: np.ndarray, outdims=32) -> np.ndarray:
        umapper = UMAP(n_components=outdims)
        return umapper.fit_transform(X)

    @timer
    def _makeClusterLabels(self, X: np.ndarray, k: int, method: str) -> List[str]:
        """Clusters data, outputs an ordered list of cluster labels"""
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=k)
        elif method == 'agglomerative_single_linkage':
            clusterer = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='single', compute_full_tree=False)
        elif method == 'agglomerative_complete':
            clusterer = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='complete', compute_full_tree=False)    
        clusters = clusterer.fit(X)
        labels = clusters.labels_.tolist()
        return [f'k{k}c{c}' for c in labels]

    def _determine_k_values(self, dataset: Dataset, max_vocab_size=600, min_k=5):
        """
        Determines number of hierarchical clusters given a datset and max_vocab_size desired.
        Returns k values in ascending order from smallest to largest k.
        """
        num_images = len(dataset.images)
        factor = 2
        k = (max_vocab_size // factor)
        output = []
        while k > min_k:
            output.append(k)
            factor = 2*factor
            k = (max_vocab_size // factor)
        return list(reversed(output))


class CaptionHandler(Dataset):
    def __init__(self, vocab_path=r'D:\GAN\bedroomProcessed\captionsAndClassIDs.json'):
        self.vocab_path = vocab_path
        self.vocab = Vocab()
        self.img2caption = dict()
        # Initialize vocab and img2caption
        self._restore_state()

    @property
    def vocab_size(self) -> int:
        return self.vocab.n_words

    def _restore_state(self) -> None:
        with open(self.vocab_path) as file:
            mapping = json.load(file)
        # Rebuild vocab
        self.vocab.buildVocabFromMapping(mapping)
        # Create img2caption mapping
        for (path, (caption, class_id)) in mapping.items():
            self.img2caption[path] = caption

    def get_captions(self, imgnames: List[str]) -> List[List[str]]:
        return [self._get_caption(name) for name in imgnames]
    
    def swap_captions(self, captions: List[List[str]], num=1, reverse=False) -> List[List[str]]:
        """Swaps elements between two captions. Default is most global elements swapped first"""
        assert len(captions) == 2
        (c1, c2) = captions
        (newc1, newc2) = c1[:], c2[:]
        for i in range(1, num+1):
            i = -i if reverse else (i-1)
            newc1[i] = c2[i]
            newc2[i] = c1[i]
        return [newc1, newc2]

    def preprocess(self, captions: List[List[str]]) -> Tuple[Tensor, Tensor]:
        """Converts a list of captions to a tensor of indices and tensor of lengths"""
        all_indices = [self.vocab.process(caption) for caption in captions] 
        all_lengths = [len(tokens) for tokens in all_indices]
        return (
            torch.LongTensor(all_indices),  # Caption indices
            torch.LongTensor(all_lengths),  # Caption lengths
        )

    def _get_caption(self, imgname: str) -> List[str]:
        max_similarity = 0
        match = None
        for imgpath in self.img2caption.keys():
            if imgname in imgpath:
                similarity = ratio(imgname, imgpath)
                if similarity > max_similarity:
                    match = imgpath
                    max_similarity = similarity
        # Lookup best match
        return self.img2caption[match]