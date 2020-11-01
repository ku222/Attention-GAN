
#%%

import os
import re
from collections import defaultdict, Counter
from typing import List, Dict
from utilities.decorators import timer

from matplotlib.pyplot import imshow
from PIL import Image
import torch
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch import Tensor

from config import Config


class BirdImage:
    def __init__(self, root_dir: str, imgpath: str, caption: str):
        self.root_dir = root_dir
        self.imgpath = imgpath
        self.imgtensors = self._make_images()
        self.caption = caption

    @property
    def class_id(self) -> int:
        return int(self.imgpath[:3])

    def _make_images(self) -> List[Tensor]:
        imgtensors = []
        img = Image.open(f"{self.root_dir}/{self.imgpath}")
        for res in (64, 128, 256):
            pipeline = Compose([
                Resize(size=(res, res)),
                ToTensor(),
            ])
            tensor = pipeline(img)
            imgtensors.append(tensor)
        return imgtensors

    def view_image(self, idx=2) -> None:
        assert idx in [0, 1, 2]
        imgtensor = self.imgtensors[idx]
        imshow(imgtensor.permute(1,2,0))
        print(self.caption)


class CaptionCollection:
    def __init__(self):
        self._relevant = ['back', 'bill', 'breast', 'crown'] # attributes
        self._captions = {
            attrib: []
            for attrib in self._relevant
        }
        self._best_captions = []

    @property
    def best_caption(self) -> str:
        return ', '.join(self._best_captions)

    def add_caption(self, adjective: str, attribute: str) -> None:
        for attrib in self._relevant:
            if attrib in attribute.lower():
                self._captions[attrib].append(adjective)

    def make_best_caption(self) -> None:
        for (attrib, captions) in self._captions.items():
            counter = Counter(captions)
            best = counter.most_common(1)
            if best:
                (adjective, _) = best[0]
                self._best_captions.append(f"{adjective} {attrib}")


class BirdsDataset:
    def __init__(self, max_images=9999):
        # Make captions
        self._captions_dir = Config.BIRDS_CAPTIONS
        self.img2cap = self._load_captions()
        self._make_best_captions()
        # Make images
        self._images_dir = Config.BIRDS_IMAGES
        self.images = self._load_images(max_images)
    
    @property
    def all_class_ids(self) -> List[int]:
        return [img.class_id for img in self.images]
    
    @property
    def all_img64(self) -> List[Tensor]:
        return [img.imgtensors[0] for img in self.images]
    
    @property
    def all_img128(self) -> List[Tensor]:
        return [img.imgtensors[1] for img in self.images]
    
    @property
    def all_img256(self) -> List[Tensor]:
        return [img.imgtensors[2] for img in self.images]

    @timer
    def _load_images(self, max_images: int) -> List[BirdImage]:
        images = []
        imageIndex = self._load_imageIndex()
        counter = 0
        for (i, imgpath) in enumerate(imageIndex.values()):
            birdimage = BirdImage(
                root_dir=self._images_dir,
                imgpath=imgpath,
                caption=self.img2cap[imgpath].best_caption
            )
            images.append(birdimage)
            if (i+1) % 100 == 0:
                print(f"\t Transformed image {i+1}")
            if (i+1) >= max_images:
                break
        return images

    @timer
    def _load_captions(self) -> Dict[str, CaptionCollection]:
        img2cap = dict()
        attributes = self._load_attributes()
        certainties = self._load_certainties()
        imageIndex = self._load_imageIndex()
        with open(f"{self._captions_dir}/labels.txt") as f:
            for line in f.readlines():
                (imgID, attrID, isPresent, certaintyID, _) = line.split()
                if isPresent == '1' and certainties[certaintyID] == 'Definitely':
                    imgpath = imageIndex[imgID]
                    (adjective, attribute) = attributes[attrID]
                    capcollection = img2cap.get(imgpath)
                    if not capcollection:
                        capcollection = CaptionCollection()
                        img2cap[imgpath] = capcollection
                    capcollection.add_caption(adjective, attribute)
        return img2cap

    def _load_attributes(self) -> Dict[str, tuple]:
        attributes = dict()
        with open(f"{self._captions_dir}/attributes.txt") as f:
            for line in f.readlines():
                splitline = line.split()
                (idx, raw_attribs) = splitline[0], ' '.join(splitline[1:])
                (attrib, adjective) = raw_attribs.split('::')
                # clean attribute
                attrib = attrib[4:]
                attrib = attrib.replace('_', ' ')
                # clean adjective
                adjective = re.sub(pattern=r'\(.*\)', repl='', string=adjective)
                adjective = ' '.join(adjective.split())
                # add to dictionary
                attributes[idx] = (adjective, attrib)
        return attributes

    def _load_certainties(self) -> Dict[str, str]:
        certainties = dict()
        with open(f"{self._captions_dir}/certainties.txt") as f:
            for line in f.readlines():
                (idx, certainty) = line.split()
                certainties[idx] = certainty
        return certainties

    def _load_imageIndex(self) -> Dict[str, str]:
        img_index = dict()
        with open(f"{self._captions_dir}/images-dirs.txt") as f:
            for line in f.readlines():
                (idx, imgpath) = line.split()
                img_index[idx] = imgpath
        return img_index

    def _make_best_captions(self) -> None:
        for capcollection in self.img2cap.values():
            capcollection.make_best_caption()
