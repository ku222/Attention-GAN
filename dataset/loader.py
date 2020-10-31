
#%%

import os
import re
from collections import defaultdict
from typing import List, Dict

from config import Config


class DatasetLoader:
    def __init__(self):
        self.images_dir = Config.BIRDS_IMAGES
        self.captions_dir = Config.BIRDS_CAPTIONS
        self.img2cap = self._load_captions()
        
    def _load_captions(self) -> Dict[str, List[str]]:
        img2cap = defaultdict(set)
        attributes = self._load_attributes()
        certainties = self._load_certainties()
        imageIndex = self._load_imageIndex()
        with open(f"{self.captions_dir}/labels.txt") as f:
            for line in f.readlines():
                (imgID, attrID, isPresent, certaintyID, _) = line.split()
                if isPresent == '1' and certainties[certaintyID] == 'Definitely':
                    imgname = imageIndex[imgID]
                    attribute = attributes[attrID]
                    if self._is_desirable(attribute):
                        img2cap[imgname].add(attribute)
        return img2cap
    
    def _is_desirable(self, attribute: str) -> bool:
        valid = False
        desirables = ['back', 'bill', 'beak', 'primary', 'breast']
        for attrib in desirables:
            if attrib in attribute.lower():
                valid = True
        return valid
    
    def _load_attributes(self) -> Dict[str, str]:
        attributes = dict()
        with open(f"{self.captions_dir}/attributes.txt") as f:
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
                attributes[idx] = f"{adjective} {attrib}"
        return attributes
    
    def _load_certainties(self) -> Dict[str, str]:
        certainties = dict()
        with open(f"{self.captions_dir}/certainties.txt") as f:
            for line in f.readlines():
                (idx, certainty) = line.split()
                certainties[idx] = certainty
        return certainties
    
    def _load_imageIndex(self) -> Dict[str, str]:
        img_index = dict()
        with open(f"{self.captions_dir}/images.txt") as f:
            for line in f.readlines():
                (idx, imgname) = line.split()
                img_index[idx] = imgname
        return img_index


#%%

loader = DatasetLoader()

#%%

[(k,v) for (k,v) in loader.img2cap.items()][1]
