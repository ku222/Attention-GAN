
import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch import Tensor
from torchvision import models
from torchvision.models import Inception3
from utilities.layers import Layers
from typing import Tuple


class CNNEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.out_dim = out_dim
        # Load pretrained model from Google
        model = self.__load_pretrained()
        # Steal layers
        self.__define_module(model)
        self.__init_trainable_weights()
        
    def __load_pretrained(self) -> Inception3:
        """Loads pretrained Inception3 model from online and freeze weights"""
        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        # freeze weights
        for param in model.parameters():
            param.requires_grad = False
        return model

    def freeze_all_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def __define_module(self, model) -> None:
        """Steal layers from Inception_v3"""
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c
        # Add 2 out heads to compute local + global features
        self.emb_features = Layers.conv1x1(768, self.out_dim)
        self.emb_cnn_code = nn.Linear(2048, self.out_dim)

    def __init_trainable_weights(self) -> None:
        """Initialize weights for newly-added heads"""
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate through Inception model
        
        Returns a Tuple of Tensors with shape:
            local:      (batch, 256, 17, 17)
            global:     (batch, 256)
        """
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x) # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)                       # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)                       # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)                       # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)    # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)                       # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)                       # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)    # 35 x 35 x 192
        x = self.Mixed_5b(x)                            # 35 x 35 x 256
        x = self.Mixed_5c(x)                            # 35 x 35 x 288
        x = self.Mixed_5d(x)                            # 35 x 35 x 288
        x = self.Mixed_6a(x)                            # 17 x 17 x 768
        x = self.Mixed_6b(x)                            # 17 x 17 x 768
        x = self.Mixed_6c(x)                            # 17 x 17 x 768
        x = self.Mixed_6d(x)                            # 17 x 17 x 768
        x = self.Mixed_6e(x)                            # 17 x 17 x 768
        # image region features
        features = x                                    # 17 x 17 x 768
        x = self.Mixed_7a(x)                            # 8 x 8 x 1280
        x = self.Mixed_7b(x)                            # 8 x 8 x 2048
        x = self.Mixed_7c(x)                            # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)              # 1 x 1 x 2048
        x = x.view(x.size(0), -1)                       # 2048
        # global image features
        cnn_code = self.emb_cnn_code(x)                 # 256
        if features is not None:
            features = self.emb_features(features)
        return (features, cnn_code)