
#%%%

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

# Load the pretrained model

class ImageEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.freeze_model()
        self.layer = self.model._modules.get('avgpool')
        self.cuda()

    def freeze_model(self) -> None:
        for (n, l) in self.model.named_parameters():
            l.requires_grad = False
        self.model.eval()

    def embed(self, images: Tensor, batch_size=32) -> Tensor:
        t_dataset = TensorDataset(images)
        loader = DataLoader(t_dataset, shuffle=False, batch_size=batch_size)
        # Embed inputs
        output = []
        with torch.no_grad():
            for batch in loader:
                batch = batch[0].cuda()
                embeddings = self.model(batch).squeeze(3).squeeze(2)
                output.append(embeddings.detach().cpu())
        return torch.cat(output, dim=0)