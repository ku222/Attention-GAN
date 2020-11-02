
import torch
from torch import nn
import numpy as np

class SentenceLoss:
    def __init__(self, device: torch.device, gamma3=10.0, slambda=5.0):
        self.device = device
        self.gamma3 = gamma3
        self.slambda = slambda

    def get_loss(self, cnn_code, rnn_code, labels, class_ids, eps=1e-8):
        # ### Mask mis-match samples  ###
        # that come from the same class as the real sample ###
        masks = []
        batch_size = len(labels)
        if class_ids is not None:
            for i in range(batch_size):
                mask = (class_ids == class_ids[i]).astype(np.uint8)
                mask[i] = 0
                masks.append(mask.reshape((1, -1)))
            masks = np.concatenate(masks, 0)
            # masks: batch_size x batch_size
            masks = torch.ByteTensor(masks)
            masks = masks.to(self.device)

        # --> seq_len x batch_size x nef
        if cnn_code.dim() == 2:
            cnn_code = cnn_code.unsqueeze(0)
            rnn_code = rnn_code.unsqueeze(0)

        # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
        cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
        rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
        # scores* / norm*: seq_len x batch_size x batch_size
        scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
        norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
        scores0 = scores0 / norm0.clamp(min=eps) * self.gamma3

        # --> batch_size x batch_size
        scores0 = scores0.squeeze()
        if class_ids is not None:
            scores0.data.masked_fill_(masks, -float('inf'))
        scores1 = scores0.transpose(0, 1)
        # Final losses
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
        # Combine
        sloss = (loss0 + loss1) * self.slambda
        return sloss