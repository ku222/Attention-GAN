
import torch
from torch import nn
import numpy as np

from networks.attention import func_attention


class WordsLoss:
    """
    Loss between words and images
    """
    def __init__(self, device: torch.device, gamma1=4.0, gamma2=5.0, gamma3=10.0, wlambda=5.0):
        self.device = device
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.wlambda = wlambda

    def cosine_similarity(self, x1, x2, dim=1, eps=1e-8):
        """
        Returns cosine similarity between x1 and x2, computed along dim.
        """
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

    def get_loss(self, img_features, words_emb, labels, cap_lens, class_ids):
        """
        Params:
            words_emb(query): batch x nef x seq_len
            img_features(context): batch x nef x 17 x 17
            labels: (batch,)
            cap_lens: (batch,)
            class_ids: (batch,)
        """
        masks = []
        att_maps = []
        similarities = []
        cap_lens = cap_lens.data.tolist()
        batch_size = len(cap_lens)
        for i in range(batch_size):
            if class_ids is not None:
                mask = (class_ids == class_ids[i]).astype(np.uint8)
                mask[i] = 0
                masks.append(mask.reshape((1, -1)))
            # Get the i-th text description
            words_num = cap_lens[i]
            # -> 1 x nef x words_num
            word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
            # -> batch_size x nef x words_num
            word = word.repeat(batch_size, 1, 1)
            # batch x nef x 17*17
            context = img_features
            """
                word(query): batch x nef x words_num
                context: batch x nef x 17 x 17
                weiContext: batch x nef x words_num
                attn: batch x words_num x 17 x 17
            """
            weiContext, attn = func_attention(word, context, gamma1=self.gamma1)
            att_maps.append(attn[i].unsqueeze(0).contiguous())
            # --> batch_size x words_num x nef
            word = word.transpose(1, 2).contiguous()
            weiContext = weiContext.transpose(1, 2).contiguous()
            # --> batch_size*words_num x nef
            word = word.view(batch_size * words_num, -1)
            weiContext = weiContext.view(batch_size * words_num, -1)
            #
            # -->batch_size*words_num
            row_sim = self.cosine_similarity(word, weiContext)
            # --> batch_size x words_num
            row_sim = row_sim.view(batch_size, words_num)

            # Eq. (10)
            row_sim.mul_(self.gamma2).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)

            # --> 1 x batch_size
            # similarities(i, j): the similarity between the i-th image and the j-th text description
            similarities.append(row_sim)

        # batch_size x batch_size
        similarities = torch.cat(similarities, 1)
        if class_ids is not None:
            masks = np.concatenate(masks, 0)
            # masks: batch_size x batch_size
            masks = torch.ByteTensor(masks)
            masks = masks.to(self.device)

        similarities = similarities * self.gamma3
        if class_ids is not None:
            similarities.data.masked_fill_(masks, -float('inf'))
        similarities1 = similarities.transpose(0, 1)
        print(similarities)
        print(labels)
        # Final losses
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
        # Combine
        wloss = (loss0 + loss1) * self.wlambda
        return (wloss, att_maps)