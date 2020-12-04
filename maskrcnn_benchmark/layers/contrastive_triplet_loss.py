import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    output1/output2: embeddings nx2
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        # check if cuda available
        target = target.cuda()
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target * distances +
                        (1 + -1 * target) * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()
    
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings [N*dim_embed] of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
#         print('============',distance_positive)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
#         print(distance_negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        if size_average:
            loss = losses.mean()
        else:
            loss = losses.sum()
        return loss