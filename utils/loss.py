import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1) + 1e-10)
        loss_positive = label * torch.pow(euclidean_distance, 2)
        loss_negative = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return torch.mean(loss_positive + loss_negative) / 2