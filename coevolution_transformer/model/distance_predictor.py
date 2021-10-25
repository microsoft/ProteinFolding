import torch
from torch import nn


class DistancePredictor(nn.Module):
    def __init__(self, ninp):
        super(DistancePredictor, self).__init__()
        nhid = 256
        self.fc_hid = nn.Sequential(nn.Linear(ninp, nhid), nn.ReLU())
        self.fc_cbcb = nn.Linear(nhid, 37)

    def forward(self, x):
        """
        x has shape (B, L, L, C)
        """
        x = self.fc_hid(x)
        return torch.softmax(self.fc_cbcb(x), dim=-1)
