# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math

import torch
from torch import nn
import torch.nn.functional as F

from .resnet import make_layers


class MHAttention(nn.Module):
    def __init__(self, ninp, nhead, dropout):
        super(MHAttention, self).__init__()
        if ninp % nhead != 0:
            raise ValueError(
                "The hidden size is not a multiple of the number of attention heads"
            )
        self.nhead = nhead
        self.ninp = ninp

        self.fc_query = nn.Linear(ninp, ninp)
        self.fc_key = nn.Linear(ninp, ninp)
        self.fc_value = nn.Linear(ninp, ninp)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        """
        x has shape (*, L, C)
        return shape (*, nhead, L, C/nhead)
        """
        new_shape = x.shape[:-1] + (self.nhead, -1)
        x = x.view(*new_shape)
        return x.transpose(-3, -2)

    def forward_fn(self, x):
        """
        x has shape (*, L, C)
        return shape (*, L, C)
        """
        query = self.transpose_for_scores(self.fc_query(x))
        key = self.transpose_for_scores(self.fc_key(x))
        value = self.transpose_for_scores(self.fc_value(x))

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.ninp / self.nhead)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        x = torch.matmul(attention_weights, value)
        x = x.transpose(-3, -2)
        x = x.reshape(*x.shape[:-2], -1)
        return x

    def forward(self, x):
        chunk_size = 100000 // x.shape[2]
        outputs = []
        for i in range(0, x.shape[1], chunk_size):
            ed = min(i + chunk_size, x.shape[1])
            partial = self.forward_fn(x[:, i:ed])
            outputs.append(partial)
        return torch.cat(outputs, dim=1)


class YAggregator(nn.Module):
    def __init__(self, ninp, nhead, nhid, dim2d, agg_dim):
        super(YAggregator, self).__init__()
        self.nhead = nhead
        self.fc_P = nn.Sequential(
            nn.Linear(ninp, nhead * nhid), nn.LayerNorm(nhead * nhid)
        )
        self.fc_Q = nn.Linear(ninp, nhead * nhid)
        self.projection = nn.Sequential(
            nn.Linear(nhead * nhid * nhid + dim2d, agg_dim),
            nn.LayerNorm(agg_dim),
            nn.ReLU(),
        )

    def transpose_for_attention(self, x):
        """
        x: (B, K, L, C)

        return: (B, nhead, K, L * C/nhead)
        """
        B, K, L, C = x.shape
        x = x.reshape(B, K, L, self.nhead, -1).permute(0, 3, 1, 2, 4)
        x = x.reshape(*x.shape[:3], -1)
        return x

    def aggregate(self, P, Q):
        """
        P, Q: (B, K, L, C)

        return: (B, L, L, C)
        """
        B, K, L, C = P.shape
        C //= self.nhead
        P = self.transpose_for_attention(P)
        Q = self.transpose_for_attention(Q)
        P = P * Q

        with torch.cuda.amp.autocast(enabled=False):
            X = torch.matmul(P.transpose(-2, -1), P)
            Y = torch.matmul(Q.transpose(-2, -1), Q) + 1e-6
            x = X / Y

        x = (
            x.reshape(B, self.nhead, L, C, L, C)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(B, L, L, -1)
        )
        return x

    def forward(self, x1d, x2d):
        """
        x1d: (B, K, L, C)
        x2d: (B, L, L, C)
        return: (B, L, L, agg_dim)
        """
        P = self.fc_P(x1d)
        Q = self.fc_Q(x1d)
        Q = torch.clamp(Q, -20, 20)
        Q = Q - torch.max(Q, dim=1, keepdims=True)[0]
        Q = torch.exp(Q)
        agg2d = self.aggregate(P, Q)

        if x2d is not None:
            agg2d = torch.cat([agg2d, x2d], dim=-1)
        x2d = self.projection(agg2d)
        return x2d


class ZRefiner(nn.Module):
    def __init__(self, ninp, repeats):
        super(ZRefiner, self).__init__()
        self._blocks = make_layers(ninp, [1, 2, 4, 8], repeats)

    def forward(self, x):
        """
        x has shape (B, L, L, C)
        """
        x = x.permute(0, 3, 1, 2)
        x = self._blocks(x)
        x = x.permute(0, 2, 3, 1)
        return x


class ZAttention(nn.Module):
    def __init__(self, ninp, nhead, dim2d, rn_inp, rn_layers):
        super(ZAttention, self).__init__()
        if ninp % nhead != 0:
            raise ValueError(
                "The hidden size is not a multiple of the number of attention heads"
            )
        self.nhead = nhead
        self.ninp = ninp

        self.agg = YAggregator(
            ninp=ninp, nhead=ninp, nhid=2, dim2d=dim2d, agg_dim=rn_inp
        )
        self.refiner = ZRefiner(ninp=rn_inp, repeats=rn_layers)
        self.fc_value = nn.Linear(ninp, ninp)
        self.projection = nn.Linear(rn_inp, nhead)

    def transpose_for_scores(self, x):
        """
        x has shape (*, L, C)
        return shape (*, nhead, L, C/nhead)
        """
        new_shape = x.shape[:-1] + (self.nhead, -1)
        x = x.view(*new_shape)
        return x.transpose(-3, -2)

    def forward(self, x1d, x2d):
        x2d = self.agg(x1d, x2d)
        x2d = self.refiner(x2d)
        att_map = self.projection(x2d)  # (B, L, L, nhead)
        att_map = att_map[:, None].permute(0, 1, 4, 2, 3)
        att_map = F.softmax(att_map, dim=-1)
        value = self.transpose_for_scores(self.fc_value(x1d))
        x1d = torch.matmul(att_map, value)
        x1d = x1d.transpose(-3, -2)
        x1d = x1d.reshape(*x1d.shape[:-2], -1)
        return x1d, x2d


class FeedForward(nn.Module):
    def __init__(self, ninp, dim_feedforward, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(ninp, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, ninp)
        self.norm1 = nn.LayerNorm(ninp)
        self.norm2 = nn.LayerNorm(ninp)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward_fn(self, x, branch):
        x = x + self.dropout1(branch)
        x = self.norm1(x)
        branch = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(branch)
        x = self.norm2(x)
        return x

    def forward(self, x, branch):
        chunk_size = 100000 // x.shape[2]
        outputs = []
        for i in range(0, x.shape[1], chunk_size):
            ed = min(i + chunk_size, x.shape[1])
            partial = self.forward_fn(x[:, i:ed], branch[:, i:ed])
            outputs.append(partial)
        return torch.cat(outputs, dim=1)


class YBlock(nn.Module):
    def __init__(self, ninp, nhead, dim_feedforward, dropout):
        super(YBlock, self).__init__()
        self.col_attention = MHAttention(ninp=ninp, nhead=nhead, dropout=dropout)
        self.feed_forward = FeedForward(
            ninp=ninp, dim_feedforward=dim_feedforward, dropout=dropout
        )

    def forward(self, x1d, x2d):
        branch = x1d.transpose(-2, -3)
        branch = self.col_attention(branch)
        branch = branch.transpose(-2, -3)
        x1d = self.feed_forward(x1d, branch)
        return x1d, x2d


class ZBlock(nn.Module):
    def __init__(self, ninp, nhead, dim2d, rn_inp, rn_layers, dim_feedforward, dropout):
        super(ZBlock, self).__init__()
        self.row_attention = ZAttention(
            ninp=ninp, nhead=nhead, dim2d=dim2d, rn_inp=rn_inp, rn_layers=rn_layers
        )
        self.feed_forward = FeedForward(
            ninp=ninp, dim_feedforward=dim_feedforward, dropout=dropout
        )

    def forward(self, x1d, x2d):
        branch, x2d = self.row_attention(x1d, x2d)
        x1d = self.feed_forward(x1d, branch)
        return x1d, x2d
