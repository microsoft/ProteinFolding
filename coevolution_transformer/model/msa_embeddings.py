import math

import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1 << 13):
        super(PositionalEncoding, self).__init__()
        self.ninp = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)  # (L, C)
        self.register_buffer("pe", pe)

    def forward(self, idx):
        """
        idx: (B, L)
        return: (B, L, C)
        """
        return self.pe[idx]


class MSAEmbeddings(nn.Module):
    def __init__(self, msa_gap, embed_dim, dropout):
        super(MSAEmbeddings, self).__init__()
        self.embed_dim = embed_dim
        self.onehot = nn.Embedding(24, 24)
        self.onehot.weight.data = torch.eye(24)
        self.onehot.weight.requires_grad = False
        self.msa_embeddings = nn.Linear((msa_gap * 2 + 2) * 24 + 2, embed_dim)
        self.position_embeddings = PositionalEncoding(embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_ids, msa_ids, position_ids):
        """
        seq_ids: (B, L)
        msa_ids: (B, K, *, L)
        position_ids: (B, L)
        return: (B, K, L, C)
        """
        B, K, _, L = msa_ids.shape
        seq = self.onehot(seq_ids)
        msa_ids = msa_ids.transpose(-2, -1)
        boundary = msa_ids[..., -2:].float()
        msa = self.onehot(msa_ids[..., :-2]).reshape(B, K, L, -1)
        msa = torch.cat([seq[:, None].repeat(1, msa.shape[1], 1, 1), msa, boundary], dim=-1)
        msa_emb = self.msa_embeddings(msa)
        pos_emb = self.position_embeddings(position_ids)
        embeddings = msa_emb * math.sqrt(self.embed_dim) + pos_emb[:, None]
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
