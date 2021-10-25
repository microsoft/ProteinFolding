import math

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from .base_model import BaseModel
from .msa_embeddings import MSAEmbeddings
from .attention import ZBlock, YBlock, YAggregator, ZRefiner
from .distance_predictor import DistancePredictor


class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__()
        self.msa_embeddings = MSAEmbeddings(msa_gap=7, embed_dim=128, dropout=0.1)
        self.blocks = nn.ModuleList()
        nblocks = 6
        for i in range(nblocks):
            self.blocks.append(
                ZBlock(
                    ninp=128,
                    nhead=8,
                    dim2d=0 if i == 0 else 96,
                    rn_inp=96,
                    rn_layers=12,
                    dim_feedforward=256,
                    dropout=0.1,
                )
            )

            self.blocks.append(
                YBlock(ninp=128, nhead=4, dim_feedforward=256, dropout=0.1)
            )
        self.aggregator = YAggregator(ninp=128, nhead=32, nhid=8, dim2d=96, agg_dim=96)
        self.refiner = ZRefiner(ninp=96, repeats=12)

        self.distance_predictor = DistancePredictor(ninp=96)

    def forward(self, data):
        x1d = self.msa_embeddings(data["seq"], data["msa"], data["index"])
        x2d = None
        for i, model_fn in enumerate(tqdm(self.blocks)):
            x1d, x2d = model_fn(x1d, x2d)
        x2d = self.aggregator(x1d, x2d)
        x2d = self.refiner(x2d)
        return self.distance_predictor(x2d)
