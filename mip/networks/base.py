"""Base network class."""

import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        emb_dim: int,
        n_layers: int,
    ):
        self.act_dim = act_dim
        self.Ta = Ta
        self.obs_dim = obs_dim
        self.To = To
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        super().__init__()

    def forward(self, x):
        raise NotImplementedError
