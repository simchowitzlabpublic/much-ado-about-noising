"""Utility functions for setting up, save and load networks.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import torch
import torch.nn as nn

from mip.config import NetworkConfig, TaskConfig
from mip.encoders import IdentityEncoder, MLPEncoder, MultiImageObsEncoder


def get_network(network_config: NetworkConfig, task_config: TaskConfig):
    # Import inside function to avoid circular imports
    from mip.networks.chitfm import ChiTransformer
    from mip.networks.chiunet import ChiUNet
    from mip.networks.jannerunet import JannerUNet
    from mip.networks.mlp import MLP, VanillaMLP
    from mip.networks.rnn import RNN, VanillaRNN
    from mip.networks.sudeepdit import SudeepDiT

    network_class = {
        "mlp": MLP,
        "vanilla_mlp": VanillaMLP,
        "chitransformer": ChiTransformer,
        "chiunet": ChiUNet,
        "jannerunet": JannerUNet,
        "rnn": RNN,
        "vanilla_rnn": VanillaRNN,
        "sudeepdit": SudeepDiT,
    }[network_config.network_type]

    # Common parameters for all networks
    common_params = {
        "act_dim": task_config.act_dim,
        "Ta": task_config.horizon,
        "obs_dim": task_config.obs_dim,
        "To": task_config.obs_steps,
    }

    if network_config.network_type == "mlp":
        return network_class(
            **common_params,
            emb_dim=network_config.emb_dim,
            n_layers=network_config.num_layers,
            dropout=network_config.dropout,
            timestep_emb_dim=network_config.timestep_emb_dim,
        )
    elif network_config.network_type == "vanilla_mlp":
        return network_class(
            **common_params,
            emb_dim=network_config.emb_dim,
            n_layers=network_config.num_layers,
            dropout=network_config.dropout,
        )
    elif network_config.network_type == "chitransformer":
        return network_class(
            **common_params,
            d_model=network_config.emb_dim,
            nhead=getattr(network_config, "nhead", 4),
            num_layers=network_config.num_layers,
            p_drop_emb=network_config.dropout,
            p_drop_attn=getattr(network_config, "attn_dropout", 0.3),
            n_cond_layers=getattr(network_config, "n_cond_layers", 0),
            timestep_emb_type=getattr(
                network_config, "timestep_emb_type", "positional"
            ),
        )
    elif network_config.network_type == "chiunet":
        return network_class(
            **common_params,
            model_dim=network_config.emb_dim,
            emb_dim=network_config.emb_dim,
            kernel_size=getattr(network_config, "kernel_size", 5),
            cond_predict_scale=getattr(network_config, "cond_predict_scale", True),
            obs_as_global_cond=getattr(network_config, "obs_as_global_cond", True),
            dim_mult=getattr(network_config, "dim_mult", [1, 2, 2]),
            timestep_emb_type=getattr(
                network_config, "timestep_emb_type", "positional"
            ),
        )
    elif network_config.network_type == "jannerunet":
        return network_class(
            **common_params,
            model_dim=getattr(network_config, "model_dim", 32),
            emb_dim=network_config.emb_dim,
            kernel_size=getattr(network_config, "kernel_size", 3),
            dim_mult=getattr(network_config, "dim_mult", [1, 2, 2, 2]),
            norm_type=getattr(network_config, "norm_type", "groupnorm"),
            attention=getattr(network_config, "attention", False),
            timestep_emb_type=getattr(
                network_config, "timestep_emb_type", "positional"
            ),
        )
    elif network_config.network_type in ["rnn", "vanilla_rnn"]:
        rnn_params = {
            **common_params,
            "rnn_hidden_dim": network_config.emb_dim,
            "rnn_num_layers": network_config.num_layers,
            "rnn_type": getattr(network_config, "rnn_type", "LSTM"),
            "dropout": network_config.dropout,
        }
        if network_config.network_type == "rnn":
            rnn_params.update(
                {
                    "timestep_emb_dim": network_config.timestep_emb_dim,
                    "max_freq": getattr(network_config, "max_freq", 100.0),
                }
            )
        return network_class(**rnn_params)
    elif network_config.network_type == "sudeepdit":
        # Ensure n_heads divides emb_dim
        default_n_heads = 4 if network_config.emb_dim % 4 == 0 else 2
        return network_class(
            **common_params,
            d_model=network_config.emb_dim,
            n_heads=getattr(network_config, "n_heads", default_n_heads),
            depth=network_config.num_layers,
            dropout=network_config.dropout,
            timestep_emb_type=getattr(
                network_config, "timestep_emb_type", "positional"
            ),
        )


def get_encoder(network_config: NetworkConfig, task_config: TaskConfig):
    if network_config.encoder_type == "identity":
        return IdentityEncoder(dropout=network_config.encoder_dropout)
    elif network_config.encoder_type == "mlp":
        return MLPEncoder(
            in_dim=task_config.obs_dim,
            out_dim=network_config.emb_dim,
            hidden_dims=[network_config.emb_dim] * network_config.num_layers,
            dropout=network_config.encoder_dropout,
        )
    elif network_config.encoder_type == "image":
        # Pass image-specific configs from task_config if available
        kwargs = {
            "shape_meta": task_config.shape_meta,
            "rgb_model_name": network_config.rgb_model_name,
            "emb_dim": network_config.emb_dim,
            "use_seq": network_config.use_seq,
            "keep_horizon_dims": network_config.keep_horizon_dims,
        }

        # Add optional image processing configs from task_config
        kwargs["resize_shape"] = task_config.resize_shape
        kwargs["crop_shape"] = task_config.crop_shape
        kwargs["random_crop"] = task_config.random_crop
        kwargs["use_group_norm"] = task_config.use_group_norm

        return MultiImageObsEncoder(**kwargs)
    else:
        raise ValueError(f"Invalid encoder type: {network_config.encoder_type}")


class GroupNorm1d(nn.Module):
    def __init__(self, dim, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, dim // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x.unsqueeze(2),
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x.squeeze(2)
