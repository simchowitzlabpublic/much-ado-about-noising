"""Utility functions for setting up, save and load networks.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import loguru
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
        "obs_dim": network_config.emb_dim,  # all encoder will encode obs to emb_dim
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
            nhead=network_config.n_heads,
            num_layers=network_config.num_layers,
            p_drop_emb=network_config.dropout,
            p_drop_attn=network_config.attn_dropout,
            n_cond_layers=network_config.n_cond_layers,
            timestep_emb_type=network_config.timestep_emb_type,
        )
    elif network_config.network_type == "chiunet":
        return network_class(
            **common_params,
            model_dim=network_config.model_dim,
            emb_dim=network_config.emb_dim,
            kernel_size=network_config.kernel_size,
            cond_predict_scale=network_config.cond_predict_scale,
            obs_as_global_cond=network_config.obs_as_global_cond,
            dim_mult=network_config.dim_mult,
            timestep_emb_type=network_config.timestep_emb_type,
        )
    elif network_config.network_type == "jannerunet":
        return network_class(
            **common_params,
            model_dim=network_config.model_dim,
            emb_dim=network_config.emb_dim,
            kernel_size=network_config.kernel_size,
            dim_mult=network_config.dim_mult,
            norm_type=network_config.norm_type,
            attention=network_config.attention,
            timestep_emb_type=network_config.timestep_emb_type,
        )
    elif network_config.network_type in ["rnn", "vanilla_rnn"]:
        rnn_params = {
            **common_params,
            "rnn_hidden_dim": network_config.emb_dim,
            "rnn_num_layers": network_config.num_layers,
            "rnn_type": network_config.rnn_type,
            "dropout": network_config.dropout,
        }
        if network_config.network_type == "rnn":
            rnn_params.update(
                {
                    "timestep_emb_dim": network_config.timestep_emb_dim,
                    "max_freq": network_config.max_freq,
                }
            )
        return network_class(**rnn_params)
    elif network_config.network_type == "sudeepdit":
        return network_class(
            **common_params,
            d_model=network_config.emb_dim,
            n_heads=network_config.n_heads,
            depth=network_config.num_layers,
            dropout=network_config.dropout,
            timestep_emb_type=network_config.timestep_emb_type,
        )


def get_encoder(network_config: NetworkConfig, task_config: TaskConfig):
    if task_config.obs_type == "image":
        # Force image encoder for image observations
        encoder_type = "image"
    elif task_config.obs_type in ["state", "keypoint"]:
        # For state/keypoint, use mlp encoder
        encoder_type = "mlp"
    else:
        raise ValueError(f"Invalid observation type: {task_config.obs_type}")
    loguru.logger.info(f"Using encoder type: {encoder_type}")

    if encoder_type == "identity":
        return IdentityEncoder(dropout=network_config.encoder_dropout)
    elif encoder_type == "mlp":
        return MLPEncoder(
            in_dim=task_config.obs_dim,
            out_dim=network_config.emb_dim,
            hidden_dims=[network_config.emb_dim] * network_config.num_encoder_layers,
            dropout=network_config.encoder_dropout,
        )
    elif encoder_type == "image":
        kwargs = {
            "shape_meta": task_config.shape_meta,
            "rgb_model_name": network_config.rgb_model_name,
            "emb_dim": network_config.emb_dim,
            "use_seq": network_config.use_seq,
            "keep_horizon_dims": network_config.keep_horizon_dims,
            "resize_shape": task_config.resize_shape,
            "crop_shape": task_config.crop_shape,
            "random_crop": task_config.random_crop,
            "use_group_norm": task_config.use_group_norm,
        }
        return MultiImageObsEncoder(**kwargs)
    else:
        raise ValueError(f"Invalid encoder type: {encoder_type}")


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
