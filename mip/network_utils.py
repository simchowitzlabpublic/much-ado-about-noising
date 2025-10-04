"""Utility functions for setting up, save and load networks.

Author: Chaoyi Pan
Date: 2025-10-03
"""

from mip.config import NetworkConfig, TaskConfig
from mip.encoders import IdentityEncoder, MLPEncoder, MultiImageObsEncoder
from mip.networks.mlp import MLP, VanillaMLP


def get_network(network_config: NetworkConfig, task_config: TaskConfig):
    network_class = {
        "mlp": MLP,
        "vanilla_mlp": VanillaMLP,
    }[network_config.network_type]
    return network_class(
        act_dim=task_config.act_dim,
        Ta=task_config.horizon,
        obs_dim=task_config.obs_dim,
        To=task_config.obs_steps,
        emb_dim=network_config.emb_dim,
        n_layers=network_config.num_layers,
        dropout=network_config.dropout,
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
        kwargs = dict(
            shape_meta=task_config.shape_meta,
            rgb_model_name=network_config.rgb_model_name,
            emb_dim=network_config.emb_dim,
            use_seq=network_config.use_seq,
            keep_horizon_dims=network_config.keep_horizon_dims,
        )

        # Add optional image processing configs from task_config
        kwargs['resize_shape'] = task_config.resize_shape
        kwargs['crop_shape'] = task_config.crop_shape
        kwargs['random_crop'] = task_config.random_crop
        kwargs['use_group_norm'] = task_config.use_group_norm

        return MultiImageObsEncoder(**kwargs)
    else:
        raise ValueError(f"Invalid encoder type: {network_config.encoder_type}")
