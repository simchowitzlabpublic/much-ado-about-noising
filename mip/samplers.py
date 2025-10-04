"""Sampler for different training objectives.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import numpy as np
import torch

from mip.config import OptimizationConfig
from mip.encoders import BaseEncoder
from mip.flow_map import FlowMap
from mip.torch_utils import at_least_ndim


def get_default_step_list(loss_type: str):
    if loss_type in ["flow", "ctm", "lmd"]:
        return 3 ** np.arange(2, -1, -1)
    elif loss_type in ["regression", "mip", "tsd"]:
        return [1]
    else:
        raise NotImplementedError(f"Loss type {loss_type} not implemented.")


def get_sampler(loss_type: str):
    if loss_type == "flow":
        return ode_sampler
    elif loss_type == "regression":
        return regression_sampler
    elif loss_type in ["tsd", "mip"]:
        return mip_sampler
    elif loss_type in ["lmd", "ctm"]:
        return flow_map_sampler
    else:
        raise NotImplementedError(f"Loss type {loss_type} not implemented.")


def ode_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    num_steps = config.num_steps
    sample_mode = config.sample_mode
    t_schedule = np.linspace(0, 1, num_steps + 1)
    if sample_mode == "stochastic":
        act_s = torch.randn_like(act_0, device=act_0.device)
    else:
        act_s = torch.zeros_like(act_0, device=act_0.device)
    obs_emb = encoder(obs, None)
    bs = act_0.shape[0]
    for i in range(num_steps):
        s_val = t_schedule[i]
        t_val = t_schedule[i + 1]
        s = torch.full((bs,), s_val, device=act_0.device)
        t = torch.full((bs,), t_val, device=act_0.device)
        b_s = flow_map.get_velocity(s, act_s, obs_emb)
        s_expanded = at_least_ndim(s, act_s.dim())
        t_expanded = at_least_ndim(t, act_s.dim())
        act_s = act_s + b_s * (t_expanded - s_expanded)
    act = act_s
    return act


def flow_map_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """This function is designed for flow map sampler, i.e. for the distilled shortcut model.

    Args:
        config (OptimizationConfig): the configuration
        flow_map (FlowMap): the flow map
        encoder (BaseEncoder): the encoder
        act_0 (torch.Tensor): the initial action
        obs (torch.Tensor): the observation

    Returns:
        torch.Tensor: the sampled action
    """
    num_steps = config.num_steps
    sample_mode = config.sample_mode
    t_schedule = np.linspace(0, 1, num_steps + 1)
    if sample_mode == "stochastic":
        act_s = torch.randn_like(act_0, device=act_0.device)
    else:
        act_s = torch.zeros_like(act_0, device=act_0.device)
    obs_emb = encoder(obs, None)
    bs = act_0.shape[0]
    for i in range(num_steps):
        s_val = t_schedule[i]
        t_val = t_schedule[i + 1]
        s = torch.full((bs,), s_val, device=act_0.device)
        t = torch.full((bs,), t_val, device=act_0.device)
        act_s = flow_map(s, t, act_s, obs_emb)
    act = act_s
    return act


def regression_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    bs = act_0.shape[0]
    act_zeros = torch.zeros_like(act_0, device=act_0.device)
    t = torch.zeros(bs, device=act_0.device)
    obs_emb = encoder(obs, None)
    act = flow_map.get_velocity(t, act_zeros, obs_emb)
    return act


def mip_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    bs = act_0.shape[0]
    s = torch.zeros((bs,), device=act_0.device)
    t = torch.full((bs,), config.t_two_step, device=act_0.device)

    obs_emb = encoder(obs, None)

    act_0 = torch.zeros_like(act_0, device=act_0.device)
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_pred_0, obs_emb)

    act = act_pred_1
    return act
