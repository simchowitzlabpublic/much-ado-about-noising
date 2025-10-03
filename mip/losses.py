"""Losses for iterative policy training."""

from collections.abc import Callable

import torch

from mip.config import OptimizationConfig
from mip.encoders import BaseEncoder
from mip.flow_map import FlowMap
from mip.interpolant import Interpolant


def get_norm(x: torch.Tensor, norm_type: str) -> torch.Tensor:
    if norm_type == "l2":
        return torch.norm(x, p=2, dim=-1)
    elif norm_type == "l1":
        return torch.norm(x, p=1, dim=-1)
    else:
        raise NotImplementedError(f"Norm type {norm_type} not implemented.")


def get_loss_fn(loss_type: str) -> Callable:
    if loss_type == "flow":
        return flow_loss
    elif loss_type == "regression":
        return regression_loss
    elif loss_type == "tsd":
        return tsd_loss
    elif loss_type == "mip":
        return mip_loss
    elif loss_type == "lmd":
        return lmd_loss
    elif loss_type == "ctm":
        return ctm_loss
    else:
        raise NotImplementedError(f"Loss type {loss_type} not implemented.")


def flow_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Flow model loss, matching the velocity field

    Args:
        flow_map (FlowMap): the flow map
        interp (Interpolant): the interpolant
        obs (torch.Tensor): the target state
        obs (torch.Tensor): the label
        delta_t (torch.Tensor): the time step difference, used for flow map / shortcut model / consistency training only.

    Returns:
        float: the loss
    """
    # sample
    t = torch.rand_like(delta_t, device=delta_t.device)
    act_0 = torch.randn_like(act, device=act.device)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t, act_0, act_1)
    b_t = flow_map.get_velocity(t, act_t, obs_emb)

    # compute loss
    loss = get_norm(b_t - act_t_dot, config.norm_type) ** 2
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}


def regression_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Standard regression loss."""
    # sample
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)

    # compute loss
    loss = get_norm(act_pred - act, config.norm_type) ** 2
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}


def tsd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Two step denoising loss."""
    # sample
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_0 = torch.randn_like(act, device=act.device)
    noise = torch.randn_like(act, device=act.device)
    act_t = act + (1 - config.t_two_step) * noise

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # compute loss
    loss0 = (get_norm(act_pred_0 - act_t, config.norm_type) / (config.t_two_step)) ** 2
    loss1 = (
        get_norm(act_pred_1 - act, config.norm_type) / (1 - config.t_two_step)
    ) ** 2
    loss = loss0 + loss1
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def mip_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Two step denoising loss."""
    # sample
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    # major difference compared to tsd: remove stochasticity in input
    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.randn_like(act, device=act.device)
    act_t = act + (1 - config.t_two_step) * noise

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # compute loss
    # difference compared to tsd: no stochasticity in prediction
    loss0 = (get_norm(act_pred_0 - act, config.norm_type) / (config.t_two_step)) ** 2
    loss1 = (
        get_norm(act_pred_1 - act, config.norm_type) / (1 - config.t_two_step)
    ) ** 2
    loss = loss0 + loss1
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def lmd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Lagrangian map matching loss for distillation."""
    # sample
    temp_batch_1 = torch.rand_like(delta_t, device=delta_t.device)
    temp_batch_2 = torch.rand_like(delta_t, device=delta_t.device)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)
    act_0 = torch.randn_like(act, device=act.device)
    act_1 = act

    # get condition
    label = encoder(obs, None)

    # predict
    Is = interp.calc_It(s, act_0, act_1)
    Xst_Is, dt_Xst = flow_map.jvp_t(s, t, Is, label)

    # compute the target velocity field
    b_eval = flow_map.get_reference_velocity(t, Xst_Is, label)

    # lmd loss
    loss = torch.mean(
        (dt_Xst.flatten(start_dim=1) - b_eval.flatten(start_dim=1)) ** 2, dim=-1
    )
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def ctm_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Consistency trajectory model loss."""
    # sample
    temp_batch_1 = torch.rand_like(delta_t, device=delta_t.device)
    temp_batch_2 = torch.rand_like(delta_t, device=delta_t.device)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)
    s_plus = s + config.discrete_dt
    t = torch.maximum(t, s_plus)
    act_0 = torch.randn_like(act, device=act.device)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    Is = interp.calc_It(s, act_0, act_1)

    # compute the CTM loss
    Xst_Is_pred = flow_map(s, t, Is, obs_emb)
    b_s = flow_map.get_reference_velocity(s, Is, obs_emb)
    Is_plus = Is + config.discrete_dt * b_s
    Xst_Is_target = flow_map(s_plus, t, Is_plus, obs_emb)
    # make sure loss is not too small
    loss = config.loss_scale * torch.mean(
        ((Xst_Is_target - Xst_Is_pred) / config.discrete_dt) ** 2
    )

    return loss, {}
