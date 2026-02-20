"""Losses for iterative policy training."""

from collections.abc import Callable

import torch
import torch.nn.functional as F

from mip.config import OptimizationConfig
from mip.encoders import BaseEncoder
from mip.flow_map import FlowMap
from mip.interpolant import Interpolant


def get_norm(x: torch.Tensor, norm_type: str) -> torch.Tensor:
    if norm_type == "l2":
        # squared L2 (no sqrt)
        return torch.sum(x * x, dim=-1)
    elif norm_type == "l1":
        return torch.sum(torch.abs(x), dim=-1)
    elif norm_type == "smooth_l1":
        # per-element smooth L1, then sum over last dim
        return torch.sum(
            F.smooth_l1_loss(x, torch.zeros_like(x), reduction="none"), dim=-1
        )
    else:
        raise NotImplementedError(f"Norm type {norm_type} not implemented.")


def get_loss_fn(loss_type: str) -> Callable:
    if loss_type == "flow":
        return flow_loss
    elif loss_type == "regression":
        return regression_loss
    elif loss_type == "straight_flow":
        return straight_flow_loss
    elif loss_type == "tsd":
        return tsd_loss
    elif loss_type == "mip":
        return mip_loss
    elif loss_type == "lmd":
        return lmd_loss
    elif loss_type == "ctm":
        return ctm_loss
    elif loss_type == "psd":
        return psd_loss
    elif loss_type == "lsd":
        return lsd_loss
    elif loss_type == "esd":
        return esd_loss
    elif loss_type == "mf":
        return mf_loss
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
    """Flow model loss, matching the velocity field.

    Args:
        flow_map (FlowMap): the flow map
        interp (Interpolant): the interpolant
        obs (torch.Tensor): the target state
        obs (torch.Tensor): the label
        delta_t (torch.Tensor): the time step difference, used for flow map / shortcut model / consistency training only.

    Returns:
        float: the loss
    """
    # sample - use empty+uniform_/normal_ for CUDA graph compatibility
    t = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t, act_0, act_1)
    b_t = flow_map.get_velocity(t, act_t, obs_emb)

    # compute loss
    loss = get_norm(b_t - act_t_dot, config.norm_type)
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
    loss = get_norm(act_pred - act, config.norm_type)
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}

def straight_flow_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Straight flow loss."""
    # sample
    t = torch.zeros_like(delta_t, device=delta_t.device)

    # Major difference compared to regression: use random noise instead of zeros
    act_0 = torch.randn_like(act, device=act.device)

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)

    # compute loss
    loss = get_norm(act_pred - act, config.norm_type)
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
    act_0 = torch.empty_like(act).normal_(0, 1)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # compute loss
    loss0 = get_norm((act_pred_0 - act_t) / config.t_two_step, config.norm_type)
    loss1 = get_norm((act_pred_1 - act) / (1 - config.t_two_step), config.norm_type)
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
    """Minimum iterative policy loss."""
    # sample
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    # major difference compared to tsd: remove stochasticity in input
    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise

    # NOTE: in paper, we use
    # act_t = config.t_two_step * act + (1 - config.t_two_step) * noise
    # but we found that this is not necessary when config.t_two_step close to 1.
    # feel free to use the original form if you want to, you can refer to mip_origin_loss

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    # for first step, scale network output by t_two_step to match the scale of the second step
    # equivalent form: directly let first step predict act
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # compute loss
    # difference compared to tsd: no stochasticity in prediction
    loss0 = get_norm((act_pred_0 - act) / config.t_two_step, config.norm_type)
    loss1 = get_norm((act_pred_1 - act) / (1 - config.t_two_step), config.norm_type)
    loss = loss0 + loss1
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def mip_origin_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Minimum iterative policy loss (original form with scale in first iteration)."""
    # sample
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    # major difference compared to tsd: remove stochasticity in input
    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = config.t_two_step * act + (1 - config.t_two_step) * noise

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    # for first step, scale network output by t_two_step to match the scale of the second step
    # equivalent form: directly let first step predict act
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_target_0 = config.t_two_step * act
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)
    act_target_1 = act

    # compute loss
    # difference compared to tsd: no stochasticity in prediction
    loss0 = get_norm((act_pred_0 - act_target_0) / config.t_two_step, config.norm_type)
    loss1 = get_norm(
        (act_pred_1 - act_target_1) / (1 - config.t_two_step), config.norm_type
    )
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
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)
    act_0 = torch.empty_like(act).normal_(0, 1)
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
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)
    s_plus = s + config.discrete_dt
    t = torch.maximum(t, s_plus)
    act_0 = torch.empty_like(act).normal_(0, 1)
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


def psd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Progressive Self-Distillation loss combined with flow matching.

    This loss combines:
    1. Standard flow matching loss
    2. PSD term that encourages consistency between single-step and multi-step predictions

    The PSD term uses uniform weighting between intermediate steps.
    """
    # ========== Flow matching loss ==========
    # sample
    t_flow = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t_flow, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t_flow, act_0, act_1)
    b_t = flow_map.get_velocity(t_flow, act_t, obs_emb)

    # compute flow loss
    flow_matching_loss = get_norm(b_t - act_t_dot, config.norm_type)
    flow_matching_loss = config.loss_scale * torch.mean(flow_matching_loss)

    # ========== PSD term ==========
    # sample s, t, u like lmd loss
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)

    # sample u uniformly between s and t
    h = torch.empty_like(delta_t).uniform_(0, 1)
    u = s + h * (t - s)

    # get interpolated starting point
    Is = interp.calc_It(s, act_0, act_1)

    # compute full jump s -> t (student)
    _, f_xst = flow_map.get_map_and_velocity(s, t, Is, obs_emb)

    # compute two-step jump s -> u -> t (teacher, no stopgrad)
    xsu, f_xsu = flow_map.get_map_and_velocity(s, u, Is, obs_emb)
    _, f_xut = flow_map.get_map_and_velocity(u, t, xsu, obs_emb)

    # uniform PSD: teacher = (1 - h) * phi_su + h * phi_ut
    # where h is the relative position of u between s and t
    student = f_xst
    # expand h to match f_xsu dimensions: [batch, horizon, act_dim]
    h_expanded = h.view(-1, 1, 1)
    teacher = (1 - h_expanded) * f_xsu + h_expanded * f_xut

    # compute PSD loss using get_norm (ignore weight_st as requested)
    psd_term = get_norm(student - teacher, config.norm_type)
    psd_term = config.loss_scale * torch.mean(psd_term)

    # combine losses
    total_loss = flow_matching_loss + psd_term

    return total_loss, {
        "flow_loss": flow_matching_loss.item(),
        "psd_term": psd_term.item(),
    }


def lsd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Lagrangian self-distillation loss combined with flow matching.

    This loss combines:
    1. Standard flow matching loss
    2. LSD term that encourages consistency in the velocity field

    The LSD term uses uniform sampling between s and t without stopgrad.
    """
    # ========== Flow matching loss ==========
    # sample
    t_flow = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t_flow, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t_flow, act_0, act_1)
    b_t = flow_map.get_velocity(t_flow, act_t, obs_emb)

    # compute flow loss
    flow_matching_loss = get_norm(b_t - act_t_dot, config.norm_type)
    flow_matching_loss = config.loss_scale * torch.mean(flow_matching_loss)

    # ========== LSD term ==========
    # sample s, t like lmd loss
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)

    # get interpolated starting point
    Is = interp.calc_It(s, act_0, act_1)

    # compute Xst and dt_Xst using jvp_t
    xst, dt_xst = flow_map.jvp_t(s, t, Is, obs_emb)

    # compute the velocity field at the endpoint (no stopgrad)
    b_eval = flow_map.get_velocity(t, xst, obs_emb)

    # lsd loss (ignore weight_st)
    error = b_eval - dt_xst
    lsd_term = get_norm(error, config.norm_type)
    lsd_term = config.loss_scale * torch.mean(lsd_term)

    # combine losses
    total_loss = flow_matching_loss + lsd_term

    return total_loss, {
        "flow_loss": flow_matching_loss.item(),
        "lsd_term": lsd_term.item(),
    }


def esd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Euler self-distillation loss."""
    # ========== Flow matching loss ==========
    # sample
    t_flow = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t_flow, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t_flow, act_0, act_1)
    b_t = flow_map.get_velocity(t_flow, act_t, obs_emb)

    # compute flow loss
    flow_matching_loss = get_norm(b_t - act_t_dot, config.norm_type)
    flow_matching_loss = config.loss_scale * torch.mean(flow_matching_loss)

    # ========== ESD term ==========
    # sample s, t like lmd loss
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)

    # get interpolated starting point
    Is = interp.calc_It(s, act_0, act_1)

    # compute Xst and ds_Xst using jvp_t
    xst, ds_xst = flow_map.jvp_s(s, t, Is, obs_emb)

    # compute the velocity field at the endpoint (stopgrad)
    with torch.no_grad():
        b_eval = flow_map.get_velocity(t, xst, obs_emb)

    # compute jvp
    _, grad_xst_b = flow_map.jvp_x(s, t, Is, b_eval, obs_emb)

    # esd loss
    error = ds_xst + grad_xst_b
    esd_term = get_norm(error, config.norm_type)
    esd_term = config.loss_scale * torch.mean(esd_term)

    # combine losses
    total_loss = flow_matching_loss + esd_term

    return total_loss, {
        "flow_loss": flow_matching_loss.item(),
        "esd_term": esd_term.item(),
    }


def mf_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Mean flow loss."""
    # ========== Flow matching loss ==========
    # sample
    t_flow = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t_flow, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t_flow, act_0, act_1)
    b_t = flow_map.get_velocity(t_flow, act_t, obs_emb)

    # compute flow loss
    flow_matching_loss = get_norm(b_t - act_t_dot, config.norm_type)
    flow_matching_loss = config.loss_scale * torch.mean(flow_matching_loss)

    # ========== Mean flow term ==========
    # sample s, t
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)

    # get interpolated starting point
    Is = interp.calc_It(s, act_0, act_1)
    dot_Is = interp.calc_It_dot(s, act_0, act_1)

    # compute Xst and ds_Xst using jvp_t
    xst, ds_xst = flow_map.jvp_s(s, t, Is, obs_emb)

    # compute the velocity field at the endpoint (stopgrad)
    with torch.no_grad():
        # Difference 1: use dot_Is instead of b_eval
        # Difference 2: also disable gradient for jvp_x
        # compute jvp
        _, grad_xst_b = flow_map.jvp_x(s, t, Is, dot_Is, obs_emb)

    # mf loss
    error = ds_xst + grad_xst_b
    mf_term = get_norm(error, config.norm_type)
    mf_term = config.loss_scale * torch.mean(mf_term)

    # combine losses
    total_loss = flow_matching_loss + mf_term

    return total_loss, {
        "flow_loss": flow_matching_loss.item(),
        "mf_term": mf_term.item(),
    }
