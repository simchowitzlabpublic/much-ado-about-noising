"""
Losses for iterative policy training.
"""

import torch
import torch.nn as nn

from mip.flow_map import FlowMap
from mip.interpolant import Interpolant


def flow_loss(
    flow_map: FlowMap,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """
    Flow model loss, matching the velocity field

    Args:
        flow_map (FlowMap): the flow map
        interp (Interpolant): the interpolant
        obs (torch.Tensor): the target state
        obs (torch.Tensor): the label
        delta_t (torch.Tensor): the time step difference, used for flow map / shortcut model / consistency training only.

    Returns:
        float: the loss
    """
    t = torch.rand_like(delta_t, device=delta_t.device)
    # get condition
    label = self.model["condition"](label, None)

    weight_tt = self.flow_map.calc_weight(t, t)
    It = self.interp.calc_It(t, x0, x1)
    It_dot = self.interp.calc_It_dot(t, x0, x1)
    dt_Xtt, _ = self.velocity_field(It, t, t, label)
    loss = torch.mean(
        (dt_Xtt.flatten(start_dim=1) - It_dot.flatten(start_dim=1)) ** 2, dim=-1
    )
    loss = torch.exp(-weight_tt) * loss + weight_tt
    loss = 100.0 * torch.mean(loss)
    return loss, {}
