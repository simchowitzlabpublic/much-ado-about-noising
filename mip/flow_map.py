"""Flow map is a neural network wrapper providing partial derivatives of the neural network.

Author: Chaoyi Pan
Date: 2025-04-18
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mip.torch_utils import at_least_ndim


class FlowMap(nn.Module):
    """FlowMap wrapper for neural networks with partial derivative computation. Designed for stochastic interpolant framework."""

    def __init__(
        self,
        net: nn.Module,
    ):
        super().__init__()
        self.net = net

    def forward(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        xs: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the flow map. Parameterizing it with the stochastic interpolant framework."""
        f_xst, _ = self.net(xs, s, t, label)
        ss = at_least_ndim(s, xs.dim())
        ts = at_least_ndim(t, xs.dim())
        xst = xs + (ts - ss) * f_xst
        return xst

    def forward_single(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        xs: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the flow map without predicting the scalar output."""
        s_batch = s.unsqueeze(0)
        t_batch = t.unsqueeze(0)
        xs_batch = xs.unsqueeze(0)
        label_batch = label.unsqueeze(0) if label is not None else None
        x = self.forward(s_batch, t_batch, xs_batch, label_batch)[0]
        return x

    def jvp_t_single(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        xs: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Jacobian-vector product with respect to t for a single sample.

        Uses jvp to compute the forward pass and partial derivative with respect to t
        at the same time, which can save computation.
        """

        def f_single_wrapped(t_input):
            return self.forward_single(s, t_input, xs, label)

        return torch.func.jvp(
            f_single_wrapped,
            (t,),
            (torch.tensor(1.0, device=xs.device),),
        )

    def jvp_t(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        xs: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute batched Jacobian-vector product with respect to t using vmap."""
        return torch.func.vmap(self.jvp_t_single)(s, t, xs, label)

    def get_velocity(self, t, xs, label):
        """Get the velocity field of the flow."""
        bt, _ = self.net(xs, t, t, label)
        return bt
