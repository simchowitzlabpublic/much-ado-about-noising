"""
Interpolant is a stochastic interpolant with different types of interpolation.
"""

import torch
from typing import Callable, Optional, Union

from mip.torch_utils import at_least_ndim


class Interpolant:
    """Class for a stochastic interpolant with different types of interpolation"""

    def __init__(self, interp_type: str = "linear"):
        """
        Initialize an interpolant with the specified type

        Args:
            interp_type: Type of interpolation ("linear" or "trig")
        """
        if interp_type == "linear":
            self.alpha = lambda t: 1.0 - t
            self.beta = lambda t: t
            self.alpha_dot = lambda _: -1.0
            self.beta_dot = lambda _: 1.0
        elif interp_type == "trig":
            self.alpha = lambda t: (
                torch.cos(torch.tensor(t) * torch.pi / 2)
                if isinstance(t, (int, float))
                else torch.cos(t * torch.pi / 2)
            )
            self.beta = lambda t: (
                torch.sin(torch.tensor(t) * torch.pi / 2)
                if isinstance(t, (int, float))
                else torch.sin(t * torch.pi / 2)
            )
            self.alpha_dot = (
                lambda t: -0.5
                * torch.pi
                * (
                    torch.sin(torch.tensor(t) * torch.pi / 2)
                    if isinstance(t, (int, float))
                    else torch.sin(t * torch.pi / 2)
                )
            )
            self.beta_dot = (
                lambda t: 0.5
                * torch.pi
                * (
                    torch.cos(torch.tensor(t) * torch.pi / 2)
                    if isinstance(t, (int, float))
                    else torch.cos(t * torch.pi / 2)
                )
            )
        else:
            raise NotImplementedError(
                f"Interpolant type '{interp_type}' not implemented."
            )

    def calc_It(
        self, t: Union[float, torch.Tensor], x0: torch.Tensor, x1: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the interpolant at time t"""
        t = at_least_ndim(t, x0.dim())
        return self.alpha(t) * x0 + self.beta(t) * x1

    def calc_It_dot(
        self, t: Union[float, torch.Tensor], x0: torch.Tensor, x1: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the time derivative of the interpolant at time t"""
        t = at_least_ndim(t, x0.dim())
        return self.alpha_dot(t) * x0 + self.beta_dot(t) * x1
