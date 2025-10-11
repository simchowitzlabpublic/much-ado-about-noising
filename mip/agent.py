"""Torch training agent for behavior cloning."""

from copy import deepcopy

import loguru
import torch
import torch.nn as nn

from mip.config import Config
from mip.flow_map import FlowMap
from mip.interpolant import Interpolant
from mip.losses import get_loss_fn
from mip.network_utils import get_encoder, get_network
from mip.torch_utils import report_parameters
from mip.samplers import get_sampler


class TrainingAgent:
    """Training agent for behavior cloning with flow matching."""

    def __init__(
        self,
        config: Config,
    ):
        """Initialize the training agent.

        Args:
            flow_map: The flow map model
            encoder: The observation encoder
            config: Full configuration object
        """
        self.config = config
        self.loss_fn = get_loss_fn(config.optimization.loss_type)
        self.sampler = get_sampler(config.optimization.loss_type)
        self.interpolant = Interpolant(config.optimization.interp_type)
        net = get_network(config.network, config.task)
        report_parameters(net, model_name="Action Network")
        self.flow_map = FlowMap(net).to(config.optimization.device)
        self.encoder = get_encoder(config.network, config.task).to(
            config.optimization.device
        )
        report_parameters(self.encoder, model_name="Encoder Network")
        self.encoder_ema = deepcopy(self.encoder).requires_grad_(False)
        self.flow_map_ema = deepcopy(self.flow_map).requires_grad_(False)

        params = list(self.encoder.parameters()) + list(self.flow_map.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.optimization.lr,
            weight_decay=config.optimization.weight_decay,
        )

    def update(self, act: torch.Tensor, obs: torch.Tensor, delta_t: torch.Tensor):
        """Update the model parameters with a training batch.

        Args:
            act: Action tensor of shape (batch_size, Ta, act_dim)
            obs: Observation tensor of shape (batch_size, To, obs_dim)
            delta_t: Time step differences of shape (batch_size,)

        Returns:
            Dictionary containing loss and gradient norm statistics
        """
        # update optimizer
        loss, info = self.loss_fn(
            self.config.optimization,
            self.flow_map,
            self.encoder,
            self.interpolant,
            act,
            obs,
            delta_t,
        )
        loss.backward()

        params = list(self.encoder.parameters()) + list(self.flow_map.parameters())
        grad_norm = (
            nn.utils.clip_grad_norm_(params, self.config.optimization.grad_clip_norm)
            if self.config.optimization.grad_clip_norm
            else torch.zeros(1)
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.config.optimization.ema_rate < 1:
            self.ema_update()

        return {"loss": loss.item(), "grad_norm": grad_norm.item(), **info}

    def ema_update(self):
        """Update exponential moving average parameters."""
        params = list(self.encoder.parameters()) + list(self.flow_map.parameters())
        ema_params = list(self.encoder_ema.parameters()) + list(
            self.flow_map_ema.parameters()
        )
        with torch.no_grad():
            for p, p_ema in zip(params, ema_params, strict=False):
                p_ema.data.mul_(self.config.optimization.ema_rate).add_(
                    p.data, alpha=1.0 - self.config.optimization.ema_rate
                )

    def sample(
        self,
        act_0: torch.Tensor,
        obs: torch.Tensor,
        num_steps: int = -1,
        use_ema: bool = True,
    ):
        """Sample actions from the learned policy.

        Args:
            act_0: Initial action tensor of shape (batch_size, Ta, act_dim)
            obs: Observation tensor of shape (batch_size, To, obs_dim)
            num_steps: Number of sampling steps (default: use config value)
            use_ema: Whether to use EMA parameters for sampling

        Returns:
            Sampled action tensor of shape (batch_size, Ta, act_dim)
        """
        # manually set num_steps if needed
        if num_steps >= 1:
            config = deepcopy(self.config.optimization)
            config.num_steps = int(num_steps)
        else:
            config = self.config.optimization
        # choose model
        if self.config.optimization.ema_rate < 1:
            if use_ema:
                flow_map = self.flow_map_ema
                encoder = self.encoder_ema
            else:
                flow_map = self.flow_map
                encoder = self.encoder
        else:
            flow_map = self.flow_map
            encoder = self.encoder
        with torch.no_grad():
            act = self.sampler(config, flow_map, encoder, act_0, obs)
        return act

    def save(self, path: str):
        """Save agent models to path."""
        # save flow map, encoder, encoder_ema, flow_map_ema
        torch.save(
            {
                "flow_map": self.flow_map.state_dict(),
                "encoder": self.encoder.state_dict(),
                "encoder_ema": self.encoder_ema.state_dict(),
                "flow_map_ema": self.flow_map_ema.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load agent models from path."""
        # load flow map, encoder, encoder_ema, flow_map_ema
        state_dict = torch.load(path)
        self.flow_map.load_state_dict(state_dict["flow_map"])
        self.encoder.load_state_dict(state_dict["encoder"])
        self.encoder_ema.load_state_dict(state_dict["encoder_ema"])
        self.flow_map_ema.load_state_dict(state_dict["flow_map_ema"])

    def eval(self):
        """Set all models to evaluation mode."""
        self.flow_map.eval()
        self.encoder.eval()
        self.flow_map_ema.eval()
        self.encoder_ema.eval()

    def train(self):
        """Set all models to training mode."""
        self.flow_map.train()
        self.encoder.train()
        self.flow_map_ema.train()
        self.encoder_ema.train()
