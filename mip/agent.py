"""Torch training agent for behavior cloning."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import loguru
import torch
import torch.nn as nn
from tensordict import TensorDict

from mip.config import Config
from mip.flow_map import FlowMap
from mip.interpolant import Interpolant
from mip.losses import get_loss_fn
from mip.network_utils import get_encoder, get_network
from mip.samplers import get_sampler
from mip.torch_utils import report_parameters


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

        # Create detached models for CUDA graphs (if enabled)
        self.use_cudagraphs = config.optimization.use_cudagraphs
        if self.use_cudagraphs:
            self.flow_map_detach = deepcopy(self.flow_map).requires_grad_(False)
            self.encoder_detach = deepcopy(self.encoder).requires_grad_(False)
            self.flow_map_ema_detach = deepcopy(self.flow_map_ema).requires_grad_(False)
            self.encoder_ema_detach = deepcopy(self.encoder_ema).requires_grad_(False)
        else:
            self.flow_map_detach = None
            self.encoder_detach = None
            self.flow_map_ema_detach = None
            self.encoder_ema_detach = None

        params = list(self.encoder.parameters()) + list(self.flow_map.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.optimization.lr,
            weight_decay=config.optimization.weight_decay,
        )

        # Compile training and sampling functions for faster execution
        self.use_compile = config.optimization.use_compile
        self.compile_mode = config.optimization.compile_mode
        self.__compile__()

    def __compile__(self):
        """Compile training and inference functions."""
        loguru.logger.info(
            f"Compile: {self.use_compile} | "
            f"Compile mode: {self.compile_mode} | "
            f"CUDA graphs: {self.use_cudagraphs}"
        )

        # Create the update function that includes optimizer operations
        self._update_impl = self._create_update_impl()
        self._sample_fn = self._sample_impl

        # Step 1: Setup CUDA graphs - copy params to detached models
        if self.use_cudagraphs:
            from tensordict import from_module

            loguru.logger.info("Setting up CUDA graphs - copying parameters to detached models")
            # Copy params to detached models without gradients
            from_module(self.flow_map).data.to_module(self.flow_map_detach)
            from_module(self.encoder).data.to_module(self.encoder_detach)
            from_module(self.flow_map_ema).data.to_module(self.flow_map_ema_detach)
            from_module(self.encoder_ema).data.to_module(self.encoder_ema_detach)

            # Wrap sampler with context manager to use detached models
            self._sample_fn = self._inference_mode()(self._sample_impl)

        # Step 2: Compile with torch.compile
        if self.use_compile:
            loguru.logger.info(
                "Compiling entire update loop (forward+backward+optimizer) with torch.compile"
            )
            self._compiled_update = torch.compile(
                self._update_impl, mode=self.compile_mode
            )
            loguru.logger.info("Compiling sampler with torch.compile")
            self._compiled_sampler = torch.compile(
                self._sample_fn, mode=self.compile_mode
            )
            loguru.logger.info("Successfully compiled models")
        else:
            self._compiled_update = self._update_impl
            self._compiled_sampler = self._sample_fn

        # Step 3: Wrap with CudaGraphModule for CUDA graph capture
        if self.use_cudagraphs:
            from tensordict.nn import CudaGraphModule

            loguru.logger.info("Wrapping update function with CudaGraphModule for CUDA graph capture")
            # Wrap the update function with CudaGraphModule
            # in_keys=[] means no input TensorDict, out_keys=[] means output TensorDict keys are inferred
            self._compiled_update = CudaGraphModule(
                self._compiled_update, in_keys=[], out_keys=[]
            )
            loguru.logger.info("CUDA graph setup complete")

    @contextmanager
    def _inference_mode(self):
        """Context manager to switch to inference models.

        For CUDA graphs: Swaps to detached models (no gradients).
        For regular mode: Sets models to eval mode temporarily.
        """
        if self.use_cudagraphs:
            # Swap to detached models for CUDA graphs
            flow_map_backup = self.flow_map
            encoder_backup = self.encoder
            flow_map_ema_backup = self.flow_map_ema
            encoder_ema_backup = self.encoder_ema

            self.flow_map = self.flow_map_detach
            self.encoder = self.encoder_detach
            self.flow_map_ema = self.flow_map_ema_detach
            self.encoder_ema = self.encoder_ema_detach

            try:
                yield
            finally:
                # Restore original models
                self.flow_map = flow_map_backup
                self.encoder = encoder_backup
                self.flow_map_ema = flow_map_ema_backup
                self.encoder_ema = encoder_ema_backup
        else:
            # For regular mode, temporarily set to eval
            was_training = self.flow_map.training
            try:
                self.flow_map.eval()
                self.encoder.eval()
                self.flow_map_ema.eval()
                self.encoder_ema.eval()
                yield
            finally:
                # Restore training mode if it was training
                if was_training:
                    self.flow_map.train()
                    self.encoder.train()

    def _sync_detached_models(self):
        """Synchronize detached models with main models for CUDA graphs."""
        if self.use_cudagraphs:
            from tensordict import from_module

            # Copy parameters without gradients using tensordict
            from_module(self.flow_map).data.to_module(self.flow_map_detach)
            from_module(self.encoder).data.to_module(self.encoder_detach)
            from_module(self.flow_map_ema).data.to_module(self.flow_map_ema_detach)
            from_module(self.encoder_ema).data.to_module(self.encoder_ema_detach)

    def _create_update_impl(self):
        """Create the update implementation function that will be compiled.

        This function includes the entire update loop: forward, backward, gradient clipping,
        optimizer step, zero_grad, and EMA update.

        Returns:
            A function that performs the complete update step
        """

        def update_impl(data: TensorDict):
            """Complete update step (can be compiled).

            Args:
                data: TensorDict containing 'act', 'obs', 'delta_t' keys

            Returns:
                TensorDict containing loss, grad_norm, and other metrics
            """
            # Extract tensors from TensorDict
            act = data["act"]
            obs = data["obs"]
            delta_t = data["delta_t"]

            # Forward pass and compute loss
            loss, _info = self.loss_fn(
                self.config.optimization,
                self.flow_map,
                self.encoder,
                self.interpolant,
                act,
                obs,
                delta_t,
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            params = list(self.encoder.parameters()) + list(self.flow_map.parameters())
            if self.config.optimization.grad_clip_norm:
                grad_norm = nn.utils.clip_grad_norm_(
                    params, self.config.optimization.grad_clip_norm
                )
            else:
                grad_norm = torch.tensor(0.0, device=loss.device)

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # EMA update
            if self.config.optimization.ema_rate < 1:
                self._ema_update_impl()

            # Return as TensorDict for CUDA graph compatibility (static shapes)
            result = TensorDict(
                {
                    "loss": loss.detach(),
                    "grad_norm": grad_norm.detach(),
                },
                batch_size=(),
            )
            return result

        return update_impl

    def _ema_update_impl(self):
        """EMA update implementation (can be part of compiled function)."""
        params = list(self.encoder.parameters()) + list(self.flow_map.parameters())
        params_ema = list(self.encoder_ema.parameters()) + list(
            self.flow_map_ema.parameters()
        )
        with torch.no_grad():
            for p, p_ema in zip(params, params_ema, strict=False):
                p_ema.data.mul_(self.config.optimization.ema_rate).add_(
                    p.data, alpha=1.0 - self.config.optimization.ema_rate
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
        # Check batch size consistency for CUDA graphs
        if self.use_cudagraphs:
            if not hasattr(self, "_expected_batch_size"):
                self._expected_batch_size = act.shape[0]
            elif act.shape[0] != self._expected_batch_size:
                raise ValueError(
                    f"CUDA graphs require static batch sizes. "
                    f"Expected {self._expected_batch_size}, got {act.shape[0]}. "
                    f"Make sure your dataloader has drop_last=True."
                )

        # Mark CUDA graph step boundary if using compile
        if self.use_compile:
            torch.compiler.cudagraph_mark_step_begin()

        # Run the compiled update (includes forward, backward, optimizer, EMA)
        # Wrap inputs in TensorDict for both CUDA graphs and regular mode
        data = TensorDict(
            {
                "act": act,
                "obs": obs,
                "delta_t": delta_t,
            },
            batch_size=act.shape[0],
        )
        result = self._compiled_update(data)

        # Convert TensorDict to regular dict with scalar values
        return {
            "loss": result["loss"].item(),
            "grad_norm": result["grad_norm"].item(),
        }

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

    def _sample_impl(
        self,
        config,
        flow_map,
        encoder,
        act_0: torch.Tensor,
        obs: torch.Tensor,
    ):
        """Internal sampling implementation (can be compiled).

        Args:
            config: Optimization config
            flow_map: Flow map model
            encoder: Encoder model
            act_0: Initial action tensor
            obs: Observation tensor

        Returns:
            Sampled action tensor
        """
        return self.sampler(config, flow_map, encoder, act_0, obs)

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
        # Sync detached models if using CUDA graphs before inference
        if self.use_cudagraphs:
            self._sync_detached_models()

        # manually set num_steps if needed
        if num_steps >= 1:
            config = deepcopy(self.config.optimization)
            config.num_steps = int(num_steps)
        else:
            config = self.config.optimization

        # choose model
        # Note: For CUDA graphs, the _inference_mode context manager was already
        # applied during __compile__, so we use the original models here
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
            # For CUDA graphs, _compiled_sampler already uses detached models
            # For regular mode, we temporarily switch to eval mode
            if not self.use_cudagraphs:
                with self._inference_mode():
                    act = self._compiled_sampler(config, flow_map, encoder, act_0, obs)
            else:
                act = self._compiled_sampler(config, flow_map, encoder, act_0, obs)
        return act

    def save(self, path: str, training_state: dict = None):
        """Save agent models to path.

        Args:
            path: Path to save checkpoint
            training_state: Optional dict with training state (n_gradient_step, best_metrics, eval_history)
        """
        # save flow map, encoder, encoder_ema, flow_map_ema, optimizer
        checkpoint = {
            "flow_map": self.flow_map.state_dict(),
            "encoder": self.encoder.state_dict(),
            "encoder_ema": self.encoder_ema.state_dict(),
            "flow_map_ema": self.flow_map_ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        # Add training state if provided
        if training_state is not None:
            checkpoint["training_state"] = training_state

        torch.save(checkpoint, path)

    def load(self, path: str, load_optimizer: bool = False):
        """Load agent models from path.

        Args:
            path: Path to load checkpoint from
            load_optimizer: Whether to load optimizer state

        Returns:
            training_state dict if available, None otherwise
        """
        # load flow map, encoder, encoder_ema, flow_map_ema
        state_dict = torch.load(
            path, map_location=self.config.optimization.device, weights_only=False
        )
        self.flow_map.load_state_dict(state_dict["flow_map"])
        self.encoder.load_state_dict(state_dict["encoder"])
        self.encoder_ema.load_state_dict(state_dict["encoder_ema"])
        self.flow_map_ema.load_state_dict(state_dict["flow_map_ema"])

        # Load optimizer state if requested and available
        if load_optimizer and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
            loguru.logger.info("Loaded optimizer state")

        # Recompile after loading to ensure compiled functions are up to date
        self.__compile__()

        # Return training state if available
        training_state = state_dict.get("training_state", None)
        if training_state:
            loguru.logger.info(
                f"Loaded training state from step {training_state.get('n_gradient_step', 'unknown')}"
            )

        return training_state

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
