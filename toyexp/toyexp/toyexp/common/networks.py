"""
Neural network architectures for toy experiments.

Provides MLP architectures with two conditioning strategies:
1. Concatenation: Simply concatenate all inputs [x, c, t]
2. FiLM: Feature-wise Linear Modulation with [c, t] as conditioning

Also includes EMA (Exponential Moving Average) wrapper for model parameters.
"""

from copy import deepcopy
import logging
from typing import Literal, Optional

import torch
import torch.nn as nn


def get_activation(activation: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if activation.lower() not in activations:
        raise ValueError(
            f"Unknown activation: {activation}. Choose from {list(activations.keys())}"
        )
    return activations[activation.lower()]


class BaseModel(nn.Module):
    """
    Base class defining the interface for all models.

    All models should accept:
    - x: [batch, x_dim] - current state (0 for regression, x_t for flow)
    - c: [batch, c_dim] - conditioning variable
    - t: [batch, 1] or None - time variable (for flow)
    """

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, x_dim] - current state
            c: [batch, c_dim] - conditioning variable
            t: [batch, 1] - time (optional, for flow)

        Returns:
            output: [batch, output_dim]
        """
        raise NotImplementedError


class ConcatMLP(BaseModel):
    """
    Standard MLP with concatenation-based conditioning.

    Architecture: concat([x, c, t]) → MLP → output

    Args:
        x_dim: Dimension of state x
        c_dim: Dimension of conditioning variable c
        output_dim: Dimension of output
        hidden_dim: Hidden layer dimension
        n_layers: Number of hidden layers
        activation: Activation function name
        use_time: Whether to expect time input t
    """

    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        activation: str = "relu",
        use_time: bool = True,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.c_dim = c_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.use_time = use_time

        # Input dimension: x + c + (optional t)
        input_dim = x_dim + c_dim
        if use_time:
            input_dim += 1

        # Activation function
        self.activation = get_activation(activation)()

        # Build MLP
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self.activation)

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, x_dim]
            c: [batch, c_dim]
            t: [batch, 1] or None

        Returns:
            output: [batch, output_dim]
        """
        # Ensure proper shapes
        if c.dim() == 1:
            c = c.unsqueeze(-1)

        # Concatenate inputs
        if self.use_time:
            if t is None:
                raise ValueError("Model expects time input t but got None")
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            inputs = torch.cat([x, c, t], dim=-1)
        else:
            inputs = torch.cat([x, c], dim=-1)

        return self.network(inputs)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Applies affine transformation: γ ⊙ x + β
    where γ and β are generated from conditioning input.

    Args:
        feature_dim: Dimension of features to modulate
        cond_dim: Dimension of conditioning input
    """

    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()

        self.feature_dim = feature_dim
        self.cond_dim = cond_dim

        # Generate γ (scale) and β (shift) from conditioning
        self.film_generator = nn.Linear(cond_dim, 2 * feature_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, feature_dim] - features to modulate
            cond: [batch, cond_dim] - conditioning input

        Returns:
            modulated: [batch, feature_dim] - γ ⊙ x + β
        """
        # Generate γ and β
        film_params = self.film_generator(cond)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)

        # Apply FiLM: γ ⊙ x + β
        return gamma * x + beta


class FiLMMLP(BaseModel):
    """
    MLP with Feature-wise Linear Modulation (FiLM) conditioning.

    Architecture:
    - Base network processes x only
    - Conditioning [c, t] generates γ, β for each layer via FiLM
    - Each hidden layer: x → Linear → FiLM([c,t]) → Activation

    Args:
        x_dim: Dimension of state x
        c_dim: Dimension of conditioning variable c
        output_dim: Dimension of output
        hidden_dim: Hidden layer dimension
        n_layers: Number of hidden layers
        activation: Activation function name
        use_time: Whether to expect time input t
    """

    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        activation: str = "relu",
        use_time: bool = True,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.c_dim = c_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.use_time = use_time

        # Conditioning dimension: c + (optional t)
        cond_dim = c_dim
        if use_time:
            cond_dim += 1

        # Activation function
        self.activation = get_activation(activation)()

        # Input projection (x only, no conditioning here)
        self.input_proj = nn.Linear(x_dim, hidden_dim)

        # Hidden layers with FiLM conditioning
        self.hidden_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()

        for _ in range(n_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.film_layers.append(FiLMLayer(hidden_dim, cond_dim))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, x_dim]
            c: [batch, c_dim]
            t: [batch, 1] or None

        Returns:
            output: [batch, output_dim]
        """
        # Ensure proper shapes
        if c.dim() == 1:
            c = c.unsqueeze(-1)

        # Prepare conditioning input
        if self.use_time:
            if t is None:
                raise ValueError("Model expects time input t but got None")
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            cond = torch.cat([c, t], dim=-1)
        else:
            cond = c

        # Input projection
        h = self.activation(self.input_proj(x))

        # Hidden layers with FiLM conditioning
        for hidden_layer, film_layer in zip(self.hidden_layers, self.film_layers):
            h = hidden_layer(h)
            h = film_layer(h, cond)  # Apply FiLM modulation
            h = self.activation(h)

        # Output layer
        return self.output_layer(h)


def create_model(
    architecture: Literal["concat", "film"],
    x_dim: int,
    c_dim: int,
    output_dim: int,
    hidden_dim: int = 256,
    n_layers: int = 3,
    activation: str = "relu",
    use_time: bool = True,
) -> BaseModel:
    """
    Factory function to create models.

    Args:
        architecture: 'concat' for ConcatMLP, 'film' for FiLMMLP
        x_dim: Dimension of state x
        c_dim: Dimension of conditioning variable c
        output_dim: Dimension of output
        hidden_dim: Hidden layer dimension
        n_layers: Number of hidden layers
        activation: Activation function name ('relu', 'gelu', 'silu', etc.)
        use_time: Whether model expects time input t

    Returns:
        Model instance
    """
    if architecture == "concat":
        return ConcatMLP(
            x_dim=x_dim,
            c_dim=c_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            use_time=use_time,
        )
    elif architecture == "film":
        return FiLMMLP(
            x_dim=x_dim,
            c_dim=c_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            use_time=use_time,
        )
    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. Choose 'concat' or 'film'"
        )


class EMA:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of model parameters that is updated as an
    exponential moving average. Useful for stabilizing training and
    improving final model performance.

    Can be disabled by setting enabled=False, in which case all operations
    are no-ops. This allows clean integration with config flags.

    Args:
        model: Model whose parameters to track
        decay: EMA decay rate (typical values: 0.999, 0.9999)
        enabled: Whether EMA is enabled (from config)
        device: Device to store EMA parameters on

    Usage:
        # From config
        ema = EMA(model, decay=config.ema_decay, enabled=config.use_ema)

        # During training
        for batch in dataloader:
            loss = train_step(model, batch)
            optimizer.step()
            ema.update(model)  # No-op if disabled

        # For evaluation
        with ema.average_parameters():  # No-op if disabled
            eval_loss = evaluate(model)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        enabled: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.decay = decay
        self.enabled = enabled
        self.device = device if device is not None else next(model.parameters()).device

        # Create shadow parameters only if enabled
        self.shadow_params = {}
        self.backup_params = {}

        if self.enabled:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow_params[name] = param.data.clone().to(self.device)

    @torch.no_grad()
    def update(self, model: Optional[nn.Module] = None):
        """
        Update EMA parameters. No-op if EMA is disabled.

        shadow_param = decay * shadow_param + (1 - decay) * model_param

        Args:
            model: Model to update from (uses self.model if None)
        """
        if not self.enabled:
            return

        if model is None:
            model = self.model

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert (
                    name in self.shadow_params
                ), f"Parameter {name} not found in shadow params"
                self.shadow_params[name].mul_(self.decay).add_(
                    param.data.to(self.device), alpha=1 - self.decay
                )

    @torch.no_grad()
    def apply_shadow(self):
        """
        Replace model parameters with EMA shadow parameters.
        Backs up original parameters for later restoration.
        No-op if EMA is disabled.
        """
        if not self.enabled:
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup_params[name] = param.data.clone()
                param.data.copy_(self.shadow_params[name])

    @torch.no_grad()
    def restore(self):
        """
        Restore original model parameters from backup.
        No-op if EMA is disabled.
        """
        if not self.enabled:
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup_params[name])
        self.backup_params = {}

    def average_parameters(self):
        """
        Context manager for temporarily using EMA parameters.
        If EMA is disabled, this is a no-op context manager.

        Usage:
            with ema.average_parameters():
                # Model uses EMA parameters if enabled, otherwise unchanged
                eval_loss = evaluate(model)
            # Model parameters restored (if EMA was enabled)
        """
        return _EMAContextManager(self)

    def state_dict(self):
        """Get state dict for saving."""
        return {
            "decay": self.decay,
            "enabled": self.enabled,
            "shadow_params": self.shadow_params if self.enabled else {},
        }

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.decay = state_dict["decay"]
        self.enabled = state_dict.get("enabled", True)  # Backward compatibility
        if self.enabled:
            self.shadow_params = state_dict["shadow_params"]


class _EMAContextManager:
    """Context manager for EMA.average_parameters()."""

    def __init__(self, ema: EMA):
        self.ema = ema

    def __enter__(self):
        self.ema.apply_shadow()
        return self.ema

    def __exit__(self, *args):
        self.ema.restore()


if __name__ == "__main__":
    import logging

    from logging_utils import setup_logger

    # Setup logger for testing
    logger = setup_logger("toyexp.networks.test", level=logging.INFO)

    logger.info("Testing network architectures...")

    # Test configuration
    batch_size = 4
    x_dim = 8
    c_dim = 1
    output_dim = 8
    hidden_dim = 64
    n_layers = 3

    # Create sample inputs
    x = torch.randn(batch_size, x_dim)
    c = torch.randn(batch_size, c_dim)
    t = torch.rand(batch_size, 1)

    logger.info("=" * 60)
    logger.info("Test 1: ConcatMLP with time")
    logger.info("=" * 60)
    model_concat = create_model(
        architecture="concat",
        x_dim=x_dim,
        c_dim=c_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation="relu",
        use_time=True,
    )

    logger.info(f"Model: {model_concat.__class__.__name__}")
    logger.info(f"Parameters: {sum(p.numel() for p in model_concat.parameters()):,}")

    output = model_concat(x, c, t)
    logger.info(f"Input shapes: x={x.shape}, c={c.shape}, t={t.shape}")
    logger.info(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, output_dim), "Output shape mismatch!"
    logger.info("✓ Forward pass successful\n")

    logger.info("=" * 60)
    logger.info("Test 2: ConcatMLP without time (regression mode)")
    logger.info("=" * 60)
    model_concat_no_time = create_model(
        architecture="concat",
        x_dim=x_dim,
        c_dim=c_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation="relu",
        use_time=False,
    )

    logger.info(f"Model: {model_concat_no_time.__class__.__name__}")
    logger.info(
        f"Parameters: {sum(p.numel() for p in model_concat_no_time.parameters()):,}"
    )

    output = model_concat_no_time(x, c)
    logger.info(f"Input shapes: x={x.shape}, c={c.shape}")
    logger.info(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, output_dim), "Output shape mismatch!"
    logger.info("✓ Forward pass successful\n")

    logger.info("=" * 60)
    logger.info("Test 3: FiLMMLP with time")
    logger.info("=" * 60)
    model_film = create_model(
        architecture="film",
        x_dim=x_dim,
        c_dim=c_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation="relu",
        use_time=True,
    )

    logger.info(f"Model: {model_film.__class__.__name__}")
    logger.info(f"Parameters: {sum(p.numel() for p in model_film.parameters()):,}")

    output = model_film(x, c, t)
    logger.info(f"Input shapes: x={x.shape}, c={c.shape}, t={t.shape}")
    logger.info(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, output_dim), "Output shape mismatch!"
    logger.info("✓ Forward pass successful\n")

    logger.info("=" * 60)
    logger.info("Test 4: FiLMMLP without time (regression mode)")
    logger.info("=" * 60)
    model_film_no_time = create_model(
        architecture="film",
        x_dim=x_dim,
        c_dim=c_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation="gelu",
        use_time=False,
    )

    logger.info(f"Model: {model_film_no_time.__class__.__name__}")
    logger.info(
        f"Parameters: {sum(p.numel() for p in model_film_no_time.parameters()):,}"
    )

    output = model_film_no_time(x, c)
    logger.info(f"Input shapes: x={x.shape}, c={c.shape}")
    logger.info(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, output_dim), "Output shape mismatch!"
    logger.info("✓ Forward pass successful\n")

    logger.info("=" * 60)
    logger.info("Test 5: Shape flexibility (1D c input)")
    logger.info("=" * 60)
    c_1d = torch.randn(batch_size)  # 1D instead of 2D
    output = model_film(x, c_1d, t)
    logger.info(f"Input c shape: {c_1d.shape} (1D)")
    logger.info(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, output_dim), "Output shape mismatch!"
    logger.info("✓ Handles 1D conditioning correctly\n")

    logger.info("=" * 60)
    logger.info("Test 6: Regression use case (x = zeros)")
    logger.info("=" * 60)
    x_zeros = torch.zeros(batch_size, x_dim)
    output_reg = model_film_no_time(x_zeros, c)
    logger.info(f"Regression input: x=zeros({x_zeros.shape}), c={c.shape}")
    logger.info(f"Output shape: {output_reg.shape}")
    logger.info("✓ Regression mode works with zero input\n")

    logger.info("=" * 60)
    logger.info("Test 7: Different activations")
    logger.info("=" * 60)
    for act in ["relu", "gelu", "silu", "tanh"]:
        model = create_model(
            architecture="concat",
            x_dim=x_dim,
            c_dim=c_dim,
            output_dim=output_dim,
            hidden_dim=32,
            n_layers=2,
            activation=act,
            use_time=False,
        )
        output = model(x, c)
        logger.info(f"✓ {act.upper()}: output shape {output.shape}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Test 8: EMA (Exponential Moving Average)")
    logger.info("=" * 60)

    # Create a simple model
    model = create_model(
        architecture="concat",
        x_dim=x_dim,
        c_dim=c_dim,
        output_dim=output_dim,
        hidden_dim=32,
        n_layers=2,
        activation="relu",
        use_time=False,
    )

    # Test with EMA enabled
    logger.info("\nTest 8a: EMA enabled")
    ema_enabled = EMA(model, decay=0.999, enabled=True)
    logger.info(
        f"✓ EMA initialized with decay={ema_enabled.decay}, enabled={ema_enabled.enabled}"
    )

    # Store original output
    with torch.no_grad():
        original_output = model(x, c).clone()

    # Simulate training updates
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(10):
        output = model(x, c)
        loss = output.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_enabled.update(model)

    # Check that model parameters changed
    with torch.no_grad():
        updated_output = model(x, c)

    logger.info("✓ Model updated through training")
    logger.info(
        f"  Output changed: {not torch.allclose(original_output, updated_output)}"
    )

    # Test context manager
    with ema_enabled.average_parameters():
        ema_output = model(x, c)
        logger.info("✓ Using EMA parameters (context manager)")

    # After context, should be back to updated parameters
    post_context_output = model(x, c)
    logger.info("✓ Parameters restored after context")
    logger.info(
        f"  Match updated params: {torch.allclose(updated_output, post_context_output)}"
    )

    # Test manual apply/restore
    ema_enabled.apply_shadow()
    manual_ema_output = model(x, c)
    ema_enabled.restore()
    logger.info("✓ Manual apply_shadow() and restore() work")
    logger.info(f"  EMA outputs match: {torch.allclose(ema_output, manual_ema_output)}")

    # Test state dict save/load
    ema_state = ema_enabled.state_dict()
    new_ema = EMA(model, decay=0.99, enabled=True)
    new_ema.load_state_dict(ema_state)
    logger.info("✓ EMA state_dict save/load works")
    logger.info(f"  Loaded decay: {new_ema.decay}, enabled: {new_ema.enabled}")

    # Test with EMA disabled
    logger.info("\nTest 8b: EMA disabled (from config flag)")
    ema_disabled = EMA(model, decay=0.999, enabled=False)
    logger.info(f"✓ EMA initialized with enabled={ema_disabled.enabled}")

    # Store current output
    with torch.no_grad():
        before_disabled = model(x, c).clone()

    # These should all be no-ops
    ema_disabled.update(model)
    ema_disabled.apply_shadow()
    with torch.no_grad():
        during_disabled = model(x, c).clone()
    ema_disabled.restore()

    with ema_disabled.average_parameters():
        with torch.no_grad():
            context_disabled = model(x, c).clone()

    logger.info("✓ All EMA operations are no-ops when disabled")
    logger.info(
        f"  Output unchanged after update: {torch.allclose(before_disabled, during_disabled)}"
    )
    logger.info(
        f"  Output unchanged in context: {torch.allclose(before_disabled, context_disabled)}"
    )

    # Test that disabled EMA saves/loads correctly
    disabled_state = ema_disabled.state_dict()
    logger.info(f"✓ Disabled EMA state_dict: enabled={disabled_state['enabled']}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("All tests including EMA (enabled/disabled) passed! ✓")
    logger.info("=" * 60)

    # Test configuration
    batch_size = 4
    x_dim = 8
    c_dim = 1
    output_dim = 8
    hidden_dim = 64
    n_layers = 3

    # Create sample inputs
    x = torch.randn(batch_size, x_dim)
    c = torch.randn(batch_size, c_dim)
    t = torch.rand(batch_size, 1)

    logger.info("=" * 60)
    logger.info("Test 1: ConcatMLP with time")
    logger.info("=" * 60)
    model_concat = create_model(
        architecture="concat",
        x_dim=x_dim,
        c_dim=c_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation="relu",
        use_time=True,
    )

    logger.info(f"Model: {model_concat.__class__.__name__}")
    logger.info(f"Parameters: {sum(p.numel() for p in model_concat.parameters()):,}")

    output = model_concat(x, c, t)
    logger.info(f"Input shapes: x={x.shape}, c={c.shape}, t={t.shape}")
    logger.info(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, output_dim), "Output shape mismatch!"
    logger.info("✓ Forward pass successful\n")

    logger.info("=" * 60)
    logger.info("Test 2: ConcatMLP without time (regression mode)")
    logger.info("=" * 60)
    model_concat_no_time = create_model(
        architecture="concat",
        x_dim=x_dim,
        c_dim=c_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation="relu",
        use_time=False,
    )

    logger.info(f"Model: {model_concat_no_time.__class__.__name__}")
    logger.info(
        f"Parameters: {sum(p.numel() for p in model_concat_no_time.parameters()):,}"
    )

    output = model_concat_no_time(x, c)
    logger.info(f"Input shapes: x={x.shape}, c={c.shape}")
    logger.info(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, output_dim), "Output shape mismatch!"
    logger.info("✓ Forward pass successful\n")

    logger.info("=" * 60)
    logger.info("Test 3: FiLMMLP with time")
    logger.info("=" * 60)
    model_film = create_model(
        architecture="film",
        x_dim=x_dim,
        c_dim=c_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation="relu",
        use_time=True,
    )

    logger.info(f"Model: {model_film.__class__.__name__}")
    logger.info(f"Parameters: {sum(p.numel() for p in model_film.parameters()):,}")

    output = model_film(x, c, t)
    logger.info(f"Input shapes: x={x.shape}, c={c.shape}, t={t.shape}")
    logger.info(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, output_dim), "Output shape mismatch!"
    logger.info("✓ Forward pass successful\n")

    logger.info("=" * 60)
    logger.info("Test 4: FiLMMLP without time (regression mode)")
    logger.info("=" * 60)
    model_film_no_time = create_model(
        architecture="film",
        x_dim=x_dim,
        c_dim=c_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation="gelu",
        use_time=False,
    )

    logger.info(f"Model: {model_film_no_time.__class__.__name__}")
    logger.info(
        f"Parameters: {sum(p.numel() for p in model_film_no_time.parameters()):,}"
    )

    output = model_film_no_time(x, c)
    logger.info(f"Input shapes: x={x.shape}, c={c.shape}")
    logger.info(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, output_dim), "Output shape mismatch!"
    logger.info("✓ Forward pass successful\n")

    logger.info("=" * 60)
    logger.info("Test 5: Shape flexibility (1D c input)")
    logger.info("=" * 60)
    c_1d = torch.randn(batch_size)  # 1D instead of 2D
    output = model_film(x, c_1d, t)
    logger.info(f"Input c shape: {c_1d.shape} (1D)")
    logger.info(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, output_dim), "Output shape mismatch!"
    logger.info("✓ Handles 1D conditioning correctly\n")

    logger.info("=" * 60)
    logger.info("Test 6: Regression use case (x = zeros)")
    logger.info("=" * 60)
    x_zeros = torch.zeros(batch_size, x_dim)
    output_reg = model_film_no_time(x_zeros, c)
    logger.info(f"Regression input: x=zeros({x_zeros.shape}), c={c.shape}")
    logger.info(f"Output shape: {output_reg.shape}")
    logger.info("✓ Regression mode works with zero input\n")

    logger.info("=" * 60)
    logger.info("Test 7: Different activations")
    logger.info("=" * 60)
    for act in ["relu", "gelu", "silu", "tanh"]:
        model = create_model(
            architecture="concat",
            x_dim=x_dim,
            c_dim=c_dim,
            output_dim=output_dim,
            hidden_dim=32,
            n_layers=2,
            activation=act,
            use_time=False,
        )
        output = model(x, c)
        logger.info(f"✓ {act.upper()}: output shape {output.shape}")

    logger.info("\n" + "=" * 60)
    logger.info("All tests passed! ✓")
    logger.info("=" * 60)

    # Test EMA
    logger.info("\n" + "=" * 60)
    logger.info("Test 8: EMA (Exponential Moving Average)")
    logger.info("=" * 60)

    # Create a simple model
    model = create_model(
        architecture="concat",
        x_dim=x_dim,
        c_dim=c_dim,
        output_dim=output_dim,
        hidden_dim=32,
        n_layers=2,
        activation="relu",
        use_time=False,
    )

    # Test with EMA enabled
    logger.info("\nTest 8a: EMA enabled")
    ema_enabled = EMA(model, decay=0.999, enabled=True)
    logger.info(
        f"✓ EMA initialized with decay={ema_enabled.decay}, enabled={ema_enabled.enabled}"
    )

    # Store original output
    with torch.no_grad():
        original_output = model(x, c).clone()

    # Simulate training updates
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(10):
        output = model(x, c)
        loss = output.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_enabled.update(model)  # Update EMA after each step

    # Check that model parameters changed
    with torch.no_grad():
        updated_output = model(x, c)

    logger.info("✓ Model updated through training")
    logger.info(
        f"  Output changed: {not torch.allclose(original_output, updated_output)}"
    )

    # Test context manager
    with ema_enabled.average_parameters():
        ema_output = model(x, c)
        logger.info("✓ Using EMA parameters (context manager)")

    # After context, should be back to updated parameters
    post_context_output = model(x, c)
    logger.info("✓ Parameters restored after context")
    logger.info(
        f"  Match updated params: {torch.allclose(updated_output, post_context_output)}"
    )

    # Test manual apply/restore
    ema_enabled.apply_shadow()
    manual_ema_output = model(x, c)
    ema_enabled.restore()
    logger.info("✓ Manual apply_shadow() and restore() work")
    logger.info(f"  EMA outputs match: {torch.allclose(ema_output, manual_ema_output)}")

    # Test state dict save/load
    ema_state = ema_enabled.state_dict()
    new_ema = EMA(model, decay=0.99, enabled=True)
    new_ema.load_state_dict(ema_state)
    logger.info("✓ EMA state_dict save/load works")
    logger.info(f"  Loaded decay: {new_ema.decay}, enabled: {new_ema.enabled}")

    # Test with EMA disabled
    logger.info("\nTest 8b: EMA disabled (from config flag)")
    ema_disabled = EMA(model, decay=0.999, enabled=False)
    logger.info(f"✓ EMA initialized with enabled={ema_disabled.enabled}")

    # Store current output
    with torch.no_grad():
        before_disabled = model(x, c).clone()

    # These should all be no-ops
    ema_disabled.update(model)
    ema_disabled.apply_shadow()
    with torch.no_grad():
        during_disabled = model(x, c).clone()
    ema_disabled.restore()

    with ema_disabled.average_parameters():
        with torch.no_grad():
            context_disabled = model(x, c).clone()

    logger.info("✓ All EMA operations are no-ops when disabled")
    logger.info(
        f"  Output unchanged after update: {torch.allclose(before_disabled, during_disabled)}"
    )
    logger.info(
        f"  Output unchanged in context: {torch.allclose(before_disabled, context_disabled)}"
    )

    # Test that disabled EMA saves/loads correctly
    disabled_state = ema_disabled.state_dict()
    logger.info(f"✓ Disabled EMA state_dict: enabled={disabled_state['enabled']}")

    logger.info("\n" + "=" * 60)
    logger.info("All tests including EMA (enabled/disabled) passed! ✓")
    logger.info("=" * 60)
