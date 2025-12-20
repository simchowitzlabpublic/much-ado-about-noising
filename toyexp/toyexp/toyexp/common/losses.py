"""
Loss functions for toy experiments.

Provides simple loss functions for:
- Regression: L1 and L2 reconstruction losses
- Flow: Flow matching loss for learning velocity fields
- Straight Flow: Predict x_1 from interpolated states without time conditioning
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L1 (Mean Absolute Error) loss.

    Args:
        pred: [batch, ...] predictions
        target: [batch, ...] targets

    Returns:
        scalar loss
    """
    return torch.abs(pred - target).mean()


def l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L2 (Mean Squared Error) loss.

    Args:
        pred: [batch, ...] predictions
        target: [batch, ...] targets

    Returns:
        scalar loss
    """
    return ((pred - target) ** 2).mean()


def flow_matching_loss(
    model: nn.Module,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    c: torch.Tensor,
    t: torch.Tensor,
    loss_type: str = "l2",
) -> torch.Tensor:
    """
    Flow matching loss for training velocity field models.

    The model learns to predict the velocity field v(x_t, c, t) such that:
    dx/dt = v(x_t, c, t)

    For linear interpolation: x_t = (1-t)*x_0 + t*x_1
    The target velocity is: v_true = x_1 - x_0

    Loss: ||v_pred - v_true||^2 or ||v_pred - v_true||

    Args:
        model: Model that predicts velocity v(x_t, c, t)
        x_0: [batch, dim] initial state (typically noise or zeros)
        x_1: [batch, dim] target state (target function values)
        c: [batch, c_dim] conditioning variable
        t: [batch, 1] time values in [0, 1]
        loss_type: 'l1' or 'l2'

    Returns:
        scalar loss
    """
    # Compute interpolated state
    x_t = (1 - t) * x_0 + t * x_1

    # True velocity (constant for linear interpolation)
    v_true = x_1 - x_0

    # Predict velocity
    v_pred = model(x_t, c, t)

    # Compute loss
    if loss_type == "l1":
        return l1_loss(v_pred, v_true)
    elif loss_type == "l2":
        return l2_loss(v_pred, v_true)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'l1' or 'l2'")


def straight_flow_loss(
    model: nn.Module,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    c: torch.Tensor,
    t: torch.Tensor,
    loss_type: str = "l2",
) -> torch.Tensor:
    """
    Straight flow loss - predict x_1 from interpolated x_t without time conditioning.

    Similar to flow matching but:
    1. Model predicts x_1 directly (not velocity)
    2. Time is always set to 0 when querying the model

    This is an ablation to test whether time conditioning is necessary.
    The model must learn a single function that works for inputs from
    any point along the interpolation path [0, 1].

    Args:
        model: Model that predicts x_1 from (x_t, c, t)
        x_0: [batch, dim] initial state (typically noise or zeros)
        x_1: [batch, dim] target state
        c: [batch, c_dim] conditioning variable
        t: [batch, 1] time values in [0, 1] (used for interpolation, not passed to model)
        loss_type: 'l1' or 'l2'

    Returns:
        scalar loss
    """
    batch_size = x_0.shape[0]
    device = x_0.device

    # Compute interpolated state (using actual t for interpolation)
    x_t = (1 - t) * x_0 + t * x_1

    # Always query model with t=0 (no time conditioning)
    t_zero = torch.zeros(batch_size, 1, device=device)

    # Predict x_1 directly
    x_1_pred = model(x_t, c, t_zero)

    # Compute loss
    if loss_type == "l1":
        return l1_loss(x_1_pred, x_1)
    elif loss_type == "l2":
        return l2_loss(x_1_pred, x_1)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'l1' or 'l2'")


def regression_loss(
    model: nn.Module,
    c: torch.Tensor,
    target: torch.Tensor,
    x_dim: int,
    loss_type: str = "l2",
) -> torch.Tensor:
    """
    Regression loss for direct function approximation.

    Model predicts f(c) directly. To match input dimensions with flow models,
    we pass zeros as the x input.

    Args:
        model: Model that predicts f(c) (expects x, c inputs)
        c: [batch, c_dim] conditioning variable
        target: [batch, output_dim] target function values
        x_dim: Dimension of x (to create zero placeholder)
        loss_type: 'l1' or 'l2'

    Returns:
        scalar loss
    """
    batch_size = c.shape[0]
    device = c.device

    # Create zero placeholder to match flow model input dimensions
    x_zeros = torch.zeros(batch_size, x_dim, device=device)

    # Predict (model expects x, c; no time for regression)
    pred = model(x_zeros, c, t=None)

    # Compute loss
    if loss_type == "l1":
        return l1_loss(pred, target)
    elif loss_type == "l2":
        return l2_loss(pred, target)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'l1' or 'l2'")


def mip_loss(
    model: nn.Module,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    c: torch.Tensor,
    t_star: float = 0.9,
    loss_type: str = "l2",
) -> torch.Tensor:
    """
    Marginal likelihood-Informed Prediction (MIP) loss.

    Combines two terms:
    1. Regression from zeros at t=0: model predicts x_1 directly from x_0=0
    2. Denoising at t=t*: model predicts x_1 from interpolated state x_t*

    Following the paper formulation:
    L_MIP = E[||f(0, c, 0) - x_1|| + ||f(x_t*, c, t*) - x_1||]

    where x_t* = (1-t*)x_0 + t*x_1

    Args:
        model: Model that predicts targets from (x, c, t)
        x_0: [batch, dim] initial state (typically zeros)
        x_1: [batch, dim] target state
        c: [batch, c_dim] conditioning variable
        t_star: Fixed time point for denoising term (default 0.9)
        loss_type: 'l1' or 'l2'

    Returns:
        scalar loss (sum of regression and denoising terms)
    """
    batch_size = c.shape[0]
    device = c.device

    # Term 1: Regression from zeros at t=0
    t_zero = torch.zeros(batch_size, 1, device=device)
    pred_from_zero = model(x_0, c, t_zero)

    # Term 2: Denoising at t=t*
    t_star_tensor = torch.full((batch_size, 1), t_star, device=device)
    x_t_star = (1 - t_star) * x_0 + t_star * x_1
    pred_from_noisy = model(x_t_star, c, t_star_tensor)

    # Compute losses based on loss_type
    if loss_type == "l1":
        regression_loss_val = l1_loss(pred_from_zero, x_1)
        denoising_loss_val = l1_loss(pred_from_noisy, x_1)
    elif loss_type == "l2":
        regression_loss_val = l2_loss(pred_from_zero, x_1)
        denoising_loss_val = l2_loss(pred_from_noisy, x_1)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'l1' or 'l2'")

    # Total MIP loss (sum of both terms)
    return regression_loss_val + denoising_loss_val


class LossManager:
    """
    Manager for computing losses with automatic mode detection.

    Simplifies loss computation by automatically selecting the appropriate
    loss function based on training mode (regression vs flow vs straight_flow vs mip).

    Supports per-component loss aggregation for Lie algebra datasets.

    Args:
        mode: 'regression', 'flow', 'straight_flow', 'mip', or 'mip_one_step_integrate'
        loss_type: 'l1' or 'l2' (applies to all modes)
        x_dim: Dimension of x (for creating zero placeholders in regression)
        loss_aggregation: 'full' or 'per_component' (for Lie datasets)
        num_components: Number of components K (required if loss_aggregation='per_component')
        component_dim: Dimension of each component (required if loss_aggregation='per_component')
        mip_t_star: Fixed time point for MIP denoising term (default 0.9)

    Usage:
        # Standard usage (full aggregation)
        loss_manager = LossManager(mode='flow', loss_type='l2', x_dim=8)

        # Straight flow mode (ablation without time conditioning)
        loss_manager = LossManager(mode='straight_flow', loss_type='l2', x_dim=8)

        # MIP mode with custom t*
        loss_manager = LossManager(mode='mip', loss_type='l1', x_dim=8, mip_t_star=0.9)

        # Per-component aggregation for Lie datasets
        loss_manager = LossManager(
            mode='flow',
            loss_type='l2',
            x_dim=6,  # K=3, dim=2, so output is 3*2=6
            loss_aggregation='per_component',
            num_components=3,
            component_dim=2
        )

        # In training loop
        if mode == 'regression':
            loss = loss_manager.compute_loss(model, c, target)
        elif mode in ['flow', 'straight_flow']:
            loss = loss_manager.compute_loss(model, x_0, x_1, c, t)
        elif mode in ['mip', 'mip_one_step_integrate']:
            loss = loss_manager.compute_loss(model, x_0, x_1, c)
    """

    def __init__(
        self,
        mode: str = "regression",
        loss_type: str = "l2",
        x_dim: Optional[int] = None,
        loss_aggregation: str = "full",
        num_components: Optional[int] = None,
        component_dim: Optional[int] = None,
        mip_t_star: float = 0.9,
    ):
        valid_modes = [
            "regression",
            "flow",
            "straight_flow",
            "mip",
            "mip_one_step_integrate",
        ]
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode: {mode}. Choose from {valid_modes}")

        if loss_type not in ["l1", "l2"]:
            raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'l1' or 'l2'")

        if mode == "regression" and x_dim is None:
            raise ValueError("x_dim must be specified for regression mode")

        if loss_aggregation not in ["full", "per_component"]:
            raise ValueError(
                f"Unknown loss_aggregation: {loss_aggregation}. Choose 'full' or 'per_component'"
            )

        if loss_aggregation == "per_component":
            if num_components is None or component_dim is None:
                raise ValueError(
                    "num_components and component_dim must be specified for per_component aggregation"
                )
            if x_dim != num_components * component_dim:
                raise ValueError(
                    f"x_dim ({x_dim}) must equal num_components * component_dim "
                    f"({num_components} * {component_dim} = {num_components * component_dim})"
                )

        self.mode = mode
        self.loss_type = loss_type
        self.x_dim = x_dim
        self.loss_aggregation = loss_aggregation
        self.num_components = num_components
        self.component_dim = component_dim
        self.mip_t_star = mip_t_star

        logger.info(
            f"LossManager initialized: mode={mode}, loss_type={loss_type}, x_dim={x_dim}, "
            f"loss_aggregation={loss_aggregation}"
        )

        if mode == "mip":
            logger.info(f"  MIP t*={mip_t_star}")

        if loss_aggregation == "per_component":
            logger.info(f"  Per-component: K={num_components}, dim={component_dim}")

    def compute_loss(
        self,
        model: nn.Module,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute loss based on mode and aggregation strategy.

        For regression mode:
            compute_loss(model, c, target)

        For flow/straight_flow mode:
            compute_loss(model, x_0, x_1, c, t)

        For mip mode:
            compute_loss(model, x_0, x_1, c)
        """
        if self.mode == "regression":
            if len(args) != 2:
                raise ValueError(
                    f"Regression mode expects 2 args (c, target), got {len(args)}"
                )
            c, target = args

            if self.loss_aggregation == "full":
                return regression_loss(model, c, target, self.x_dim, self.loss_type)
            else:  # per_component
                return self._per_component_regression_loss(model, c, target)

        elif self.mode == "flow":
            if len(args) != 4:
                raise ValueError(
                    f"Flow mode expects 4 args (x_0, x_1, c, t), got {len(args)}"
                )
            x_0, x_1, c, t = args

            if self.loss_aggregation == "full":
                return flow_matching_loss(model, x_0, x_1, c, t, self.loss_type)
            else:  # per_component
                return self._per_component_flow_loss(model, x_0, x_1, c, t)

        elif self.mode == "straight_flow":
            if len(args) != 4:
                raise ValueError(
                    f"Straight flow mode expects 4 args (x_0, x_1, c, t), got {len(args)}"
                )
            x_0, x_1, c, t = args

            if self.loss_aggregation == "full":
                return straight_flow_loss(model, x_0, x_1, c, t, self.loss_type)
            else:  # per_component
                return self._per_component_straight_flow_loss(model, x_0, x_1, c, t)

        elif self.mode in ["mip", "mip_one_step_integrate"]:
            if len(args) != 3:
                raise ValueError(
                    f"MIP mode expects 3 args (x_0, x_1, c), got {len(args)}"
                )
            x_0, x_1, c = args

            if self.loss_aggregation == "full":
                return mip_loss(model, x_0, x_1, c, self.mip_t_star, self.loss_type)
            else:  # per_component
                return self._per_component_mip_loss(model, x_0, x_1, c)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _per_component_regression_loss(
        self,
        model: nn.Module,
        c: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-component regression loss and aggregate.

        Target shape: [batch, K * dim]
        Split into K components of dim each, compute loss separately, then sum.
        """
        batch_size = c.shape[0]
        device = c.device

        # Create zero placeholder
        x_zeros = torch.zeros(batch_size, self.x_dim, device=device)

        # Predict
        pred = model(x_zeros, c, t=None)  # [batch, K * dim]

        # Reshape to [batch, K, dim]
        pred = pred.reshape(batch_size, self.num_components, self.component_dim)
        target = target.reshape(batch_size, self.num_components, self.component_dim)

        # Compute loss for each component
        component_losses = []
        for k in range(self.num_components):
            pred_k = pred[:, k, :]  # [batch, dim]
            target_k = target[:, k, :]  # [batch, dim]

            if self.loss_type == "l1":
                loss_k = l1_loss(pred_k, target_k)
            elif self.loss_type == "l2":
                loss_k = l2_loss(pred_k, target_k)
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            component_losses.append(loss_k)

        # Sum losses across components
        return sum(component_losses)

    def _per_component_flow_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-component flow matching loss and aggregate.

        x_0, x_1 shape: [batch, K * dim]
        Split into K components of dim each, compute loss separately, then sum.
        """
        batch_size = x_0.shape[0]

        # Interpolate
        x_t = (1 - t) * x_0 + t * x_1

        # True velocity
        v_true = x_1 - x_0

        # Predict velocity
        v_pred = model(x_t, c, t)  # [batch, K * dim]

        # Reshape to [batch, K, dim]
        v_pred = v_pred.reshape(batch_size, self.num_components, self.component_dim)
        v_true = v_true.reshape(batch_size, self.num_components, self.component_dim)

        # Compute loss for each component
        component_losses = []
        for k in range(self.num_components):
            v_pred_k = v_pred[:, k, :]  # [batch, dim]
            v_true_k = v_true[:, k, :]  # [batch, dim]

            if self.loss_type == "l1":
                loss_k = l1_loss(v_pred_k, v_true_k)
            elif self.loss_type == "l2":
                loss_k = l2_loss(v_pred_k, v_true_k)
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            component_losses.append(loss_k)

        # Sum losses across components
        return sum(component_losses)

    def _per_component_straight_flow_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-component straight flow loss and aggregate.

        x_0, x_1 shape: [batch, K * dim]
        Split into K components of dim each, compute loss separately, then sum.
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Interpolate
        x_t = (1 - t) * x_0 + t * x_1

        # Always query with t=0
        t_zero = torch.zeros(batch_size, 1, device=device)

        # Predict x_1 directly
        x_1_pred = model(x_t, c, t_zero)  # [batch, K * dim]

        # Reshape to [batch, K, dim]
        x_1_pred = x_1_pred.reshape(batch_size, self.num_components, self.component_dim)
        x_1_reshaped = x_1.reshape(batch_size, self.num_components, self.component_dim)

        # Compute loss for each component
        component_losses = []
        for k in range(self.num_components):
            pred_k = x_1_pred[:, k, :]  # [batch, dim]
            target_k = x_1_reshaped[:, k, :]  # [batch, dim]

            if self.loss_type == "l1":
                loss_k = l1_loss(pred_k, target_k)
            elif self.loss_type == "l2":
                loss_k = l2_loss(pred_k, target_k)
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            component_losses.append(loss_k)

        # Sum losses across components
        return sum(component_losses)

    def _per_component_mip_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-component MIP loss and aggregate.

        x_0, x_1 shape: [batch, K * dim]
        Split into K components of dim each, compute MIP loss separately, then sum.
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Prepare time tensors
        t_zero = torch.zeros(batch_size, 1, device=device)
        t_star_tensor = torch.full((batch_size, 1), self.mip_t_star, device=device)

        # Interpolate for denoising term
        x_t_star = (1 - self.mip_t_star) * x_0 + self.mip_t_star * x_1

        # Predict from zeros (t=0)
        pred_from_zero = model(x_0, c, t_zero)  # [batch, K * dim]

        # Predict from noisy (t=t*)
        pred_from_noisy = model(x_t_star, c, t_star_tensor)  # [batch, K * dim]

        # Reshape all to [batch, K, dim]
        pred_from_zero = pred_from_zero.reshape(
            batch_size, self.num_components, self.component_dim
        )
        pred_from_noisy = pred_from_noisy.reshape(
            batch_size, self.num_components, self.component_dim
        )
        x_1_reshaped = x_1.reshape(batch_size, self.num_components, self.component_dim)

        # Compute loss for each component
        component_losses = []
        for k in range(self.num_components):
            pred_zero_k = pred_from_zero[:, k, :]  # [batch, dim]
            pred_noisy_k = pred_from_noisy[:, k, :]  # [batch, dim]
            target_k = x_1_reshaped[:, k, :]  # [batch, dim]

            # MIP loss = regression term + denoising term
            if self.loss_type == "l1":
                regression_loss_k = l1_loss(pred_zero_k, target_k)
                denoising_loss_k = l1_loss(pred_noisy_k, target_k)
            elif self.loss_type == "l2":
                regression_loss_k = l2_loss(pred_zero_k, target_k)
                denoising_loss_k = l2_loss(pred_noisy_k, target_k)
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            loss_k = regression_loss_k + denoising_loss_k
            component_losses.append(loss_k)

        # Sum losses across components
        return sum(component_losses)


if __name__ == "__main__":
    from networks import create_model

    # Setup logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Testing loss functions...")

    batch_size = 4
    dim = 8
    c_dim = 1

    # Create sample data
    pred = torch.randn(batch_size, dim)
    target = torch.randn(batch_size, dim)
    x_0 = torch.randn(batch_size, dim)
    x_1 = torch.randn(batch_size, dim)
    c = torch.randn(batch_size, c_dim)
    t = torch.rand(batch_size, 1)

    # Test basic losses
    logger.info("\n=== Test 1: Basic loss functions ===")
    loss_l1 = l1_loss(pred, target)
    loss_l2 = l2_loss(pred, target)
    logger.info(f"L1 loss: {loss_l1.item():.6f}")
    logger.info(f"L2 loss: {loss_l2.item():.6f}")
    logger.info("✓ Basic losses work")

    # Test flow matching loss
    logger.info("\n=== Test 2: Flow matching loss ===")
    model = create_model(
        architecture="concat",
        x_dim=dim,
        c_dim=c_dim,
        output_dim=dim,
        hidden_dim=32,
        n_layers=2,
        activation="relu",
        use_time=True,
    )

    loss_flow = flow_matching_loss(model, x_0, x_1, c, t)
    logger.info(f"Flow matching loss: {loss_flow.item():.6f}")
    logger.info("✓ Flow matching loss works")

    # Test straight flow loss
    logger.info("\n=== Test 3: Straight flow loss ===")
    loss_straight = straight_flow_loss(model, x_0, x_1, c, t)
    logger.info(f"Straight flow loss: {loss_straight.item():.6f}")
    logger.info("✓ Straight flow loss works")

    # Test regression loss
    logger.info("\n=== Test 4: Regression loss ===")
    model_reg = create_model(
        architecture="concat",
        x_dim=dim,
        c_dim=c_dim,
        output_dim=dim,
        hidden_dim=32,
        n_layers=2,
        activation="relu",
        use_time=False,
    )

    loss_reg_l1 = regression_loss(model_reg, c, target, x_dim=dim, loss_type="l1")
    loss_reg_l2 = regression_loss(model_reg, c, target, x_dim=dim, loss_type="l2")
    logger.info(f"Regression L1 loss: {loss_reg_l1.item():.6f}")
    logger.info(f"Regression L2 loss: {loss_reg_l2.item():.6f}")
    logger.info("✓ Regression losses work")

    # Test LossManager (regression mode)
    logger.info("\n=== Test 5: LossManager (regression mode) ===")
    loss_manager_reg = LossManager(mode="regression", loss_type="l2", x_dim=dim)
    loss_managed = loss_manager_reg.compute_loss(model_reg, c, target)
    logger.info(f"LossManager (regression): {loss_managed.item():.6f}")
    logger.info(f"Matches direct call: {torch.allclose(loss_managed, loss_reg_l2)}")
    logger.info("✓ LossManager works in regression mode")

    # Test LossManager (flow mode)
    logger.info("\n=== Test 6: LossManager (flow mode) ===")
    loss_manager_flow = LossManager(mode="flow", x_dim=dim)
    loss_managed_flow = loss_manager_flow.compute_loss(model, x_0, x_1, c, t)
    logger.info(f"LossManager (flow): {loss_managed_flow.item():.6f}")
    logger.info(f"Matches direct call: {torch.allclose(loss_managed_flow, loss_flow)}")
    logger.info("✓ LossManager works in flow mode")

    # Test LossManager (straight_flow mode)
    logger.info("\n=== Test 7: LossManager (straight_flow mode) ===")
    loss_manager_straight = LossManager(mode="straight_flow", x_dim=dim)
    loss_managed_straight = loss_manager_straight.compute_loss(model, x_0, x_1, c, t)
    logger.info(f"LossManager (straight_flow): {loss_managed_straight.item():.6f}")
    logger.info(
        f"Matches direct call: {torch.allclose(loss_managed_straight, loss_straight)}"
    )
    logger.info("✓ LossManager works in straight_flow mode")

    # Test gradient flow
    logger.info("\n=== Test 8: Gradient flow ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss = flow_matching_loss(model, x_0, x_1, c, t)
    logger.info(f"Initial loss: {loss.item():.6f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_after = flow_matching_loss(model, x_0, x_1, c, t)
    logger.info(f"Loss after 1 step: {loss_after.item():.6f}")
    logger.info(f"Loss changed: {loss_after.item() != loss.item()}")
    logger.info("✓ Gradients flow correctly")

    logger.info("\n" + "=" * 60)
    logger.info("All loss tests passed! ✓")
    logger.info("=" * 60)
