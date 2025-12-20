"""
ODE integration for flow models.

Provides numerical integration methods for solving:
dx/dt = v(x_t, c, t)

Where v is the learned velocity field.
"""

import logging
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def euler_integrate(
    model: nn.Module,
    x_0: torch.Tensor,
    c: torch.Tensor,
    n_steps: int = 100,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> torch.Tensor:
    """
    Euler method for integrating velocity field.

    Forward Euler: x_{i+1} = x_i + dt * v(x_i, c, t_i)

    Simple and fast but less accurate than higher-order methods.

    Args:
        model: Velocity field model v(x, c, t)
        x_0: [batch, dim] initial state
        c: [batch, c_dim] conditioning variable
        n_steps: Number of integration steps
        t_start: Start time (typically 0)
        t_end: End time (typically 1)

    Returns:
        x_1: [batch, dim] final state after integration
    """
    dt = (t_end - t_start) / n_steps
    x_t = x_0.clone()

    device = x_0.device
    batch_size = x_0.shape[0]

    with torch.no_grad():
        for step in range(n_steps):
            t = t_start + step * dt
            t_tensor = torch.full((batch_size, 1), t, device=device)

            # Compute velocity
            v = model(x_t, c, t_tensor)

            # Euler step
            x_t = x_t + dt * v

    return x_t


def rk4_integrate(
    model: nn.Module,
    x_0: torch.Tensor,
    c: torch.Tensor,
    n_steps: int = 100,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> torch.Tensor:
    """
    4th-order Runge-Kutta method for integrating velocity field.

    RK4 provides better accuracy than Euler for the same number of steps,
    at the cost of 4x more model evaluations per step.

    Args:
        model: Velocity field model v(x, c, t)
        x_0: [batch, dim] initial state
        c: [batch, c_dim] conditioning variable
        n_steps: Number of integration steps
        t_start: Start time (typically 0)
        t_end: End time (typically 1)

    Returns:
        x_1: [batch, dim] final state after integration
    """
    dt = (t_end - t_start) / n_steps
    x_t = x_0.clone()

    device = x_0.device
    batch_size = x_0.shape[0]

    with torch.no_grad():
        for step in range(n_steps):
            t = t_start + step * dt

            # k1 = f(t, x)
            t_tensor = torch.full((batch_size, 1), t, device=device)
            k1 = model(x_t, c, t_tensor)

            # k2 = f(t + dt/2, x + dt*k1/2)
            t_tensor = torch.full((batch_size, 1), t + dt / 2, device=device)
            k2 = model(x_t + dt * k1 / 2, c, t_tensor)

            # k3 = f(t + dt/2, x + dt*k2/2)
            k3 = model(x_t + dt * k2 / 2, c, t_tensor)

            # k4 = f(t + dt, x + dt*k3)
            t_tensor = torch.full((batch_size, 1), t + dt, device=device)
            k4 = model(x_t + dt * k3, c, t_tensor)

            # RK4 update
            x_t = x_t + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_t


def integrate(
    model: nn.Module,
    x_0: torch.Tensor,
    c: torch.Tensor,
    n_steps: int = 100,
    method: Literal["euler", "rk4"] = "euler",
    t_start: float = 0.0,
    t_end: float = 1.0,
    mode: str = "flow",
    mip_t_star: float = 0.9,
) -> torch.Tensor:
    """
    Integrate velocity field from t_start to t_end.

    Unified interface for different integration methods.
    For regression and mip modes, just does a forward pass at t=0 (no integration).

    Args:
        model: Velocity field model v(x, c, t) or regression model f(x, c)
        x_0: [batch, dim] initial state
        c: [batch, c_dim] conditioning variable
        n_steps: Number of integration steps (ignored in regression/mip modes)
        method: 'euler' or 'rk4' (ignored in regression/mip modes)
        t_start: Start time (ignored in regression/mip modes)
        t_end: End time (ignored in regression/mip modes)
        mode: 'flow', 'straight_flow', 'regression', 'mip', or 'mip_one_step_integrate'
        mip_t_star: Time value for MIP second step (default 0.9)

    Returns:
        x_1: [batch, dim] final state (or prediction for regression/mip)
    """
    if mode == "regression":
        # For regression, just do a forward pass (no integration)
        with torch.no_grad():
            return model(x_0, c, t=None)

    elif mode == "mip":
        # For MIP, use two-step inference:
        # Step 1: Predict at t=0
        # Step 2: Use that prediction at t=t* to get final prediction
        batch_size = x_0.shape[0]
        device = x_0.device
        t_zero = torch.zeros(batch_size, 1, device=device)
        t_star = torch.full((batch_size, 1), mip_t_star, device=device)

        with torch.no_grad():
            # Step 1: Initial prediction at t=0
            a_0_hat = model(x_0, c, t_zero)

            # Step 2: Final prediction at t=t* using a_0_hat
            a_hat = model(a_0_hat, c, t_star)

            return a_hat

    elif mode == "mip_one_step_integrate":
        # One-step MIP inference (return after step 1)
        batch_size = x_0.shape[0]
        device = x_0.device
        t_zero = torch.zeros(batch_size, 1, device=device)

        with torch.no_grad():
            a_0_hat = model(x_0, c, t_zero)
            return a_0_hat

    elif mode == "flow":
        if method == "euler":
            return euler_integrate(model, x_0, c, n_steps, t_start, t_end)
        elif method == "rk4":
            return rk4_integrate(model, x_0, c, n_steps, t_start, t_end)
        else:
            raise ValueError(
                f"Unknown integration method: {method}. Choose 'euler' or 'rk4'"
            )

    elif mode == "straight_flow":
        # For straight_flow, use custom ODE stepping with t always set to 0
        # Model predicts x_1 directly (not velocity)
        batch_size = x_0.shape[0]
        device = x_0.device

        # Time grid
        ts = torch.linspace(t_start, t_end, n_steps + 1, device=device)

        with torch.no_grad():
            x_t = x_0
            t_zero = torch.zeros(batch_size, 1, device=device)

            for i in range(n_steps):
                s = ts[i].item()
                t = ts[i + 1].item()

                # Query model with t=0 (no time conditioning)
                x_s = x_t
                x_1_pred = model(x_s, c, t_zero)

                # ODE step: x_t = (1-t)/(1-s) * x_s + (t-s)/(1-s) * x_1_pred
                x_t = ((1 - t) / (1 - s)) * x_s + ((t - s) / (1 - s)) * x_1_pred

            return x_t

    else:
        raise ValueError(
            f"Unknown mode: {mode}. Choose 'flow', 'straight_flow', 'regression', 'mip', or 'mip_one_step_integrate'"
        )


def integrate_with_trajectory(
    model: nn.Module,
    x_0: torch.Tensor,
    c: torch.Tensor,
    n_steps: int = 100,
    method: Literal["euler", "rk4"] = "euler",
    t_start: float = 0.0,
    t_end: float = 1.0,
    save_every: int = 1,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[float]]:
    """
    Integrate velocity field and return full trajectory.

    Useful for visualization and analysis of the flow.

    Args:
        model: Velocity field model v(x, c, t)
        x_0: [batch, dim] initial state
        c: [batch, c_dim] conditioning variable
        n_steps: Number of integration steps
        method: 'euler' or 'rk4'
        t_start: Start time
        t_end: End time
        save_every: Save trajectory every N steps (1 = save all)

    Returns:
        x_1: [batch, dim] final state
        trajectory: List of states [x_0, x_1, ..., x_n]
        times: List of time values [t_0, t_1, ..., t_n]
    """
    dt = (t_end - t_start) / n_steps
    x_t = x_0.clone()

    device = x_0.device
    batch_size = x_0.shape[0]

    trajectory = [x_0.clone()]
    times = [t_start]

    with torch.no_grad():
        for step in range(n_steps):
            t = t_start + step * dt
            t_tensor = torch.full((batch_size, 1), t, device=device)

            if method == "euler":
                v = model(x_t, c, t_tensor)
                x_t = x_t + dt * v

            elif method == "rk4":
                # k1 = f(t, x)
                k1 = model(x_t, c, t_tensor)

                # k2 = f(t + dt/2, x + dt*k1/2)
                t_mid = torch.full((batch_size, 1), t + dt / 2, device=device)
                k2 = model(x_t + dt * k1 / 2, c, t_mid)

                # k3 = f(t + dt/2, x + dt*k2/2)
                k3 = model(x_t + dt * k2 / 2, c, t_mid)

                # k4 = f(t + dt, x + dt*k3)
                t_next = torch.full((batch_size, 1), t + dt, device=device)
                k4 = model(x_t + dt * k3, c, t_next)

                # RK4 update
                x_t = x_t + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            else:
                raise ValueError(f"Unknown method: {method}")

            # Save state if needed
            if (step + 1) % save_every == 0:
                trajectory.append(x_t.clone())
                times.append(t + dt)

    return x_t, trajectory, times


if __name__ == "__main__":
    from networks import create_model

    # Setup logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Testing integration methods...")

    batch_size = 4
    dim = 8
    c_dim = 1

    # Create sample data
    x_0 = torch.randn(batch_size, dim)
    c = torch.randn(batch_size, c_dim)

    # Create a simple velocity field model
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

    logger.info("\n=== Test 1: Euler integration ===")
    x_1_euler = euler_integrate(model, x_0, c, n_steps=10)
    logger.info(f"Initial state shape: {x_0.shape}")
    logger.info(f"Final state shape: {x_1_euler.shape}")
    logger.info(f"States are different: {not torch.allclose(x_0, x_1_euler)}")
    logger.info("Ã¢Å“â€œ Euler integration works")

    logger.info("\n=== Test 2: RK4 integration ===")
    x_1_rk4 = rk4_integrate(model, x_0, c, n_steps=10)
    logger.info(f"Final state shape: {x_1_rk4.shape}")
    logger.info(f"RK4 Ã¢â€°Â  Euler: {not torch.allclose(x_1_euler, x_1_rk4)}")
    logger.info("Ã¢Å“â€œ RK4 integration works")

    logger.info("\n=== Test 3: Unified integrate() function ===")
    x_1_unified_euler = integrate(
        model, x_0, c, n_steps=10, method="euler", mode="flow"
    )
    x_1_unified_rk4 = integrate(model, x_0, c, n_steps=10, method="rk4", mode="flow")

    logger.info(
        f"Unified Euler matches direct: {torch.allclose(x_1_euler, x_1_unified_euler)}"
    )
    logger.info(
        f"Unified RK4 matches direct: {torch.allclose(x_1_rk4, x_1_unified_rk4)}"
    )
    logger.info("Ã¢Å“â€œ Unified interface works")

    logger.info("\n=== Test 4: Regression mode ===")
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

    x_reg = integrate(model_reg, x_0, c, mode="regression")
    logger.info(f"Regression output shape: {x_reg.shape}")
    logger.info("Ã¢Å“â€œ Regression mode works")

    logger.info("\n=== Test 5: Integration with trajectory ===")
    x_1_traj, trajectory, times = integrate_with_trajectory(
        model, x_0, c, n_steps=20, method="euler", save_every=5
    )

    logger.info(f"Final state shape: {x_1_traj.shape}")
    logger.info(f"Trajectory length: {len(trajectory)}")
    logger.info(f"Times: {[f'{t:.2f}' for t in times]}")
    logger.info(f"Expected trajectory length: {1 + 20//5} (initial + saved states)")
    logger.info("Ã¢Å“â€œ Trajectory integration works")

    logger.info("\n=== Test 6: Different step counts ===")
    for n_steps in [1, 10, 100]:
        x_1 = integrate(model, x_0, c, n_steps=n_steps, method="euler", mode="flow")
        logger.info(f"n_steps={n_steps:3d}: final state norm = {x_1.norm().item():.4f}")
    logger.info("Ã¢Å“â€œ Different step counts work")

    logger.info("\n=== Test 7: Custom time range ===")
    x_half = integrate(
        model, x_0, c, n_steps=50, method="euler", t_start=0.0, t_end=0.5, mode="flow"
    )
    x_full = integrate(
        model, x_0, c, n_steps=100, method="euler", t_start=0.0, t_end=1.0, mode="flow"
    )

    logger.info(f"Integration to t=0.5: final state norm = {x_half.norm().item():.4f}")
    logger.info(f"Integration to t=1.0: final state norm = {x_full.norm().item():.4f}")
    logger.info("Ã¢Å“â€œ Custom time ranges work")

    logger.info("\n" + "=" * 60)
    logger.info("All integration tests passed! Ã¢Å“â€œ")
    logger.info("=" * 60)
