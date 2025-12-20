"""
Utility functions for toy experiments.

This module provides:
- Seeding utilities for reproducibility
- Checkpoint saving/loading
- Plotting and visualization
- Command-line override parsing
"""

import logging
from pathlib import Path
import random
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Seeding Utilities
# =============================================================================


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed}")


# =============================================================================
# Checkpoint Utilities
# =============================================================================


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    save_dir: Union[str, Path],
    filename: Optional[str] = None,
    additional_info: Optional[Dict] = None,
) -> Path:
    """
    Save model checkpoint with training state.

    Args:
        model: The model to save
        optimizer: Optimizer state (optional)
        epoch: Current epoch number
        loss: Current loss value
        save_dir: Directory to save checkpoint
        filename: Optional custom filename (default: checkpoint_epoch_{epoch}.pt)
        additional_info: Optional dict with extra info to save

    Returns:
        Path to saved checkpoint
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pt"

    checkpoint_path = save_dir / filename

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if additional_info is not None:
        checkpoint.update(additional_info)

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Load model checkpoint and optionally restore optimizer state.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load model onto

    Returns:
        Dict containing checkpoint information (epoch, loss, etc.)
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded model from {checkpoint_path}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Loaded optimizer state")

    return checkpoint


# =============================================================================
# Plotting Utilities
# =============================================================================


def plot_training_curves(
    losses: Dict[str, list],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Training Curves",
    log_scale: bool = True,
) -> plt.Figure:
    """
    Plot training loss curves.

    Args:
        losses: Dict mapping loss names to lists of values
        save_path: Optional path to save figure
        title: Plot title
        log_scale: Whether to use log scale for y-axis

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, values in losses.items():
        ax.plot(values, label=name, linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale("log")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_predictions(
    c_values: np.ndarray,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Predictions vs Ground Truth",
    xlabel: str = "c",
    ylabel: str = "f(c)",
    nfe: Optional[int] = None,
) -> plt.Figure:
    """
    Plot predictions against ground truth.

    Args:
        c_values: Conditioning values
        true_values: Ground truth values
        pred_values: Predicted values
        save_path: Optional path to save figure
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        nfe: Optional number of function evaluations to include in title

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        c_values, true_values, alpha=0.6, s=20, label="Ground Truth", color="blue"
    )
    ax.scatter(c_values, pred_values, alpha=0.6, s=20, label="Predictions", color="red")

    # Add NFE to title if provided
    plot_title = title if nfe is None else f"{title} (NFE={nfe})"

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(plot_title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_errors(
    c_values: np.ndarray,
    errors: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Prediction Errors",
    xlabel: str = "c",
    ylabel: str = "Error",
    error_type: str = "L2",
    nfe: Optional[int] = None,
) -> plt.Figure:
    """
    Plot prediction errors.

    Args:
        c_values: Conditioning values
        errors: Error values
        save_path: Optional path to save figure
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        error_type: Type of error (for legend)
        nfe: Optional number of function evaluations to include in title

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        c_values, errors, alpha=0.6, s=20, label=f"{error_type} Error", color="red"
    )

    # Add mean error line
    mean_error = np.mean(errors)
    ax.axhline(
        y=mean_error,
        color="black",
        linestyle="--",
        label=f"Mean Error: {mean_error:.4f}",
    )

    # Add NFE to title if provided
    plot_title = title if nfe is None else f"{title} (NFE={nfe})"

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(plot_title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_comparison_grid(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Model Comparison",
) -> plt.Figure:
    """
    Create a grid comparing multiple models/methods.

    Args:
        results_dict: Dict mapping method names to dicts containing
                     'c_values', 'true_values', 'pred_values'
        save_path: Optional path to save figure
        title: Overall plot title

    Returns:
        Matplotlib figure object
    """
    num_methods = len(results_dict)
    fig, axes = plt.subplots(1, num_methods, figsize=(6 * num_methods, 5))

    if num_methods == 1:
        axes = [axes]

    for ax, (method_name, data) in zip(axes, results_dict.items()):
        c_vals = data["c_values"]
        true_vals = data["true_values"]
        pred_vals = data["pred_values"]

        ax.scatter(
            c_vals, true_vals, alpha=0.6, s=20, label="Ground Truth", color="blue"
        )
        ax.scatter(c_vals, pred_vals, alpha=0.6, s=20, label="Predictions", color="red")

        # Calculate error
        error = np.mean(np.abs(pred_vals - true_vals))

        ax.set_xlabel("c", fontsize=10)
        ax.set_ylabel("f(c)", fontsize=10)
        ax.set_title(
            f"{method_name}\nMean L1: {error:.4f}", fontsize=11, fontweight="bold"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


# =============================================================================
# Command-Line Override Utilities
# =============================================================================


def build_experiment_name(
    config, mode: Optional[str] = None, seed: Optional[int] = None
) -> str:
    """
    Build experiment name from config.

    Args:
        config: Configuration object
        mode: Optional mode override
        seed: Optional seed override

    Returns:
        Experiment name string (e.g., "recon_regression_seed42")
    """
    exp_type = "recon" if config.dataset.target_dim == 1 else "proj"
    exp_mode = mode if mode is not None else config.experiment.mode
    exp_seed = seed if seed is not None else config.experiment.seed

    return f"{exp_type}_{exp_mode}_seed{exp_seed}"


# =============================================================================
# Testing Code
# =============================================================================

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Testing utility functions...")
    logger.info("=" * 60)

    # Test 1: Seeding
    logger.info("\nTest 1: Seeding")
    set_seed(42)
    val1 = torch.rand(3)
    set_seed(42)
    val2 = torch.rand(3)
    assert torch.allclose(val1, val2), "Seeding failed!"
    logger.info("Ã¢Å“â€œ Seeding works")

    # Test 2: Checkpoint save/load
    logger.info("\nTest 2: Checkpoint save/load")
    import tempfile

    from networks import create_model

    model = create_model(
        architecture="concat",
        x_dim=8,
        c_dim=1,
        output_dim=8,
        hidden_dim=32,
        n_layers=2,
        activation="relu",
        use_time=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=10,
            loss=0.123,
            save_dir=tmpdir,
            filename="test.pt",
        )

        # Load
        model2 = create_model(
            architecture="concat",
            x_dim=8,
            c_dim=1,
            output_dim=8,
            hidden_dim=32,
            n_layers=2,
            activation="relu",
            use_time=False,
        )

        checkpoint = load_checkpoint(
            checkpoint_path=Path(tmpdir) / "test.pt",
            model=model2,
        )

        assert checkpoint["epoch"] == 10
        assert checkpoint["loss"] == 0.123

    logger.info("Ã¢Å“â€œ Checkpoint save/load works")

    # Test 3: Plotting functions
    logger.info("\nTest 3: Plotting functions")

    c_vals = np.linspace(0, 10, 100)
    true_vals = np.sin(c_vals)
    pred_vals = np.sin(c_vals) + np.random.normal(0, 0.1, 100)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Training curves
        losses = {"train": np.exp(-np.linspace(0, 5, 100))}
        fig1 = plot_training_curves(losses, save_path=Path(tmpdir) / "curves.png")
        plt.close(fig1)

        # Predictions
        fig2 = plot_predictions(
            c_vals, true_vals, pred_vals, save_path=Path(tmpdir) / "pred.png"
        )
        plt.close(fig2)

        # Errors
        errors = np.abs(pred_vals - true_vals)
        fig3 = plot_errors(c_vals, errors, save_path=Path(tmpdir) / "errors.png")
        plt.close(fig3)

        # Comparison grid
        results = {
            "Method 1": {
                "c_values": c_vals,
                "true_values": true_vals,
                "pred_values": pred_vals,
            },
            "Method 2": {
                "c_values": c_vals,
                "true_values": true_vals,
                "pred_values": true_vals + np.random.normal(0, 0.05, 100),
            },
        }
        fig4 = plot_comparison_grid(results, save_path=Path(tmpdir) / "comp.png")
        plt.close(fig4)

        # Check files exist
        assert (Path(tmpdir) / "curves.png").exists()
        assert (Path(tmpdir) / "pred.png").exists()
        assert (Path(tmpdir) / "errors.png").exists()
        assert (Path(tmpdir) / "comp.png").exists()

    logger.info("Ã¢Å“â€œ Plotting functions work")

    logger.info("\n" + "=" * 60)
    logger.info("All utility tests passed! Ã¢Å“â€œ")
    logger.info("=" * 60)
