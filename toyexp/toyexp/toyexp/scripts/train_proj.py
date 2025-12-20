"""
Training script for projection experiment.

Compare regression vs flow models with high-dimensional output in low-dimensional subspace.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from toyexp.common.config import (
    apply_overrides,
    load_config,
    parse_override_args,
    save_config,
    validate_config,
)
from toyexp.common.datasets import ProjectedTargetFunctionDataset
from toyexp.common.integrate import integrate
from toyexp.common.logging_utils import (
    create_metrics_logger,
    get_logger,
    log_config,
    log_evaluation,
    log_model_info,
    log_training_step,
    setup_logging,
)
from toyexp.common.losses import LossManager
from toyexp.common.networks import create_model
from toyexp.common.utils import (
    plot_errors,
    plot_predictions,
    plot_training_curves,
    save_checkpoint,
    set_seed,
)

logger = logging.getLogger(__name__)


def create_datasets(config):
    """Create training and evaluation datasets from config."""
    logger.info("Creating datasets...")

    # Training dataset
    train_dataset = ProjectedTargetFunctionDataset(
        num_samples=config.dataset.num_train,
        target_dim=config.dataset.target_dim,
        condition_dim=config.dataset.condition_dim,
        low_dim=config.dataset.low_dim,
        num_components=config.dataset.num_components,
        c_min=config.dataset.c_min,
        c_max=config.dataset.c_max,
        weight_strategy=config.dataset.weight_strategy,
        sampling_strategy=config.dataset.sampling_strategy,
        freq_seed=config.dataset.freq_seed,
        phase_seed=config.dataset.phase_seed,
        weight_seed=config.dataset.weight_seed,
        proj_seed=config.dataset.proj_seed,
        sample_seed=config.dataset.sample_seed,
    )

    logger.info(f"Training dataset: {len(train_dataset)} samples")
    logger.info(
        f"Projection: {config.dataset.target_dim}D -> {config.dataset.low_dim}D subspace"
    )

    return train_dataset


def train_epoch(model, dataloader, loss_manager, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        c = batch["c"].to(device)
        x_1 = batch["x"].to(device)

        # Compute loss
        if config.experiment.mode == "regression":
            loss = loss_manager.compute_loss(model, c, x_1)
        elif config.experiment.mode in ["mip", "mip_one_step_integrate"]:
            # MIP mode: needs x_0, x_1, c (no t sampled)
            # Initial distribution
            if config.training.initial_dist == "gaussian":
                x_0 = torch.randn_like(x_1)
            else:
                x_0 = torch.zeros_like(x_1)

            loss = loss_manager.compute_loss(model, x_0, x_1, c)
        elif config.experiment.mode in ["flow", "straight_flow"]:
            # Sample time uniformly
            batch_size = c.shape[0]
            t = torch.rand(batch_size, 1, device=device)

            # Initial distribution
            if config.training.initial_dist == "gaussian":
                x_0 = torch.randn_like(x_1)
            else:
                x_0 = torch.zeros_like(x_1)

            loss = loss_manager.compute_loss(model, x_0, x_1, c, t)
        else:
            raise ValueError(f"Unknown mode: {config.experiment.mode}")

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    return epoch_loss / num_batches


def analyze_subspace(model, dataset, device, config):
    """
    Analyze learned vs true subspace structure.

    NOTE: This function is deprecated for interval-based projections.
    It will be replaced with evaluate_subspace_metric() in Phase 2.

    Returns dictionary with subspace analysis metrics.
    """
    # Skip analysis for now - will be replaced with interval-aware metrics
    logger.info(
        "Subspace analysis: Skipping (deprecated for interval-based projections)"
    )
    return {}


def evaluate_subspace_metric(
    model,
    dataset,
    device,
    config,
    nfe: int,
    n_test_per_interval: int = 20,
):
    """
    Evaluate subspace complement metric using interval regions.

    For each (i,j) pair in 10x10 matrix:
    - i = test interval (where we sample points)
    - j = projection matrix to test against

    Computes 4 metrics measuring how much predictions "leak" outside the correct subspace:
    1. ||(I - P_j) f_hat||
    2. ||(I - P_j) f_hat|| / ||f_hat||
    3. ||(I - P_j) (f_hat - f_true)||
    4. ||(I - P_j) (f_hat - f_true)|| / ||f_hat - f_true||

    Expected pattern:
    - Diagonal (i=j): LOW values ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ predictions stay in correct subspace
    - Off-diagonal (iÃƒÂ¢Ã¢â‚¬Â°Ã‚Â j): HIGH values ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ predictions orthogonal to wrong subspaces

    Args:
        model: Trained model
        dataset: ProjectedTargetFunctionDataset with interval-based projections
        device: torch device
        config: Configuration object
        n_test_per_interval: Number of test points per interval

    Returns:
        Dictionary with metric matrices and summary statistics
    """
    model.eval()

    num_intervals = dataset.num_intervals
    target_dim = dataset.target_dim
    c_min = dataset.c_min
    c_max = dataset.c_max

    results = {}

    with torch.no_grad():
        # Create 10x10 matrices for each of the four metrics
        matrices = {
            1: np.zeros((num_intervals, num_intervals)),  # ||(I - P_j) f_hat||
            2: np.zeros((num_intervals, num_intervals)),  # normalized
            3: np.zeros(
                (num_intervals, num_intervals)
            ),  # ||(I - P_j) (f_hat - f_true)||
            4: np.zeros((num_intervals, num_intervals)),  # normalized
        }

        # For each test interval i
        for i in range(num_intervals):
            # Sample points from interval i
            interval_start = c_min + (i / num_intervals) * (c_max - c_min)
            interval_end = c_min + ((i + 1) / num_intervals) * (c_max - c_min)

            test_points = np.linspace(interval_start, interval_end, n_test_per_interval)
            c_tensor = torch.FloatTensor(test_points).unsqueeze(1).to(device)

            # Get model predictions
            if config.training.initial_dist == "gaussian":
                x_0 = torch.randn(len(test_points), target_dim, device=device)
            else:
                x_0 = torch.zeros(len(test_points), target_dim, device=device)

            f_hat = integrate(
                model=model,
                x_0=x_0,
                c=c_tensor,
                n_steps=nfe,
                method=config.evaluation.integration_method,
                mode=config.experiment.mode,
                mip_t_star=config.training.get("mip_t_star", 0.9),
            )
            f_hat_np = f_hat.cpu().numpy()

            # Get ground truth values (from dataset's base function + projection)
            eval_data = dataset.base_dataset.generate_eval_data(
                num_samples=len(test_points),
                eval_seed=config.experiment.seed + 3000 + i,
            )
            # Manually set c values to our test points
            eval_data["c"] = c_tensor.cpu()
            f_base = eval_data["x"].numpy()

            # Apply true projections to get f_true
            f_true_list = []
            for idx, c_val in enumerate(test_points):
                interval_id = dataset._get_interval_id(c_val)
                P_true = dataset.projection_matrices[interval_id].numpy()
                f_true_list.append(P_true @ f_base[idx])
            f_true_np = np.array(f_true_list)

            # For each projection matrix j
            for j in range(num_intervals):
                # Get projection matrix P_j
                P_j = dataset.projection_matrices[j].numpy()

                # Compute complement projection (I - P_j)
                I_minus_P = np.eye(target_dim) - P_j

                # Metric 1: ||(I - P_j) f_hat||
                complement_f_hat = f_hat_np @ I_minus_P.T
                complement_f_hat_norms = np.linalg.norm(complement_f_hat, axis=1)
                matrices[1][i, j] = np.mean(complement_f_hat_norms)

                # Metric 2: ||(I - P_j) f_hat|| / ||f_hat||
                f_hat_norms = np.linalg.norm(f_hat_np, axis=1)
                normalized_complement_f_hat = np.divide(
                    complement_f_hat_norms,
                    f_hat_norms,
                    out=np.zeros_like(complement_f_hat_norms),
                    where=(f_hat_norms != 0),
                )
                matrices[2][i, j] = np.mean(normalized_complement_f_hat)

                # Metric 3: ||(I - P_j) (f_hat - f_true)||
                error_vector = f_hat_np - f_true_np
                complement_error = error_vector @ I_minus_P.T
                complement_error_norms = np.linalg.norm(complement_error, axis=1)
                matrices[3][i, j] = np.mean(complement_error_norms)

                # Metric 4: ||(I - P_j) (f_hat - f_true)|| / ||f_hat - f_true||
                error_norms = np.linalg.norm(error_vector, axis=1)
                normalized_complement_error = np.divide(
                    complement_error_norms,
                    error_norms,
                    out=np.zeros_like(complement_error_norms),
                    where=(error_norms != 0),
                )
                matrices[4][i, j] = np.mean(normalized_complement_error)

        # Store all metric matrices
        for metric_id in [1, 2, 3, 4]:
            results[f"subspace_complement_matrix_{metric_id}"] = matrices[metric_id]

            # Add summary statistics
            diagonal_mean = np.mean(np.diag(matrices[metric_id]))
            off_diagonal_mask = ~np.eye(num_intervals, dtype=bool)
            off_diagonal_mean = np.mean(matrices[metric_id][off_diagonal_mask])

            results[f"subspace_diagonal_mean_{metric_id}"] = diagonal_mean
            results[f"subspace_off_diagonal_mean_{metric_id}"] = off_diagonal_mean

    model.train()
    return results


def evaluate_subspace_metric_adjacent(
    model,
    dataset,
    device,
    config,
    nfe: int,
    n_test_per_boundary: int = 20,
    boundary_width: float = 0.03,
):
    """
    Evaluate subspace metrics at boundary regions between adjacent intervals.

    For each boundary point c_i (i=1 to 9), tests points in [c_i - width, c_i + width]
    against combined projection subspace P_{i-1} ÃƒÂ¢Ã‹â€ Ã‚Âª P_i.

    Measures whether the model creates smooth transitions or "jumps" between subspaces.

    Computes 4 metrics:
    1. ||(I - P_{i-1,i}) f_hat||
    2. ||(I - P_{i-1,i}) f_hat|| / ||f_hat||
    3. ||(I - P_{i-1,i}) (f_hat - f_true)||
    4. ||(I - P_{i-1,i}) (f_hat - f_true)|| / ||f_hat - f_true||

    Args:
        model: Trained model
        dataset: ProjectedTargetFunctionDataset with interval-based projections
        device: torch device
        config: Configuration object
        n_test_per_boundary: Number of test points per boundary region
        boundary_width: Width of boundary region around each boundary point

    Returns:
        Dictionary with boundary metrics for all 4 metrics
    """
    model.eval()

    num_intervals = dataset.num_intervals
    target_dim = dataset.target_dim
    low_dim = dataset.low_dim
    c_min = dataset.c_min
    c_max = dataset.c_max

    results = {}

    with torch.no_grad():
        # Initialize arrays for each metric
        boundary_metrics = {
            1: [],  # ||(I - P) f_hat||
            2: [],  # normalized
            3: [],  # ||(I - P) (f_hat - f_true)||
            4: [],  # normalized
        }

        # For each boundary point c_i (skip c_0=c_min and c_10=c_max)
        for i in range(1, num_intervals):
            c_i = c_min + (i / num_intervals) * (c_max - c_min)

            # Create test interval [c_i - width, c_i + width]
            interval_start = max(c_min, c_i - boundary_width)
            interval_end = min(c_max, c_i + boundary_width)

            test_points = np.linspace(interval_start, interval_end, n_test_per_boundary)
            c_tensor = torch.FloatTensor(test_points).unsqueeze(1).to(device)

            # Get model predictions
            if config.training.initial_dist == "gaussian":
                x_0 = torch.randn(len(test_points), target_dim, device=device)
            else:
                x_0 = torch.zeros(len(test_points), target_dim, device=device)

            f_hat = integrate(
                model=model,
                x_0=x_0,
                c=c_tensor,
                n_steps=nfe,
                method=config.evaluation.integration_method,
                mode=config.experiment.mode,
                mip_t_star=config.training.get("mip_t_star", 0.9),
            )
            f_hat_np = f_hat.cpu().numpy()

            # Get ground truth values
            eval_data = dataset.base_dataset.generate_eval_data(
                num_samples=len(test_points),
                eval_seed=config.experiment.seed + 4000 + i,
            )
            eval_data["c"] = c_tensor.cpu()
            f_base = eval_data["x"].numpy()

            # Apply true projections
            f_true_list = []
            for idx, c_val in enumerate(test_points):
                interval_id = dataset._get_interval_id(c_val)
                P_true = dataset.projection_matrices[interval_id].numpy()
                f_true_list.append(P_true @ f_base[idx])
            f_true_np = np.array(f_true_list)

            # Combine P_{i-1} and P_i subspaces
            P_i_minus_1 = dataset.projection_matrices[i - 1].numpy()
            P_i = dataset.projection_matrices[i].numpy()

            # Combine by concatenating column spaces
            combined_columns = np.hstack([P_i_minus_1, P_i])  # (target_dim, 2*low_dim)

            # SVD to get orthonormal basis for combined subspace
            U, s, Vt = np.linalg.svd(combined_columns, full_matrices=False)
            rank_threshold = 1e-10
            valid_dims = np.sum(s > rank_threshold)
            valid_dims = min(valid_dims, 2 * low_dim)  # At most 2*low_dim dimensions

            U_reduced = U[:, :valid_dims]
            P_combined = U_reduced @ U_reduced.T  # Combined projection matrix

            # Compute complement projection (I - P_combined)
            I_minus_P = np.eye(target_dim) - P_combined

            # Metric 1: ||(I - P) f_hat||
            complement_f_hat = f_hat_np @ I_minus_P.T
            complement_f_hat_norms = np.linalg.norm(complement_f_hat, axis=1)
            metric_1 = np.mean(complement_f_hat_norms)

            # Metric 2: ||(I - P) f_hat|| / ||f_hat||
            f_hat_norms = np.linalg.norm(f_hat_np, axis=1)
            normalized_complement_f_hat = np.divide(
                complement_f_hat_norms,
                f_hat_norms,
                out=np.zeros_like(complement_f_hat_norms),
                where=(f_hat_norms != 0),
            )
            metric_2 = np.mean(normalized_complement_f_hat)

            # Metric 3: ||(I - P) (f_hat - f_true)||
            error_vector = f_hat_np - f_true_np
            complement_error = error_vector @ I_minus_P.T
            complement_error_norms = np.linalg.norm(complement_error, axis=1)
            metric_3 = np.mean(complement_error_norms)

            # Metric 4: ||(I - P) (f_hat - f_true)|| / ||f_hat - f_true||
            error_norms = np.linalg.norm(error_vector, axis=1)
            normalized_complement_error = np.divide(
                complement_error_norms,
                error_norms,
                out=np.zeros_like(complement_error_norms),
                where=(error_norms != 0),
            )
            metric_4 = np.mean(normalized_complement_error)

            # Store all metrics
            boundary_metrics[1].append(metric_1)
            boundary_metrics[2].append(metric_2)
            boundary_metrics[3].append(metric_3)
            boundary_metrics[4].append(metric_4)

        # Convert to arrays and store results
        for metric_id in [1, 2, 3, 4]:
            results[f"boundary_metrics_{metric_id}"] = np.array(
                boundary_metrics[metric_id]
            )
            results[f"boundary_mean_{metric_id}"] = np.mean(boundary_metrics[metric_id])

    model.train()
    return results


def plot_subspace_analysis(
    subspace_results: dict,
    save_dir: Path,
    mode: str,
    loss_type: str,
    nfe: int = None,
):
    """
    Plot subspace complement analysis as heatmaps for all 4 metrics.

    Creates a grid of heatmaps showing the 10ÃƒÆ’Ã¢â‚¬â€10 matrices for each metric.

    Args:
        subspace_results: Dictionary from evaluate_subspace_metric()
        save_dir: Directory to save plots
        mode: Experiment mode (regression/flow)
        loss_type: Loss type (l2, etc.)
    """
    # Extract matrices for all 4 metrics
    matrices = {}
    for metric_id in [1, 2, 3, 4]:
        key = f"subspace_complement_matrix_{metric_id}"
        if key in subspace_results:
            matrices[metric_id] = subspace_results[key]

    if not matrices:
        logger.warning("No subspace matrices found for plotting")
        return

    # Metric names with LaTeX formatting
    metric_names = {
        1: r"$\|(I - P_j)\hat{f}\|$",
        2: r"$\frac{\|(I - P_j)\hat{f}\|}{\|\hat{f}\|}$",
        3: r"$\|(I - P_j)(\hat{f} - f_{\mathrm{true}})\|$",
        4: r"$\frac{\|(I - P_j)(\hat{f} - f_{\mathrm{true}})\|}{\|\hat{f} - f_{\mathrm{true}}\|}$",
    }

    # Create figure with subplots for each metric
    n_metrics = len(matrices)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    for idx, metric_id in enumerate(sorted(matrices.keys())):
        matrix = matrices[metric_id]
        ax = axes[idx]

        # Create heatmap
        im = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_title(
            f"Metric {metric_id}: {metric_names[metric_id]}", fontsize=12, pad=20
        )
        ax.set_xlabel("Projection Matrix Index j", fontsize=10)
        ax.set_ylabel("Test Interval Index i", fontsize=10)

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Add text annotations
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text_color = "white" if matrix[i, j] < matrix.max() * 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

        # Set ticks
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))

    title = f"Subspace Analysis - {mode} - {loss_type}"
    if nfe is not None:
        title += f" (NFE={nfe})"
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    # Save plot
    filename = f"subspace_analysis_{mode}_{loss_type}"
    if nfe is not None:
        filename += f"_nfe{nfe}"
    save_path = save_dir / f"{filename}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved subspace analysis plot to {save_path}")


def plot_boundary_analysis(
    boundary_results: dict,
    save_dir: Path,
    mode: str,
    loss_type: str,
    metric_id: int = 2,
    nfe: int = None,
):
    """
    Plot boundary subspace analysis for a specific metric.

    Creates a 10ÃƒÆ’Ã¢â‚¬â€10 matrix visualization where only the off-diagonal (boundary)
    positions are filled, showing the boundary metrics.

    Args:
        boundary_results: Dictionary from evaluate_subspace_metric_adjacent()
        save_dir: Directory to save plots
        mode: Experiment mode (regression/flow)
        loss_type: Loss type (l2, etc.)
        metric_id: Which metric to plot (default: 2 for normalized version)
    """
    key = f"boundary_metrics_{metric_id}"
    if key not in boundary_results:
        logger.warning(f"No boundary metrics found for metric {metric_id}")
        return

    boundary_metrics = boundary_results[key]

    # Metric names
    metric_names = {
        1: r"$\|(I - P)\hat{f}\|$",
        2: r"$\frac{\|(I - P)\hat{f}\|}{\|\hat{f}\|}$",
        3: r"$\|(I - P)(\hat{f} - f_{\mathrm{true}})\|$",
        4: r"$\frac{\|(I - P)(\hat{f} - f_{\mathrm{true}})\|}{\|\hat{f} - f_{\mathrm{true}}\|}$",
    }

    # Create 10ÃƒÆ’Ã¢â‚¬â€10 matrix with NaN for non-boundary positions
    num_intervals = 10
    boundary_matrix = np.full((num_intervals, num_intervals), np.nan)

    # Fill boundary metrics below the main diagonal
    # boundary_metrics[i] corresponds to boundary between interval i and i+1
    for i in range(len(boundary_metrics)):
        # Place at position [i+1, i] (below diagonal)
        boundary_matrix[i + 1, i] = boundary_metrics[i]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot heatmap
    im = ax.imshow(boundary_matrix, cmap="viridis", aspect="auto", vmin=0)
    ax.set_title(
        f"Boundary Analysis - Metric {metric_id}: {metric_names[metric_id]}",
        fontsize=14,
        pad=20,
    )
    ax.set_xlabel("Interval Index", fontsize=12)
    ax.set_ylabel("Interval Index", fontsize=12)

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Add text annotations for non-NaN values
    for i in range(num_intervals):
        for j in range(num_intervals):
            if not np.isnan(boundary_matrix[i, j]):
                # This is a boundary position
                boundary_idx = min(i, j)
                ax.text(
                    j,
                    i,
                    f"{boundary_matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    weight="bold",
                )

    # Set ticks and labels
    ax.set_xticks(range(num_intervals))
    ax.set_yticks(range(num_intervals))
    ax.set_xticklabels([f"$I_{{{i}}}$" for i in range(num_intervals)])
    ax.set_yticklabels([f"$I_{{{i}}}$" for i in range(num_intervals)])

    title = f"Boundary Subspace Analysis - {mode} - {loss_type}"
    if nfe is not None:
        title += f" (NFE={nfe})"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Save plot
    filename = f"boundary_analysis_{mode}_{loss_type}_metric{metric_id}"
    if nfe is not None:
        filename += f"_nfe{nfe}"
    save_path = save_dir / f"{filename}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved boundary analysis plot to {save_path}")


def plot_predictions_grid(
    c_values: np.ndarray,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    save_dir: Path,
    mode: str,
    loss_type: str,
    num_dims_to_plot: int = 8,
    nfe: int = None,
):
    """
    Plot predictions vs true values for multiple dimensions in a grid.

    Args:
        c_values: Conditioning values, shape (n_samples,)
        true_values: True target values, shape (n_samples, target_dim)
        pred_values: Predicted values, shape (n_samples, target_dim)
        save_dir: Directory to save plot
        mode: Experiment mode
        loss_type: Loss type
        num_dims_to_plot: Number of dimensions to plot (default: 8)
        nfe: Number of function evaluations (optional, for filename/title)
    """
    target_dim = true_values.shape[1]
    num_dims_to_plot = min(num_dims_to_plot, target_dim)

    # Create grid: 3 rows x 3 cols = 9 subplots (one extra for flexibility)
    n_cols = 3
    n_rows = (num_dims_to_plot + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if num_dims_to_plot > 1 else [axes]

    for dim in range(num_dims_to_plot):
        ax = axes[dim]

        # Plot true function
        ax.plot(
            c_values, true_values[:, dim], "b-", label="True", linewidth=2, alpha=0.7
        )

        # Plot predictions
        ax.plot(
            c_values,
            pred_values[:, dim],
            "r--",
            label="Predicted",
            linewidth=2,
            alpha=0.7,
        )

        ax.set_title(f"Dimension {dim}", fontsize=12)
        ax.set_xlabel("c (conditioning)", fontsize=10)
        ax.set_ylabel(f"x_{dim}", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for dim in range(num_dims_to_plot, len(axes)):
        axes[dim].set_visible(False)

    title = f"Predictions vs True Values (All Dimensions) - {mode} - {loss_type}"
    if nfe is not None:
        title += f" (NFE={nfe})"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Save plot
    filename = f"predictions_grid_{mode}_{loss_type}"
    if nfe is not None:
        filename += f"_nfe{nfe}"
    save_path = save_dir / f"{filename}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved multi-dimension predictions plot to {save_path}")


def plot_l1_errors_grid(
    c_values: np.ndarray,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    save_dir: Path,
    mode: str,
    loss_type: str,
    num_dims_to_plot: int = 8,
    nfe: int = None,
):
    """
    Plot L1 errors for multiple dimensions in a grid.

    Args:
        c_values: Conditioning values, shape (n_samples,)
        true_values: True target values, shape (n_samples, target_dim)
        pred_values: Predicted values, shape (n_samples, target_dim)
        save_dir: Directory to save plot
        mode: Experiment mode
        loss_type: Loss type
        num_dims_to_plot: Number of dimensions to plot (default: 8)
        nfe: Number of function evaluations (optional, for filename/title)
    """
    target_dim = true_values.shape[1]
    num_dims_to_plot = min(num_dims_to_plot, target_dim)

    # Create grid
    n_cols = 3
    n_rows = (num_dims_to_plot + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if num_dims_to_plot > 1 else [axes]

    for dim in range(num_dims_to_plot):
        ax = axes[dim]

        # Compute L1 errors
        l1_errors = np.abs(pred_values[:, dim] - true_values[:, dim])
        mean_error = np.mean(l1_errors)

        # Plot errors
        ax.scatter(c_values, l1_errors, alpha=0.6, s=10, color="red")
        ax.axhline(
            y=mean_error,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Mean={mean_error:.4f}",
        )

        ax.set_title(f"Dimension {dim}", fontsize=12)
        ax.set_xlabel("c (conditioning)", fontsize=10)
        ax.set_ylabel("L1 Error", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for dim in range(num_dims_to_plot, len(axes)):
        axes[dim].set_visible(False)

    title = f"L1 Errors (All Dimensions) - {mode} - {loss_type}"
    if nfe is not None:
        title += f" (NFE={nfe})"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Save plot
    filename = f"l1_errors_grid_{mode}_{loss_type}"
    if nfe is not None:
        filename += f"_nfe{nfe}"
    save_path = save_dir / f"{filename}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved L1 errors grid to {save_path}")


def plot_l2_errors_grid(
    c_values: np.ndarray,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    save_dir: Path,
    mode: str,
    loss_type: str,
    num_dims_to_plot: int = 8,
    nfe: int = None,
):
    """
    Plot L2 (squared) errors for multiple dimensions in a grid.

    Args:
        c_values: Conditioning values, shape (n_samples,)
        true_values: True target values, shape (n_samples, target_dim)
        pred_values: Predicted values, shape (n_samples, target_dim)
        save_dir: Directory to save plot
        mode: Experiment mode
        loss_type: Loss type
        num_dims_to_plot: Number of dimensions to plot (default: 8)
        nfe: Number of function evaluations (optional, for filename/title)
    """
    target_dim = true_values.shape[1]
    num_dims_to_plot = min(num_dims_to_plot, target_dim)

    # Create grid
    n_cols = 3
    n_rows = (num_dims_to_plot + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if num_dims_to_plot > 1 else [axes]

    for dim in range(num_dims_to_plot):
        ax = axes[dim]

        # Compute L2 (squared) errors
        l2_errors = (pred_values[:, dim] - true_values[:, dim]) ** 2
        mean_error = np.mean(l2_errors)

        # Plot errors
        ax.scatter(c_values, l2_errors, alpha=0.6, s=10, color="orange")
        ax.axhline(
            y=mean_error,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Mean={mean_error:.4f}",
        )

        ax.set_title(f"Dimension {dim}", fontsize=12)
        ax.set_xlabel("c (conditioning)", fontsize=10)
        ax.set_ylabel("L2 Error (squared)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for dim in range(num_dims_to_plot, len(axes)):
        axes[dim].set_visible(False)

    title = f"L2 Errors (All Dimensions) - {mode} - {loss_type}"
    if nfe is not None:
        title += f" (NFE={nfe})"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Save plot
    filename = f"l2_errors_grid_{mode}_{loss_type}"
    if nfe is not None:
        filename += f"_nfe{nfe}"
    save_path = save_dir / f"{filename}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved L2 errors grid to {save_path}")


def evaluate(model, dataset, device, config):
    """Evaluate model on dataset for all NFE values."""
    model.eval()

    # Generate evaluation data
    eval_data = dataset.generate_eval_data(
        num_samples=config.dataset.num_eval,
        eval_seed=config.experiment.seed + 1000,
    )

    c_eval = eval_data["c"].to(device)
    x_true = eval_data["x"].cpu().numpy()

    # Handle num_eval_steps as int or list
    num_eval_steps = config.evaluation.num_eval_steps
    if isinstance(num_eval_steps, int):
        num_eval_steps = [num_eval_steps]

    results = []

    with torch.no_grad():
        # Initial distribution
        if config.evaluation.initial_dist == "gaussian":
            x_0 = torch.randn_like(eval_data["x"]).to(device)
        else:
            x_0 = torch.zeros_like(eval_data["x"]).to(device)

        # Evaluate for each NFE
        for nfe in num_eval_steps:
            # Get predictions
            x_pred = integrate(
                model=model,
                x_0=x_0,
                c=c_eval,
                n_steps=nfe,
                method=config.evaluation.integration_method,
                mode=config.experiment.mode,
                mip_t_star=config.training.get("mip_t_star", 0.9),
            )

            x_pred = x_pred.cpu().numpy()

            # Compute metrics
            l1_error = np.mean(np.abs(x_pred - x_true))
            l2_error = np.sqrt(np.mean((x_pred - x_true) ** 2))

            # Per-dimension errors
            l1_per_dim = np.mean(np.abs(x_pred - x_true), axis=0)
            l2_per_dim = np.sqrt(np.mean((x_pred - x_true) ** 2, axis=0))

            metrics = {
                "nfe": nfe,
                "l1_error": l1_error,
                "l2_error": l2_error,
                "l1_per_dim_mean": np.mean(l1_per_dim),
                "l1_per_dim_std": np.std(l1_per_dim),
                "l2_per_dim_mean": np.mean(l2_per_dim),
                "l2_per_dim_std": np.std(l2_per_dim),
            }

            # Interval-based subspace analysis (always computed in train_proj)
            logger.info(f"Computing interval-based subspace metrics for NFE={nfe}...")

            # Main subspace metrics (10x10 matrices)
            subspace_results = evaluate_subspace_metric(
                model=model,
                dataset=dataset,
                device=device,
                config=config,
                nfe=nfe,
            )

            # Add summary statistics to main metrics (for CSV logging)
            for key, value in subspace_results.items():
                if "mean" in key:  # Only add summary statistics
                    metrics[key] = value

            # Boundary metrics
            boundary_results = evaluate_subspace_metric_adjacent(
                model=model,
                dataset=dataset,
                device=device,
                config=config,
                nfe=nfe,
            )

            # Add boundary summary statistics
            for key, value in boundary_results.items():
                if "mean" in key:
                    metrics[key] = value

            logger.info(
                f"Subspace diagonal mean (metric 4): {subspace_results.get('subspace_diagonal_mean_4', 0):.4f}"
            )
            logger.info(
                f"Subspace off-diagonal mean (metric 4): {subspace_results.get('subspace_off_diagonal_mean_4', 0):.4f}"
            )
            logger.info(
                f"Boundary mean (metric 2): {boundary_results.get('boundary_mean_2', 0):.4f}"
            )

            # Store matrices for plotting (Phase 3)
            metrics["_subspace_matrices"] = subspace_results
            metrics["_boundary_matrices"] = boundary_results

            # Prepare data for plotting
            plot_data = {
                "nfe": nfe,
                "c_values": c_eval.cpu().numpy().flatten(),
                "true_values": x_true,  # All dimensions
                "pred_values": x_pred,  # All dimensions
            }

            results.append((metrics, plot_data))

    return results


def main(config_path: str, overrides: dict = None):
    """Main training function."""
    # Load and validate configuration
    config = load_config(config_path)

    if overrides:
        config = apply_overrides(config, overrides)

    validate_config(config)

    # Build output directory with subdirectories for mode, loss_type, architecture, and seed
    base_output_dir = Path(config.experiment.output_dir)
    output_dir = (
        base_output_dir
        / config.experiment.mode
        / config.training.loss_type
        / config.network.architecture
        / f"seed_{config.experiment.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(
        name="",
        level=logging.INFO,
        log_file=output_dir / "train.log",
    )

    logger.info("=" * 80)
    logger.info(f"Starting {config.experiment.name}")
    logger.info("=" * 80)

    # Save config
    save_config(config, output_dir / "config.yaml")
    log_config(config.to_dict())

    # Set seed
    set_seed(config.experiment.seed)

    # Device
    device = torch.device(
        config.experiment.device
        if torch.cuda.is_available() and config.experiment.device == "cuda"
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Create datasets
    train_dataset = create_datasets(config)

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Create model
    model = create_model(
        architecture=config.network.architecture,
        x_dim=config.dataset.target_dim,
        c_dim=config.dataset.condition_dim,
        output_dim=config.dataset.target_dim,
        hidden_dim=config.network.hidden_dim,
        n_layers=config.network.num_layers,
        activation=config.network.activation,
        use_time=(config.experiment.mode in ["flow", "mip"]),
    ).to(device)

    log_model_info(model)

    # Create loss manager
    loss_manager = LossManager(
        mode=config.experiment.mode,
        loss_type=config.training.loss_type,
        x_dim=config.dataset.target_dim,
        mip_t_star=config.training.get("mip_t_star", 0.9),
    )

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Create CSV metrics logger
    metrics_logger = create_metrics_logger(output_dir, experiment_type="proj")

    # Training loop
    logger.info("Starting training...")
    train_losses = []
    best_l2_error = float("inf")

    for epoch in range(config.training.num_epochs):
        # Train
        epoch_loss = train_epoch(
            model, train_loader, loss_manager, optimizer, device, config
        )
        train_losses.append(epoch_loss)

        # Log to CSV
        metrics_logger.log("training", {"epoch": epoch + 1, "loss": epoch_loss})

        # Log training
        if (epoch + 1) % config.training.log_interval == 0:
            log_training_step(
                epoch=epoch + 1,
                step=epoch + 1,
                loss=epoch_loss,
            )

        # Evaluate
        if (epoch + 1) % config.training.eval_interval == 0:
            results = evaluate(model, train_dataset, device, config)

            # Log all NFE results
            for metrics, plot_data in results:
                # Filter out numpy arrays from logging
                metrics_for_logging = {
                    k: v for k, v in metrics.items() if not k.startswith("_")
                }
                log_evaluation(metrics_for_logging, prefix=f"Epoch {epoch + 1}")

                # Log to CSV
                metrics_logger.log(
                    "evaluation",
                    {
                        "epoch": epoch + 1,
                        "nfe": metrics["nfe"],
                        "l1_error": metrics["l1_error"],
                        "l2_error": metrics["l2_error"],
                    },
                )

            # Save best model (using first NFE for tracking)
            first_metrics = results[0][0]
            if first_metrics["l2_error"] < best_l2_error:
                best_l2_error = first_metrics["l2_error"]
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    loss=epoch_loss,
                    save_dir=output_dir / "checkpoints",
                    filename="best_model.pt",
                    additional_info={"metrics": first_metrics},
                )

        # Save periodic checkpoint
        if (epoch + 1) % config.training.save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=epoch_loss,
                save_dir=output_dir / "checkpoints",
                filename=f"checkpoint_epoch_{epoch+1}.pt",
            )

    # Final evaluation
    logger.info("=" * 80)
    logger.info("Final evaluation...")
    logger.info("=" * 80)

    results = evaluate(model, train_dataset, device, config)

    # Log all NFE results
    for metrics, plot_data in results:
        # Filter out numpy arrays from logging (keep them for plotting)
        metrics_for_logging = {
            k: v for k, v in metrics.items() if not k.startswith("_")
        }
        log_evaluation(metrics_for_logging, prefix="Final")

    # Create plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Training curves
    plot_training_curves(
        {"train_loss": train_losses},
        save_path=plots_dir / "training_curves.png",
        title=f"{config.experiment.name} - Training Curves",
    )

    # Create plots for each NFE
    for metrics, plot_data in results:
        nfe = metrics["nfe"]

        # Multi-dimension predictions grid
        plot_predictions_grid(
            c_values=plot_data["c_values"],
            true_values=plot_data["true_values"],
            pred_values=plot_data["pred_values"],
            save_dir=plots_dir,
            mode=config.experiment.mode,
            loss_type=config.training.loss_type,
            num_dims_to_plot=min(8, config.dataset.target_dim),
            nfe=nfe,
        )

        # L1 errors grid
        plot_l1_errors_grid(
            c_values=plot_data["c_values"],
            true_values=plot_data["true_values"],
            pred_values=plot_data["pred_values"],
            save_dir=plots_dir,
            mode=config.experiment.mode,
            loss_type=config.training.loss_type,
            num_dims_to_plot=min(8, config.dataset.target_dim),
            nfe=nfe,
        )

        # L2 errors grid
        plot_l2_errors_grid(
            c_values=plot_data["c_values"],
            true_values=plot_data["true_values"],
            pred_values=plot_data["pred_values"],
            save_dir=plots_dir,
            mode=config.experiment.mode,
            loss_type=config.training.loss_type,
            num_dims_to_plot=min(8, config.dataset.target_dim),
            nfe=nfe,
        )

        # Subspace analysis plots (always generated in train_proj)
        logger.info(f"Creating subspace analysis plots for NFE={nfe}...")

        # Plot subspace heatmaps
        plot_subspace_analysis(
            subspace_results=metrics["_subspace_matrices"],
            save_dir=plots_dir,
            mode=config.experiment.mode,
            loss_type=config.training.loss_type,
            nfe=nfe,
        )

        # Plot boundary analysis
        plot_boundary_analysis(
            boundary_results=metrics["_boundary_matrices"],
            save_dir=plots_dir,
            mode=config.experiment.mode,
            loss_type=config.training.loss_type,
            metric_id=2,  # Focus on normalized metric
            nfe=nfe,
        )

    # Save final checkpoint (with first NFE metrics)
    first_metrics = results[0][0]
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config.training.num_epochs,
        loss=train_losses[-1],
        save_dir=output_dir / "checkpoints",
        filename="final_model.pt",
        additional_info={"metrics": first_metrics},
    )

    logger.info("Training complete!")
    logger.info(f"Results saved to {output_dir}")

    # Close CSV loggers
    metrics_logger.close_all()

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train projection experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python train_proj.py --config config_proj.yaml
  
  # Override config values using full key paths
  python train_proj.py --config config_proj.yaml \\
      experiment.mode=flow \\
      experiment.seed=123 \\
      training.learning_rate=0.001 \\
      dataset.low_dim=3
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_proj.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in key=value format using full paths (e.g., experiment.mode=flow training.learning_rate=0.01)",
    )

    args = parser.parse_args()

    # Parse overrides from positional arguments
    overrides = None
    if args.overrides:
        overrides = parse_override_args(args.overrides)

    main(args.config, overrides)
