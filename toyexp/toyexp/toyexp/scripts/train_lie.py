"""
Training script for Lie algebra rotation experiment (REDESIGNED).

Compare regression vs flow models on rotation group functions:
f_i(alpha, c) = w_i(c) * exp(alpha_i*c * A) * e_1

Output is concat(f_1, f_2, ..., f_K)
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
from toyexp.common.datasets import LieAlgebraRotationDataset
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
    """Create training dataset from config."""
    logger.info("Creating datasets...")

    # Get alpha velocities
    alpha_values = None
    if (
        hasattr(config.dataset, "alpha_values")
        and config.dataset.alpha_values is not None
    ):
        alpha_values = config.dataset.alpha_values
        # Parse string to list if needed (for command-line overrides)
        if isinstance(alpha_values, str):
            import ast

            try:
                alpha_values = ast.literal_eval(alpha_values)
            except (ValueError, SyntaxError) as e:
                logger.error(
                    f"Failed to parse alpha_values string '{alpha_values}': {e}"
                )
                raise ValueError(
                    f"alpha_values must be a valid list, got: {alpha_values}"
                )

    # Training dataset
    train_dataset = LieAlgebraRotationDataset(
        num_c_samples=config.dataset.num_c_train,
        rotation_dim=config.dataset.rotation_dim,
        num_rotations=config.dataset.num_rotations,
        c_min=config.dataset.c_min,
        c_max=config.dataset.c_max,
        alpha_values=alpha_values,
        alpha_min=config.dataset.get("alpha_min", 0.0),
        alpha_max=config.dataset.get("alpha_max", 2 * np.pi),
        weight_mode=config.dataset.weight_mode,
        num_weight_components=config.dataset.get("num_weight_components", 3),
        weight_strategy=config.dataset.get("weight_strategy", "uniform"),
        sampling_strategy=config.dataset.get("sampling_strategy", "grid"),
        rotation_seed=config.dataset.get("rotation_seed", 42),
        weight_seed=config.dataset.get("weight_seed", 43),
        sample_seed=config.dataset.get("sample_seed", 44),
    )

    logger.info(f"Training dataset: {len(train_dataset)} samples")
    logger.info(
        f"Rotation: SO({config.dataset.rotation_dim}), K={config.dataset.num_rotations} components"
    )
    logger.info(
        f"Output dimension: {config.dataset.num_rotations * config.dataset.rotation_dim}"
    )
    logger.info(f"Weight mode: {config.dataset.weight_mode}")
    logger.info(f"\n{train_dataset.get_function_description()}")

    return train_dataset


def train_epoch(model, dataloader, loss_manager, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        c = batch["c"].to(device)  # [batch, 1] containing c values
        x_1 = batch["x"].to(device)  # [batch, K*rotation_dim] concatenated output

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


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: Vector of shape [dim] or [..., dim]
        vec2: Vector of shape [dim] or [..., dim]

    Returns:
        Cosine similarity value
    """
    # Flatten if needed
    v1_flat = vec1.flatten()
    v2_flat = vec2.flatten()

    # Compute dot product and norms
    dot_product = np.dot(v1_flat, v2_flat)
    norm1 = np.linalg.norm(v1_flat)
    norm2 = np.linalg.norm(v2_flat)

    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def compute_perpendicular_projection_error(
    pred_vec: np.ndarray, manifold_vec: np.ndarray
) -> float:
    """
    Compute perpendicular projection error: ||(I - P_i) f_i|| / ||f_i||

    where P_i projects onto the span of the manifold vector.

    This measures what fraction of the prediction lies outside the
    manifold direction defined by the true rotation.

    Args:
        pred_vec: Predicted vector [rotation_dim]
        manifold_vec: True manifold vector [rotation_dim]

    Returns:
        Ratio of perpendicular component to total magnitude (0 to 1)
        - 0 means perfectly aligned (no perpendicular component)
        - 1 means completely perpendicular
    """
    # Flatten if needed
    v_p = pred_vec.flatten()
    v_m = manifold_vec.flatten()

    # Compute norms
    norm_pred = np.linalg.norm(v_p)
    norm_manifold = np.linalg.norm(v_m)

    # Avoid division by zero
    if norm_pred == 0 or norm_manifold == 0:
        return 0.0

    # Normalize manifold vector
    v_m_hat = v_m / norm_manifold

    # Project prediction onto manifold direction: P v_p = (v_m_hat Â· v_p) v_m_hat
    projection = np.dot(v_m_hat, v_p) * v_m_hat

    # Compute perpendicular component: (I - P) v_p = v_p - P v_p
    perpendicular = v_p - projection

    # Compute ratio
    perp_norm = np.linalg.norm(perpendicular)
    ratio = perp_norm / norm_pred

    return ratio


def evaluate(model, dataset, device, config):
    """Evaluate model on dataset with cosine similarity metrics for all NFE values."""
    model.eval()

    # Generate evaluation data
    eval_data = dataset.generate_eval_data(
        num_c_samples=config.dataset.num_c_eval,
        eval_seed=config.experiment.seed + 1000,
    )

    c_eval = eval_data["c"].to(device)  # [num_eval, 1] containing c values
    x_true = eval_data["x"].cpu().numpy()  # [num_eval, K*rotation_dim]

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

            # Compute standard metrics
            l1_error = np.mean(np.abs(x_pred - x_true))
            l2_error = np.sqrt(np.mean((x_pred - x_true) ** 2))

            metrics = {
                "nfe": nfe,
                "l1_error": l1_error,
                "l2_error": l2_error,
            }

            # Compute cosine similarity metrics
            c_vals_np = c_eval.cpu().numpy().flatten()
            if config.evaluation.get("compute_cosine_similarity", True):
                # Get manifold vectors
                manifold_vecs = dataset.get_manifold_vectors(
                    c_vals_np
                )  # [num_eval, K, rotation_dim]

                # Reshape predictions to [num_eval, K, rotation_dim]
                K = dataset.num_rotations
                rotation_dim = dataset.rotation_dim
                x_pred_reshaped = x_pred.reshape(-1, K, rotation_dim)

                # Compute cosine similarity for each component
                cos_similarities = []
                perp_errors = []
                for k in range(K):
                    # Get vectors for component k across all eval samples
                    manifold_k = manifold_vecs[:, k, :]  # [num_eval, rotation_dim]
                    pred_k = x_pred_reshaped[:, k, :]  # [num_eval, rotation_dim]

                    # Compute average cosine similarity for this component
                    cos_sims_k = []
                    perp_errs_k = []
                    for i in range(len(c_vals_np)):
                        cos_sim = compute_cosine_similarity(manifold_k[i], pred_k[i])
                        cos_sims_k.append(cos_sim)

                        perp_err = compute_perpendicular_projection_error(
                            pred_k[i], manifold_k[i]
                        )
                        perp_errs_k.append(perp_err)

                    avg_cos_sim_k = np.mean(cos_sims_k)
                    cos_similarities.append(avg_cos_sim_k)

                    avg_perp_err_k = np.mean(perp_errs_k)
                    perp_errors.append(avg_perp_err_k)

                # Add to metrics
                metrics["cos_similarities"] = cos_similarities  # List of K values
                metrics["avg_cos_similarity"] = np.mean(cos_similarities)
                metrics["min_cos_similarity"] = np.min(cos_similarities)

                # Compute absolute cosine similarity
                abs_cos_similarities = [abs(cs) for cs in cos_similarities]
                metrics["avg_abs_cos_similarity"] = np.mean(abs_cos_similarities)
                metrics["min_abs_cos_similarity"] = np.min(abs_cos_similarities)

                metrics["perp_errors"] = perp_errors  # List of K values
                metrics["avg_perp_error"] = np.mean(perp_errors)
                metrics["max_perp_error"] = np.max(perp_errors)

                # Log per-component cosine similarities
                logger.info(f"Per-component cosine similarities (NFE={nfe}):")
                for k, cos_sim in enumerate(cos_similarities):
                    alpha_k = dataset.alpha_velocities[k]
                    logger.info(f"  Component {k} (alpha={alpha_k:.4f}): {cos_sim:.6f}")

                # Log per-component perpendicular errors
                logger.info(f"Per-component perpendicular errors (NFE={nfe}):")
                for k, perp_err in enumerate(perp_errors):
                    alpha_k = dataset.alpha_velocities[k]
                    logger.info(
                        f"  Component {k} (alpha={alpha_k:.4f}): {perp_err:.6f}"
                    )

            # Prepare data for plotting - include ALL components
            K = dataset.num_rotations
            rotation_dim = dataset.rotation_dim

            # Reshape to [num_eval, K, rotation_dim] for easier access
            x_true_reshaped = x_true.reshape(-1, K, rotation_dim)
            x_pred_reshaped = x_pred.reshape(-1, K, rotation_dim)

            plot_data = {
                "nfe": nfe,
                "c_values": c_vals_np,
                "x_true_all": x_true_reshaped,  # [num_eval, K, rotation_dim]
                "x_pred_all": x_pred_reshaped,  # [num_eval, K, rotation_dim]
                "cos_similarities": metrics.get("cos_similarities", None),
                "alpha_velocities": (
                    dataset.alpha_velocities if "cos_similarities" in metrics else None
                ),
                "num_components": K,
                "rotation_dim": rotation_dim,
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
    # Note: condition_dim is 1 (just c, not [alpha, c])
    # Output dim is K * rotation_dim (concatenated)
    # Output dim is K * rotation_dim (concatenated)
    output_dim = config.dataset.num_rotations * config.dataset.rotation_dim

    model = create_model(
        architecture=config.network.architecture,
        x_dim=output_dim,
        c_dim=1,  # Just c
        output_dim=output_dim,  # K * rotation_dim
        hidden_dim=config.network.hidden_dim,
        n_layers=config.network.num_layers,
        activation=config.network.activation,
        use_time=(config.experiment.mode in ["flow", "mip"]),
    ).to(device)

    log_model_info(model)

    # Create loss manager
    loss_aggregation = config.training.get("loss_aggregation", "full")

    if loss_aggregation == "per_component":
        loss_manager = LossManager(
            mode=config.experiment.mode,
            loss_type=config.training.loss_type,
            x_dim=output_dim,
            loss_aggregation="per_component",
            num_components=config.dataset.num_rotations,
            component_dim=config.dataset.rotation_dim,
            mip_t_star=config.training.get("mip_t_star", 0.9),
        )
    else:
        loss_manager = LossManager(
            mode=config.experiment.mode,
            loss_type=config.training.loss_type,
            x_dim=output_dim,
            mip_t_star=config.training.get("mip_t_star", 0.9),
        )

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Create CSV metrics logger
    metrics_logger = create_metrics_logger(output_dir, experiment_type="lie")

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
                log_evaluation(metrics, prefix=f"Epoch {epoch + 1}")

                # Log to CSV
                eval_row = {
                    "epoch": epoch + 1,
                    "nfe": metrics["nfe"],
                    "l1_error": metrics["l1_error"],
                    "l2_error": metrics["l2_error"],
                }
                if "avg_cos_similarity" in metrics:
                    eval_row.update(
                        {
                            "avg_cos_similarity": metrics["avg_cos_similarity"],
                            "min_cos_similarity": metrics["min_cos_similarity"],
                            "avg_abs_cos_similarity": metrics["avg_abs_cos_similarity"],
                            "min_abs_cos_similarity": metrics["min_abs_cos_similarity"],
                        }
                    )

                if "avg_perp_error" in metrics:
                    eval_row.update(
                        {
                            "avg_perp_error": metrics["avg_perp_error"],
                            "max_perp_error": metrics["max_perp_error"],
                        }
                    )

                    # Log per-component metrics
                    for k, (cos_sim, perp_err) in enumerate(
                        zip(metrics["cos_similarities"], metrics["perp_errors"])
                    ):
                        metrics_logger.log(
                            "components",
                            {
                                "epoch": epoch + 1,
                                "nfe": metrics["nfe"],
                                "component": k,
                                "alpha": train_dataset.alpha_velocities[k],
                                "cos_similarity": cos_sim,
                                "abs_cos_similarity": abs(cos_sim),
                                "perp_error": perp_err,
                            },
                        )
                elif "cos_similarities" in metrics:
                    # Log per-component cosine similarities only
                    for k, cos_sim in enumerate(metrics["cos_similarities"]):
                        metrics_logger.log(
                            "components",
                            {
                                "epoch": epoch + 1,
                                "nfe": metrics["nfe"],
                                "component": k,
                                "alpha": train_dataset.alpha_velocities[k],
                                "cos_similarity": cos_sim,
                                "abs_cos_similarity": abs(cos_sim),
                            },
                        )

                metrics_logger.log("evaluation", eval_row)

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
        log_evaluation(metrics, prefix="Final Results")

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Training curves (only once, not per NFE)
    plot_training_curves(
        {"train_loss": train_losses},
        save_path=plots_dir / "training_curves.png",
        title=f"{config.experiment.name} - Training Curves",
    )

    # Create plots for EACH NFE
    for metrics, plot_data in results:
        nfe = metrics["nfe"]

        # =========================================================================
        # Grid Visualizations - All Components
        # =========================================================================
        logger.info(f"Creating grid visualizations for all components (NFE={nfe})...")

        K = plot_data["num_components"]
        rotation_dim = plot_data["rotation_dim"]
        c_values = plot_data["c_values"]
        x_true_all = plot_data["x_true_all"]  # [num_eval, K, rotation_dim]
        x_pred_all = plot_data["x_pred_all"]  # [num_eval, K, rotation_dim]

        if K <= 12:  # Only create grids if reasonable number of components
            n_cols = min(4, K)
            n_rows = (K + n_cols - 1) // n_cols

            # 1. Predictions Grid
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if K == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            for k in range(K):
                row = k // n_cols
                col = k % n_cols
                ax = axes[row, col]

                alpha_k = train_dataset.alpha_velocities[k]
                true_vals = x_true_all[:, k, 0]  # First element
                pred_vals = x_pred_all[:, k, 0]

                ax.scatter(
                    c_values, true_vals, alpha=0.6, s=10, label="True", color="blue"
                )
                ax.scatter(
                    c_values, pred_vals, alpha=0.6, s=10, label="Pred", color="red"
                )
                ax.set_xlabel("c", fontsize=9)
                ax.set_ylabel(f"f_{k}[0]", fontsize=9)
                ax.set_title(rf"Comp {k} ($\alpha$={alpha_k:.3f})", fontsize=10)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for k in range(K, n_rows * n_cols):
                row = k // n_cols
                col = k % n_cols
                axes[row, col].axis("off")

            plt.tight_layout()
            save_path = plots_dir / f"predictions_all_components_grid_nfe{nfe}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved predictions grid to {save_path}")
            plt.close(fig)

            # 2. L1 Errors Grid
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if K == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            for k in range(K):
                row = k // n_cols
                col = k % n_cols
                ax = axes[row, col]

                alpha_k = train_dataset.alpha_velocities[k]
                true_vals = x_true_all[:, k, 0]
                pred_vals = x_pred_all[:, k, 0]
                l1_errors = np.abs(pred_vals - true_vals)

                ax.scatter(c_values, l1_errors, alpha=0.6, s=10, color="red")
                mean_error = np.mean(l1_errors)
                ax.axhline(y=mean_error, color="black", linestyle="--", linewidth=1)
                ax.set_xlabel("c", fontsize=9)
                ax.set_ylabel("L1 Error", fontsize=9)
                ax.set_title(
                    rf"Comp {k} ($\alpha$={alpha_k:.3f}), Mean={mean_error:.4f}",
                    fontsize=10,
                )
                ax.grid(True, alpha=0.3)

            for k in range(K, n_rows * n_cols):
                row = k // n_cols
                col = k % n_cols
                axes[row, col].axis("off")

            plt.tight_layout()
            save_path = plots_dir / f"l1_errors_all_components_grid_nfe{nfe}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved L1 errors grid to {save_path}")
            plt.close(fig)

            # 3. L2 Errors Grid
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if K == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            for k in range(K):
                row = k // n_cols
                col = k % n_cols
                ax = axes[row, col]

                alpha_k = train_dataset.alpha_velocities[k]
                true_vals = x_true_all[:, k, 0]
                pred_vals = x_pred_all[:, k, 0]
                l2_errors = (pred_vals - true_vals) ** 2

                ax.scatter(c_values, l2_errors, alpha=0.6, s=10, color="orange")
                mean_error = np.mean(l2_errors)
                ax.axhline(y=mean_error, color="black", linestyle="--", linewidth=1)
                ax.set_xlabel("c", fontsize=9)
                ax.set_ylabel("L2 Error (squared)", fontsize=9)
                ax.set_title(
                    rf"Comp {k} ($\alpha$={alpha_k:.3f}), Mean={mean_error:.4f}",
                    fontsize=10,
                )
                ax.grid(True, alpha=0.3)

            for k in range(K, n_rows * n_cols):
                row = k // n_cols
                col = k % n_cols
                axes[row, col].axis("off")

            plt.tight_layout()
            save_path = plots_dir / f"l2_errors_all_components_grid_nfe{nfe}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved L2 errors grid to {save_path}")
            plt.close(fig)

        # =========================================================================
        # Cosine Similarity Plots (if computed)
        # =========================================================================
        if plot_data["cos_similarities"] is not None:
            # Summary plot - cosine similarity vs component
            fig, ax = plt.subplots(figsize=(10, 6))

            component_indices = list(range(len(plot_data["cos_similarities"])))
            ax.plot(
                component_indices,
                plot_data["cos_similarities"],
                marker="o",
                linewidth=2,
                markersize=8,
            )

            ax.set_xlabel("Component Index i", fontsize=12)
            ax.set_ylabel("Mean Cosine Similarity", fontsize=12)
            ax.set_title(
                f"{config.experiment.name} - Mean Cosine Similarity vs Component (NFE={nfe})",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)

            # Add alpha values as secondary x-axis labels
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(component_indices)
            alpha_labels = [f"{alpha:.2f}" for alpha in plot_data["alpha_velocities"]]
            ax2.set_xticklabels(alpha_labels, fontsize=9)
            ax2.set_xlabel(r"$\alpha$ velocity", fontsize=10)

            plt.tight_layout()
            save_path = plots_dir / f"cosine_similarity_vs_component_nfe{nfe}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved cosine similarity summary to {save_path}")
            plt.close(fig)

            # Grid plot - cosine similarity per component
            if K <= 12:
                fig, axes = plt.subplots(
                    n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows)
                )
                if K == 1:
                    axes = np.array([[axes]])
                elif n_rows == 1:
                    axes = axes.reshape(1, -1)
                elif n_cols == 1:
                    axes = axes.reshape(-1, 1)

                # Get per-sample cosine similarities for each component
                c_vals_np = c_values
                manifold_vecs = train_dataset.get_manifold_vectors(c_vals_np)
                x_pred_reshaped = x_pred_all

                for k in range(K):
                    row = k // n_cols
                    col = k % n_cols
                    ax = axes[row, col]

                    alpha_k = train_dataset.alpha_velocities[k]

                    # Compute cosine similarity for each sample
                    cos_sims = []
                    for i in range(len(c_vals_np)):
                        cos_sim = compute_cosine_similarity(
                            x_pred_reshaped[i, k, :], manifold_vecs[i, k, :]
                        )
                        cos_sims.append(cos_sim)

                    ax.scatter(c_vals_np, cos_sims, alpha=0.6, s=10, color="green")
                    mean_cos = np.mean(cos_sims)
                    ax.axhline(y=mean_cos, color="black", linestyle="--", linewidth=1)
                    ax.set_xlabel("c", fontsize=9)
                    ax.set_ylabel("Cosine Similarity", fontsize=9)
                    ax.set_title(
                        rf"Comp {k} ($\alpha$={alpha_k:.3f}), Mean={mean_cos:.4f}",
                        fontsize=10,
                    )
                    ax.grid(True, alpha=0.3)

                for k in range(K, n_rows * n_cols):
                    row = k // n_cols
                    col = k % n_cols
                    axes[row, col].axis("off")

                plt.tight_layout()
                save_path = (
                    plots_dir / f"cosine_similarity_all_components_grid_nfe{nfe}.png"
                )
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info(f"Saved cosine similarity grid to {save_path}")
                plt.close(fig)

        # =========================================================================
        # Perpendicular Error Plots (if computed)
        # =========================================================================
        if "perp_errors" in metrics:
            # Summary plot - perpendicular error vs component
            fig, ax = plt.subplots(figsize=(10, 6))

            component_indices = list(range(len(metrics["perp_errors"])))
            ax.plot(
                component_indices,
                metrics["perp_errors"],
                marker="o",
                linewidth=2,
                markersize=8,
                color="purple",
            )

            ax.set_xlabel("Component Index i", fontsize=12)
            ax.set_ylabel("Mean Perpendicular Error", fontsize=12)
            ax.set_title(
                f"{config.experiment.name} - Mean Perpendicular Error vs Component (NFE={nfe})",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)

            # Add alpha values as secondary x-axis labels
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(component_indices)
            alpha_labels = [f"{alpha:.2f}" for alpha in plot_data["alpha_velocities"]]
            ax2.set_xticklabels(alpha_labels, fontsize=9)
            ax2.set_xlabel(r"$\alpha$ velocity", fontsize=10)

            plt.tight_layout()
            save_path = plots_dir / f"perpendicular_error_vs_component_nfe{nfe}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved perpendicular error summary to {save_path}")
            plt.close(fig)

            # Grid plot - perpendicular error per component
            if K <= 12:
                fig, axes = plt.subplots(
                    n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows)
                )
                if K == 1:
                    axes = np.array([[axes]])
                elif n_rows == 1:
                    axes = axes.reshape(1, -1)
                elif n_cols == 1:
                    axes = axes.reshape(-1, 1)

                # Get per-sample perpendicular errors for each component
                c_vals_np = c_values
                manifold_vecs = train_dataset.get_manifold_vectors(c_vals_np)
                x_pred_reshaped = x_pred_all

                for k in range(K):
                    row = k // n_cols
                    col = k % n_cols
                    ax = axes[row, col]

                    alpha_k = train_dataset.alpha_velocities[k]

                    # Compute perpendicular error for each sample
                    perp_errs = []
                    for i in range(len(c_vals_np)):
                        perp_err = compute_perpendicular_projection_error(
                            x_pred_reshaped[i, k, :], manifold_vecs[i, k, :]
                        )
                        perp_errs.append(perp_err)

                    ax.scatter(c_vals_np, perp_errs, alpha=0.6, s=10, color="purple")
                    mean_perp = np.mean(perp_errs)
                    ax.axhline(y=mean_perp, color="black", linestyle="--", linewidth=1)
                    ax.set_xlabel("c", fontsize=9)
                    ax.set_ylabel("Perpendicular Error", fontsize=9)
                    ax.set_title(
                        rf"Comp {k} ($\alpha$={alpha_k:.3f}), Mean={mean_perp:.4f}",
                        fontsize=10,
                    )
                    ax.grid(True, alpha=0.3)

                for k in range(K, n_rows * n_cols):
                    row = k // n_cols
                    col = k % n_cols
                    axes[row, col].axis("off")

                plt.tight_layout()
                save_path = (
                    plots_dir / f"perpendicular_error_all_components_grid_nfe{nfe}.png"
                )
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info(f"Saved perpendicular error grid to {save_path}")
                plt.close(fig)

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

    return first_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Lie algebra rotation experiment (REDESIGNED)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python train_lie.py --config config_lie.yaml
  
  # Override config values using full key paths
  python train_lie.py --config config_lie.yaml \\
      experiment.mode=flow \\
      experiment.seed=123 \\
      training.learning_rate=0.001 \\
      dataset.weight_mode=high_frequency
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_lie.yaml",
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
