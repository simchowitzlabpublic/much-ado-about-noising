"""
Training script for reconstruction experiment.

Compare regression vs flow models on scalar target functions f(c).
"""

import argparse
import logging
from pathlib import Path

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
from toyexp.common.datasets import TargetFunctionDataset
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
    train_dataset = TargetFunctionDataset(
        num_samples=config.dataset.num_train,
        target_dim=config.dataset.target_dim,
        condition_dim=config.dataset.condition_dim,
        num_components=config.dataset.num_components,
        c_min=config.dataset.c_min,
        c_max=config.dataset.c_max,
        weight_strategy=config.dataset.weight_strategy,
        sampling_strategy=config.dataset.sampling_strategy,
        freq_seed=config.dataset.freq_seed,
        phase_seed=config.dataset.phase_seed,
        weight_seed=config.dataset.weight_seed,
        sample_seed=config.dataset.sample_seed,
    )

    logger.info(f"Training dataset: {len(train_dataset)} samples")

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

            metrics = {
                "nfe": nfe,
                "l1_error": l1_error,
                "l2_error": l2_error,
            }

            # Prepare data for plotting
            plot_data = {
                "nfe": nfe,
                "c_values": c_eval.cpu().numpy().flatten(),
                "true_values": x_true.flatten(),
                "pred_values": x_pred.flatten(),
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
    metrics_logger = create_metrics_logger(output_dir, experiment_type="recon")

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
        log_evaluation(metrics, prefix="Final")

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

        # Predictions
        plot_predictions(
            plot_data["c_values"],
            plot_data["true_values"],
            plot_data["pred_values"],
            save_path=plots_dir / f"predictions_nfe{nfe}.png",
            title=f"{config.experiment.name} - Predictions",
            nfe=nfe,
        )

        # Errors
        errors = np.abs(plot_data["pred_values"] - plot_data["true_values"])
        plot_errors(
            plot_data["c_values"],
            errors,
            save_path=plots_dir / f"errors_nfe{nfe}.png",
            title=f"{config.experiment.name} - Prediction Errors",
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

    return first_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train reconstruction experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python train_recon.py --config config_recon.yaml
  
  # Override config values using full key paths
  python train_recon.py --config config_recon.yaml \\
      experiment.mode=flow \\
      experiment.seed=123 \\
      training.learning_rate=0.001 \\
      dataset.num_train=100
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_recon.yaml",
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
