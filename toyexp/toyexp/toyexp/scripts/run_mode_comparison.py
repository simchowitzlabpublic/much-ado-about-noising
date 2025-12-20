"""
Batch wrapper script for running mode comparison experiments.

This script automates running training across different modes (regression/flow/mip)
and loss types (l1/l2), then generates a LaTeX table comparing the results.

Usage:
    python -m toyexp.scripts.run_mode_comparison --experiment recon --config toyexp/configs/config_recon.yaml
    python -m toyexp.scripts.run_mode_comparison --experiment proj --config toyexp/configs/config_proj.yaml
    python -m toyexp.scripts.run_mode_comparison --experiment lie --config toyexp/configs/config_lie.yaml
"""

import argparse
import importlib
import logging
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

# Experiment configurations
EXPERIMENT_CONFIGS = {
    "recon": {
        "module": "toyexp.scripts.train_recon",
        "modes": [
            "straight_flow",
            "mip_one_step_integrate",
            "regression",
            "flow",
            "mip",
        ],
        "loss_types": ["l1", "l2"],
        "metrics": ["L1", "L2"],
    },
    "proj": {
        "module": "toyexp.scripts.train_proj",
        "modes": [
            "straight_flow",
            "mip_one_step_integrate",
            "regression",
            "flow",
            "mip",
        ],
        "loss_types": ["l1", "l2"],
        "metrics": ["L1", "L2", "Subspace Diag", "Subspace Off-Diag", "Boundary"],
    },
    "lie": {
        "module": "toyexp.scripts.train_lie",
        "modes": [
            "straight_flow",
            "mip_one_step_integrate",
            "regression",
            "flow",
            "mip",
        ],
        "loss_types": ["l1", "l2"],
        "metrics": [
            "L1",
            "L2",
            "Avg Cos Sim",
            "Min Cos Sim",
            "Avg Perp Error",
            "Max Perp Error",
        ],
    },
}

# Seeds for multi-seed runs
SEEDS = [0, 1, 2]

# Architectures to compare
ARCHITECTURES = ["concat", "film"]


def setup_logging():
    """Setup logging for the batch wrapper."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def run_training(
    module_name: str,
    config_path: str,
    mode: str,
    loss_type: str,
    architecture: str,
    seed: int,
    logger,
):
    """
    Run a single training job with specified mode, loss_type, architecture, and seed.

    Args:
        module_name: Full module name (e.g., 'toyexp.train_recon')
        config_path: Path to config file
        mode: Training mode (regression/flow/mip)
        loss_type: Loss type (l1/l2)
        architecture: Network architecture (concat/film)
        seed: Random seed
        logger: Logger instance

    Returns:
        bool: True if training completed successfully
    """
    logger.info("=" * 80)
    logger.info(
        f"Starting training: mode={mode}, loss_type={loss_type}, architecture={architecture}, seed={seed}"
    )
    logger.info("=" * 80)

    try:
        # Import the training module
        train_module = importlib.import_module(module_name)

        # Build overrides dict
        overrides = {
            "experiment": {"mode": mode, "seed": seed},
            "training": {"loss_type": loss_type},
            "network": {"architecture": architecture},
        }

        logger.info(f"Module: {module_name}")
        logger.info(f"Config: {config_path}")
        logger.info(f"Overrides: {overrides}")

        # Call the main function with overrides
        train_module.main(config_path, overrides)

        logger.info(
            f"Training completed successfully: mode={mode}, loss_type={loss_type}, architecture={architecture}, seed={seed}"
        )
        return True

    except Exception as e:
        logger.error(
            f"Training failed: mode={mode}, loss_type={loss_type}, architecture={architecture}, seed={seed}"
        )
        logger.error(f"Error: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def load_results(
    output_dir: Path, mode: str, loss_type: str, architecture: str, seed: int, logger
):
    """
    Load results from a completed training run.

    Args:
        output_dir: Base output directory
        mode: Training mode
        loss_type: Loss type
        architecture: Network architecture
        seed: Random seed
        logger: Logger instance

    Returns:
        dict: Metrics from best checkpoint, or None if not found
    """
    # Results are saved in: output_dir / mode / loss_type / architecture / seed_X / checkpoints / final_model.pt
    results_path = (
        output_dir
        / mode
        / loss_type
        / architecture
        / f"seed_{seed}"
        / "checkpoints"
        / "final_model.pt"
    )

    if not results_path.exists():
        logger.warning(f"Results not found: {results_path}")
        return None

    try:
        checkpoint = torch.load(results_path, map_location="cpu", weights_only=False)
        metrics = checkpoint.get("metrics", {})

        logger.info(f"Loaded results from {results_path}")
        logger.info(f"  L1: {metrics.get('l1_error', np.nan):.6f}")
        logger.info(f"  L2: {metrics.get('l2_error', np.nan):.6f}")

        return metrics

    except Exception as e:
        logger.error(f"Failed to load results from {results_path}: {e}")
        return None


def generate_latex_table(
    results: dict,
    experiment_type: str,
    save_dir: Path,
    logger,
):
    """
    Generate LaTeX tables comparing results across modes, loss types, architectures, and seeds.
    Creates two tables: one with individual seed results and one with averaged results.

    Args:
        results: Dict mapping (mode, loss_type, architecture) -> list of metrics dicts (one per seed)
        experiment_type: Type of experiment (recon/proj/lie)
        save_dir: Directory where to save the LaTeX tables
        logger: Logger instance
    """
    logger.info("=" * 80)
    logger.info("Generating LaTeX tables")
    logger.info("=" * 80)

    exp_config = EXPERIMENT_CONFIGS[experiment_type]
    modes = exp_config["modes"]
    loss_types = exp_config["loss_types"]
    metric_names = exp_config["metrics"]

    # Helper function to extract metric values based on experiment type
    def extract_metrics(metrics_dict):
        """Extract metrics in order based on experiment type."""
        if metrics_dict is None:
            return [np.nan] * len(metric_names)

        if experiment_type == "recon":
            # Recon: L1, L2 only
            return [
                metrics_dict.get("l1_error", np.nan),
                metrics_dict.get("l2_error", np.nan),
            ]
        elif experiment_type == "proj":
            # Proj: L1, L2, Subspace Diag, Subspace Off-Diag, Boundary
            return [
                metrics_dict.get("l1_error", np.nan),
                metrics_dict.get("l2_error", np.nan),
                metrics_dict.get("subspace_diagonal_mean_4", np.nan),
                metrics_dict.get("subspace_off_diagonal_mean_4", np.nan),
                metrics_dict.get("boundary_mean_2", np.nan),
            ]
        elif experiment_type == "lie":
            # Complex: L1, L2, Avg Cos Sim, Min Cos Sim, Avg Perp Error, Max Perp Error
            return [
                metrics_dict.get("l1_error", np.nan),
                metrics_dict.get("l2_error", np.nan),
                metrics_dict.get("avg_cos_similarity", np.nan),
                metrics_dict.get("min_cos_similarity", np.nan),
                metrics_dict.get("avg_perp_error", np.nan),
                metrics_dict.get("max_perp_error", np.nan),
            ]

    # --- TABLE 1: SEED-WISE RESULTS ---
    logger.info("Generating seed-wise table...")
    lines_seedwise = []
    lines_seedwise.append("\\begin{table}[h]")
    lines_seedwise.append("\\centering")
    lines_seedwise.append("\\begin{tabular}{|l|l|l|c|" + "c|" * len(metric_names) + "}")
    lines_seedwise.append("\\hline")

    # Header row
    header = "Mode & Loss & Arch & Seed & " + " & ".join(metric_names) + " \\\\"
    lines_seedwise.append(header)
    lines_seedwise.append("\\hline")

    # Data rows - iterate through each mode, loss_type, architecture, and seed
    for mode in modes:
        for loss_type in loss_types:
            for architecture in ARCHITECTURES:
                key = (mode, loss_type, architecture)

                if key not in results or results[key] is None:
                    # Missing results for all seeds - fill with dashes
                    for seed in SEEDS:
                        values = ["---"] * len(metric_names)
                        row = (
                            f"{mode} & {loss_type} & {architecture} & {seed} & "
                            + " & ".join(values)
                            + " \\\\"
                        )
                        lines_seedwise.append(row)
                else:
                    # Have results - show each seed
                    seed_metrics_list = results[key]
                    for seed_idx, seed in enumerate(SEEDS):
                        if (
                            seed_idx < len(seed_metrics_list)
                            and seed_metrics_list[seed_idx] is not None
                        ):
                            metric_values = extract_metrics(seed_metrics_list[seed_idx])
                            values = [
                                f"{v:.6f}" if not np.isnan(v) else "---"
                                for v in metric_values
                            ]
                        else:
                            values = ["---"] * len(metric_names)

                        row = (
                            f"{mode} & {loss_type} & {architecture} & {seed} & "
                            + " & ".join(values)
                            + " \\\\"
                        )
                        lines_seedwise.append(row)

                lines_seedwise.append("\\hline")

    # Close table
    lines_seedwise.append("\\end{tabular}")
    lines_seedwise.append(
        f"\\caption{{Seed-wise results for {experiment_type} experiment across modes, loss types, and architectures}}"
    )
    lines_seedwise.append(f"\\label{{tab:{experiment_type}_seedwise}}")
    lines_seedwise.append("\\end{table}")

    # Save seed-wise table
    seedwise_path = save_dir / "results_table_seedwise.tex"
    latex_seedwise = "\n".join(lines_seedwise)
    seedwise_path.write_text(latex_seedwise)
    logger.info(f"Saved seed-wise LaTeX table to {seedwise_path}")

    # --- TABLE 2: AVERAGED RESULTS ---
    logger.info("Generating averaged table...")
    lines_avg = []
    lines_avg.append("\\begin{table}[h]")
    lines_avg.append("\\centering")
    lines_avg.append("\\begin{tabular}{|l|l|l|" + "c|" * len(metric_names) + "}")
    lines_avg.append("\\hline")

    # Header row
    header = "Mode & Loss & Arch & " + " & ".join(metric_names) + " \\\\"
    lines_avg.append(header)
    lines_avg.append("\\hline")

    # Data rows - average across seeds
    for mode in modes:
        for loss_type in loss_types:
            for architecture in ARCHITECTURES:
                key = (mode, loss_type, architecture)

                if key not in results or results[key] is None:
                    # Missing results
                    values = ["---"] * len(metric_names)
                else:
                    seed_metrics_list = results[key]

                    # Collect metrics across seeds
                    all_metrics = []
                    for seed_metrics in seed_metrics_list:
                        if seed_metrics is not None:
                            all_metrics.append(extract_metrics(seed_metrics))

                    if len(all_metrics) == 0:
                        values = ["---"] * len(metric_names)
                    else:
                        # Convert to numpy array for easy computation
                        all_metrics = np.array(
                            all_metrics
                        )  # shape: (num_seeds, num_metrics)

                        # Compute mean and std for each metric
                        means = np.nanmean(all_metrics, axis=0)
                        stds = np.nanstd(all_metrics, axis=0)

                        # Format as "mean ± std"
                        values = []
                        for mean, std in zip(means, stds):
                            if np.isnan(mean):
                                values.append("---")
                            else:
                                values.append(f"{mean:.6f} $\\pm$ {std:.6f}")

                # Format row
                row = (
                    f"{mode} & {loss_type} & {architecture} & "
                    + " & ".join(values)
                    + " \\\\"
                )
                lines_avg.append(row)
                lines_avg.append("\\hline")

    # Close table
    lines_avg.append("\\end{tabular}")
    lines_avg.append(
        f"\\caption{{Averaged results (mean $\\pm$ std) for {experiment_type} experiment across modes, loss types, and architectures}}"
    )
    lines_avg.append(f"\\label{{tab:{experiment_type}_averaged}}")
    lines_avg.append("\\end{table}")

    # Save averaged table
    avg_path = save_dir / "results_table_averaged.tex"
    latex_avg = "\n".join(lines_avg)
    avg_path.write_text(latex_avg)
    logger.info(f"Saved averaged LaTeX table to {avg_path}")

    logger.info("\nSeed-wise table preview:")
    logger.info(
        latex_seedwise[:1000] + "..." if len(latex_seedwise) > 1000 else latex_seedwise
    )
    logger.info("\nAveraged table preview:")
    logger.info(latex_avg)


def main():
    """Main function for batch mode comparison."""
    parser = argparse.ArgumentParser(
        description="Run mode comparison experiments and generate LaTeX tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run reconstruction experiment
  python -m toyexp.scripts.run_mode_comparison --experiment recon --config toyexp/configs/config_recon.yaml
  
  # Run projection experiment
  python -m toyexp.scripts.run_mode_comparison --experiment proj --config toyexp/configs/config_proj.yaml
  
  # Run Lie experiment
  python -m toyexp.scripts.run_mode_comparison --experiment lie --config toyexp/configs/config_lie.yaml
  
  # Skip training and only generate table from existing results
  python -m toyexp.scripts.run_mode_comparison --experiment recon --config toyexp/configs/config_recon.yaml --skip-training
        """,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["recon", "proj", "lie"],
        help="Type of experiment to run",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base config file",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only generate table from existing results",
    )

    args = parser.parse_args()

    # Setup
    logger = setup_logging()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # Get experiment configuration
    exp_config = EXPERIMENT_CONFIGS[args.experiment]
    module_name = exp_config["module"]

    logger.info("=" * 80)
    logger.info(f"Batch Mode Comparison: {args.experiment.upper()}")
    logger.info("=" * 80)
    logger.info(f"Config: {config_path}")
    logger.info(f"Module: {module_name}")
    logger.info(f"Modes: {exp_config['modes']}")
    logger.info(f"Loss types: {exp_config['loss_types']}")
    logger.info("")

    # Load base config to get output directory
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    base_output_dir = Path(base_config["experiment"]["output_dir"])

    logger.info(f"Output directory: {base_output_dir}")
    logger.info("")

    # Run training for each mode/loss_type/architecture/seed combination
    if not args.skip_training:
        success_count = 0
        total_count = (
            len(exp_config["modes"])
            * len(exp_config["loss_types"])
            * len(ARCHITECTURES)
            * len(SEEDS)
        )

        for mode in exp_config["modes"]:
            for loss_type in exp_config["loss_types"]:
                for architecture in ARCHITECTURES:
                    for seed in SEEDS:
                        success = run_training(
                            module_name,
                            str(config_path),
                            mode,
                            loss_type,
                            architecture,
                            seed,
                            logger,
                        )
                        if success:
                            success_count += 1
                        logger.info("")

        logger.info("=" * 80)
        logger.info(
            f"Training Summary: {success_count}/{total_count} completed successfully"
        )
        logger.info("=" * 80)
        logger.info("")
    else:
        logger.info("Skipping training (--skip-training flag set)")
        logger.info("")

    # Load all results (all seeds for each mode/loss_type/architecture)
    logger.info("=" * 80)
    logger.info("Loading results")
    logger.info("=" * 80)

    results = {}
    for mode in exp_config["modes"]:
        for loss_type in exp_config["loss_types"]:
            for architecture in ARCHITECTURES:
                # Collect metrics for all seeds
                seed_metrics_list = []
                for seed in SEEDS:
                    metrics = load_results(
                        base_output_dir, mode, loss_type, architecture, seed, logger
                    )
                    seed_metrics_list.append(metrics)
                    logger.info("")

                results[(mode, loss_type, architecture)] = seed_metrics_list

    # Generate LaTeX tables
    generate_latex_table(results, args.experiment, base_output_dir, logger)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Batch comparison complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
