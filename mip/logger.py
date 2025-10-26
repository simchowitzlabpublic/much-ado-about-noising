"""Logger for training and evaluation.
Port from https://github.com/CleanDiffuserTeam/CleanDiffuser.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import contextlib
import json
import os
import uuid
from pathlib import Path

import loguru
import torch
import wandb
from omegaconf import OmegaConf

from mip.config import Config
from mip.env_utils import VideoRecordingWrapper


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    with contextlib.suppress(OSError):
        os.makedirs(dir_path)
    return dir_path


class Logger:
    """Primary logger object. Logs in wandb."""

    def __init__(self, config: Config):
        self.config = config.log
        self._log_dir = Path(make_dir(config.log.log_dir))
        self._model_dir = Path(make_dir(str(self._log_dir / "models")))
        self._video_dir = Path(make_dir(str(self._log_dir / "videos")))
        # Global checkpoints directory (one level up from log_dir)
        self._global_checkpoints_dir = Path(make_dir("checkpoints"))

        # Convert full config dataclass to OmegaConf config, then to container
        # This uploads all config (optimization, network, task, log) to wandb
        omega_config = OmegaConf.structured(config)

        wandb.init(
            config=OmegaConf.to_container(omega_config),
            project=config.log.project,
            group=config.log.group,
            name=config.log.exp_name,
            id=str(uuid.uuid4()),
            mode=config.log.wandb_mode,
            dir=self._log_dir,
        )
        self._wandb = wandb

    def video_init(self, env, enable=False, video_id=""):
        """Initialize video recording for an environment.

        Args:
            env: The environment (potentially wrapped with VideoRecordingWrapper)
            enable: Whether to enable video recording
            video_id: Identifier for the video file
        """
        # Check if env has VideoRecordingWrapper
        video_env = env.env if isinstance(env.env, VideoRecordingWrapper) else env

        # Only proceed if the environment has video recording capability
        if not isinstance(video_env, VideoRecordingWrapper):
            return

        if enable:
            video_env.video_recoder.stop()
            video_filename = self._video_dir / f"{video_id}_{uuid.uuid4()}.mp4"
            video_env.file_path = str(video_filename)
        else:
            video_env.file_path = None

    def log(self, d, category):
        assert category in ["train", "eval"]
        assert "step" in d

        # Print metrics, but skip wandb.Image objects for console output
        printable_items = {
            k: v for k, v in d.items() if not isinstance(v, self._wandb.Image)
        }
        # convert torch to float
        printable_items = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in printable_items.items()
        }
        metrics_str = f"[Step {d['step']}] " + " | ".join(
            [f"{k}: {v:.2e}" for k, v in printable_items.items() if k != "step"]
        )
        loguru.logger.info(metrics_str)

        # For JSON logging, create a copy without wandb.Image objects
        json_safe_dict = {"step": d["step"]}
        for k, v in d.items():
            if not isinstance(v, self._wandb.Image):
                json_safe_dict[k] = v

        # Write to metrics file
        with (self._log_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(json_safe_dict) + "\n")

        # Prepare wandb logging dict with category prefix
        _d = {}
        for k, v in d.items():
            _d[category + "/" + k] = v

        # Log to wandb
        self._wandb.log(_d, step=d["step"])

    def save_agent(self, agent=None, identifier="final"):
        if agent:
            fp = self._model_dir / f"model_{str(identifier)}.pt"
            agent.save(fp)
            loguru.logger.info(f"model_{str(identifier)} saved")

    def save_global_checkpoint(
        self, agent, checkpoint_name, success_rate, training_state=None
    ):
        """Save checkpoint to global checkpoints folder with success rate comparison.

        Args:
            agent: The agent to save
            checkpoint_name: Base name without success rate (e.g., "lift_ph_state_flow_chiunet_128_seed0")
            success_rate: Current success rate (0.0 to 1.0)
            training_state: Optional dict with training state (n_gradient_step, best_metrics, eval_history)
        """
        if not agent:
            return

        success_rate_pct = int(success_rate * 100)
        new_checkpoint_name = f"{checkpoint_name}_success{success_rate_pct}"
        new_checkpoint_path = self._global_checkpoints_dir / f"{new_checkpoint_name}.pt"

        # Find existing checkpoints with the same base name but different success rates
        existing_checkpoints = list(
            self._global_checkpoints_dir.glob(f"{checkpoint_name}_success*.pt")
        )

        should_save = True
        if existing_checkpoints:
            # Extract success rates from existing checkpoints
            for existing_checkpoint in existing_checkpoints:
                try:
                    # Extract success rate from filename (e.g., "..._success85.pt")
                    existing_name = existing_checkpoint.stem
                    existing_success_str = existing_name.split("_success")[-1]
                    existing_success_pct = int(existing_success_str)

                    if success_rate_pct > existing_success_pct:
                        # New checkpoint is better, delete old one
                        loguru.logger.info(
                            f"New checkpoint has better success rate ({success_rate_pct}% > {existing_success_pct}%), "
                            f"removing old checkpoint: {existing_checkpoint.name}"
                        )
                        existing_checkpoint.unlink()
                    elif success_rate_pct <= existing_success_pct:
                        # Existing checkpoint is better or equal, don't save
                        loguru.logger.info(
                            f"Existing checkpoint has better or equal success rate ({existing_success_pct}% >= {success_rate_pct}%), "
                            f"skipping save"
                        )
                        should_save = False
                        break
                except (ValueError, IndexError) as e:
                    loguru.logger.warning(
                        f"Could not parse success rate from {existing_checkpoint.name}: {e}"
                    )

        if should_save:
            agent.save(new_checkpoint_path, training_state=training_state)
            loguru.logger.info(f"Global checkpoint saved: {new_checkpoint_path}")

        return should_save

    def find_latest_checkpoint(self, checkpoint_base_name):
        """Find the latest checkpoint matching the base name pattern.

        Args:
            checkpoint_base_name: Base name without success rate (e.g., "lift_ph_state_flow_chiunet_128_seed0")

        Returns:
            Path to the best checkpoint, or None if not found
        """
        # Find all checkpoints matching the pattern
        matching_checkpoints = list(
            self._global_checkpoints_dir.glob(f"{checkpoint_base_name}_success*.pt")
        )

        if not matching_checkpoints:
            loguru.logger.info(
                f"No existing checkpoints found for pattern: {checkpoint_base_name}"
            )
            return None

        # Find the checkpoint with the highest success rate
        best_checkpoint = None
        best_success_rate = -1

        for checkpoint_path in matching_checkpoints:
            try:
                # Extract success rate from filename
                checkpoint_name = checkpoint_path.stem
                success_str = checkpoint_name.split("_success")[-1]
                success_rate = int(success_str)

                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_checkpoint = checkpoint_path
            except (ValueError, IndexError) as e:
                loguru.logger.warning(
                    f"Could not parse success rate from {checkpoint_path.name}: {e}"
                )

        if best_checkpoint:
            loguru.logger.info(
                f"Found checkpoint: {best_checkpoint.name} with success rate: {best_success_rate}%"
            )

        return best_checkpoint

    def finish(self, agent):
        try:
            self.save_agent(agent)
        except Exception as e:
            loguru.logger.error(f"Failed to save model: {e}")
        if self._wandb:
            self._wandb.finish()


def update_best_metrics(best_metrics, current_metrics):
    """Update best metrics with current evaluation results.

    Args:
        best_metrics: Dictionary storing best metrics so far
        current_metrics: Dictionary with current evaluation metrics

    Returns:
        updated best_metrics dictionary
    """
    for key, value in current_metrics.items():
        if key.startswith(
            ("mean_success_", "mean_reward_", "p4_")
        ) and not key.startswith("val_"):
            if key not in best_metrics or value > best_metrics[key]:
                best_metrics[key] = value
        elif key.startswith("mean_step_") and not key.startswith("val_"):
            # For step count, lower is better
            if key not in best_metrics or value < best_metrics[key]:
                best_metrics[key] = value
    return best_metrics


def compute_average_metrics(eval_history):
    """Compute average metrics from the last 5 evaluation results.

    Args:
        eval_history: List of evaluation metric dictionaries

    Returns:
        Dictionary with average metrics
    """
    if not eval_history:
        return {}

    # Take last 5 evaluations
    recent_evals = eval_history[-5:]

    # Get all unique metric keys
    all_keys = set()
    for eval_result in recent_evals:
        all_keys.update(eval_result.keys())

    avg_metrics = {}
    for key in all_keys:
        if not key.startswith("val_"):
            values = [
                eval_result.get(key, 0)
                for eval_result in recent_evals
                if key in eval_result
            ]
            if values:
                avg_metrics[f"avg_{key}"] = sum(values) / len(values)

    return avg_metrics
