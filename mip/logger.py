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
