"""Simple tests for logger.py.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import os
import tempfile
from unittest.mock import Mock, patch

import gymnasium as gym

from mip.config import Config, LogConfig, NetworkConfig, OptimizationConfig, TaskConfig
from mip.env_utils import VideoRecorder, VideoRecordingWrapper
from mip.logger import (
    Logger,
    compute_average_metrics,
    make_dir,
    update_best_metrics,
)


class TestMakeDir:
    """Test the make_dir function."""

    def test_make_dir_creates_new_directory(self):
        """Test that make_dir creates a new directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = os.path.join(tmpdir, "test_dir")
            result = make_dir(test_dir)
            assert os.path.exists(test_dir)
            assert result == test_dir

    def test_make_dir_existing_directory(self):
        """Test that make_dir handles existing directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Directory already exists
            result = make_dir(tmpdir)
            assert os.path.exists(tmpdir)
            assert result == tmpdir


class TestUpdateBestMetrics:
    """Test the update_best_metrics function."""

    def test_update_with_higher_success_rate(self):
        """Test that higher success rates update best metrics."""
        best_metrics = {"mean_success_task1": 0.5}
        current_metrics = {"mean_success_task1": 0.8}
        result = update_best_metrics(best_metrics, current_metrics)
        assert result["mean_success_task1"] == 0.8

    def test_update_with_higher_reward(self):
        """Test that higher rewards update best metrics."""
        best_metrics = {"mean_reward_task1": 10.0}
        current_metrics = {"mean_reward_task1": 15.0}
        result = update_best_metrics(best_metrics, current_metrics)
        assert result["mean_reward_task1"] == 15.0

    def test_update_with_lower_steps(self):
        """Test that lower step counts update best metrics."""
        best_metrics = {"mean_step_task1": 100}
        current_metrics = {"mean_step_task1": 80}
        result = update_best_metrics(best_metrics, current_metrics)
        assert result["mean_step_task1"] == 80

    def test_ignore_val_metrics(self):
        """Test that validation metrics are ignored."""
        best_metrics = {}
        current_metrics = {
            "val_mean_success_task1": 0.9,
            "mean_success_task1": 0.8,
        }
        result = update_best_metrics(best_metrics, current_metrics)
        assert "val_mean_success_task1" not in result
        assert result["mean_success_task1"] == 0.8

    def test_new_metric_added(self):
        """Test that new metrics are added to best_metrics."""
        best_metrics = {}
        current_metrics = {"mean_success_new_task": 0.7}
        result = update_best_metrics(best_metrics, current_metrics)
        assert result["mean_success_new_task"] == 0.7


class TestComputeAverageMetrics:
    """Test the compute_average_metrics function."""

    def test_empty_history(self):
        """Test with empty evaluation history."""
        result = compute_average_metrics([])
        assert result == {}

    def test_single_evaluation(self):
        """Test with single evaluation."""
        eval_history = [{"mean_success_task1": 0.8, "mean_reward_task1": 10.0}]
        result = compute_average_metrics(eval_history)
        assert result["avg_mean_success_task1"] == 0.8
        assert result["avg_mean_reward_task1"] == 10.0

    def test_multiple_evaluations(self):
        """Test averaging over multiple evaluations."""
        eval_history = [
            {"mean_success_task1": 0.6, "mean_reward_task1": 10.0},
            {"mean_success_task1": 0.8, "mean_reward_task1": 15.0},
            {"mean_success_task1": 0.7, "mean_reward_task1": 12.0},
        ]
        result = compute_average_metrics(eval_history)
        assert abs(result["avg_mean_success_task1"] - 0.7) < 1e-9
        # (10.0 + 15.0 + 12.0) / 3 = 12.333...
        assert abs(result["avg_mean_reward_task1"] - 12.333333333333334) < 1e-9

    def test_last_five_evaluations(self):
        """Test that only last 5 evaluations are used."""
        eval_history = [
            {"mean_success_task1": 0.1},
            {"mean_success_task1": 0.2},
            {"mean_success_task1": 0.3},
            {"mean_success_task1": 0.4},
            {"mean_success_task1": 0.5},
            {"mean_success_task1": 0.6},
            {"mean_success_task1": 0.7},
        ]
        result = compute_average_metrics(eval_history)
        # Should average last 5: 0.3, 0.4, 0.5, 0.6, 0.7
        expected = (0.3 + 0.4 + 0.5 + 0.6 + 0.7) / 5
        assert result["avg_mean_success_task1"] == expected

    def test_ignore_val_metrics(self):
        """Test that validation metrics are ignored."""
        eval_history = [
            {
                "val_mean_success_task1": 0.9,
                "mean_success_task1": 0.8,
            }
        ]
        result = compute_average_metrics(eval_history)
        assert "avg_val_mean_success_task1" not in result
        assert result["avg_mean_success_task1"] == 0.8


class TestLogger:
    """Test the Logger class."""

    @patch("mip.logger.wandb")
    def test_logger_initialization(self, mock_wandb):
        """Test Logger initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_config = LogConfig(
                log_dir=tmpdir,
                wandb_mode="disabled",
                project="test_project",
                group="test_group",
                exp_name="test_exp",
            )
            config = Config(
                optimization=OptimizationConfig(),
                network=NetworkConfig(),
                task=TaskConfig(),
                log=log_config,
            )

            Logger(config)

            # Check that directories were created
            assert os.path.exists(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "models"))

            # Check that wandb.init was called
            mock_wandb.init.assert_called_once()
            call_kwargs = mock_wandb.init.call_args[1]
            assert call_kwargs["project"] == "test_project"
            assert call_kwargs["group"] == "test_group"
            assert call_kwargs["name"] == "test_exp"
            assert call_kwargs["mode"] == "disabled"

    @patch("mip.logger.wandb")
    @patch("mip.logger.loguru")
    def test_logger_log_method(self, mock_loguru, mock_wandb):
        """Test Logger log method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_config = LogConfig(
                log_dir=tmpdir,
                wandb_mode="disabled",
                project="test_project",
                group="test_group",
                exp_name="test_exp",
            )
            config = Config(
                optimization=OptimizationConfig(),
                network=NetworkConfig(),
                task=TaskConfig(),
                log=log_config,
            )

            # Create a mock Image class for wandb
            mock_wandb.Image = type("Image", (), {})

            logger = Logger(config)

            # Test logging
            log_data = {
                "step": 100,
                "loss": 0.5,
                "accuracy": 0.95,
            }
            logger.log(log_data, "train")

            # Check that wandb.log was called
            mock_wandb.log.assert_called_once()
            call_args = mock_wandb.log.call_args[0][0]
            assert call_args["train/step"] == 100
            assert call_args["train/loss"] == 0.5
            assert call_args["train/accuracy"] == 0.95

            # Check that metrics were written to file
            metrics_file = os.path.join(tmpdir, "metrics.jsonl")
            assert os.path.exists(metrics_file)

    @patch("mip.logger.wandb")
    def test_logger_save_agent(self, mock_wandb):
        """Test Logger save_agent method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_config = LogConfig(
                log_dir=tmpdir,
                wandb_mode="disabled",
                project="test_project",
                group="test_group",
                exp_name="test_exp",
            )
            config = Config(
                optimization=OptimizationConfig(),
                network=NetworkConfig(),
                task=TaskConfig(),
                log=log_config,
            )

            logger = Logger(config)

            # Create a mock agent
            mock_agent = Mock()
            mock_agent.save = Mock()

            logger.save_agent(mock_agent, identifier="checkpoint_100")

            # Check that agent.save was called
            mock_agent.save.assert_called_once()

    @patch("mip.logger.wandb")
    def test_logger_finish(self, mock_wandb):
        """Test Logger finish method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_config = LogConfig(
                log_dir=tmpdir,
                wandb_mode="disabled",
                project="test_project",
                group="test_group",
                exp_name="test_exp",
            )
            config = Config(
                optimization=OptimizationConfig(),
                network=NetworkConfig(),
                task=TaskConfig(),
                log=log_config,
            )

            logger = Logger(config)

            # Create a mock agent
            mock_agent = Mock()
            mock_agent.save = Mock()

            logger.finish(mock_agent)

            # Check that wandb.finish was called
            mock_wandb.finish.assert_called_once()

    @patch("mip.logger.wandb")
    def test_logger_creates_video_directory(self, mock_wandb):
        """Test that Logger creates video directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_config = LogConfig(
                log_dir=tmpdir,
                wandb_mode="disabled",
                project="test_project",
                group="test_group",
                exp_name="test_exp",
            )
            config = Config(
                optimization=OptimizationConfig(),
                network=NetworkConfig(),
                task=TaskConfig(),
                log=log_config,
            )

            _ = Logger(config)

            # Check that video directory was created
            assert os.path.exists(os.path.join(tmpdir, "videos"))

    @patch("mip.logger.wandb")
    def test_video_init_with_video_recording_wrapper(self, mock_wandb):
        """Test video_init with VideoRecordingWrapper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_config = LogConfig(
                log_dir=tmpdir,
                wandb_mode="disabled",
                project="test_project",
                group="test_group",
                exp_name="test_exp",
            )
            config = Config(
                optimization=OptimizationConfig(),
                network=NetworkConfig(),
                task=TaskConfig(),
                log=log_config,
            )

            logger = Logger(config)

            # Create a proper mock gymnasium environment
            Mock(spec=gym.Env)
            mock_recorder = Mock(spec=VideoRecorder)
            mock_recorder.stop = Mock()

            # Manually create a mock VideoRecordingWrapper rather than instantiating it
            video_env = Mock(spec=VideoRecordingWrapper)
            video_env.video_recoder = mock_recorder
            video_env.file_path = None

            # Create outer env wrapper
            mock_env = Mock()
            mock_env.env = video_env

            # Test enabling video recording
            logger.video_init(mock_env, enable=True, video_id="test_video")

            # Check that file_path was set
            assert video_env.file_path is not None
            assert "test_video" in video_env.file_path
            assert video_env.file_path.endswith(".mp4")

            # Check that stop was called
            mock_recorder.stop.assert_called_once()

    @patch("mip.logger.wandb")
    def test_video_init_disable_recording(self, mock_wandb):
        """Test video_init with recording disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_config = LogConfig(
                log_dir=tmpdir,
                wandb_mode="disabled",
                project="test_project",
                group="test_group",
                exp_name="test_exp",
            )
            config = Config(
                optimization=OptimizationConfig(),
                network=NetworkConfig(),
                task=TaskConfig(),
                log=log_config,
            )

            logger = Logger(config)

            # Create a mock environment with VideoRecordingWrapper
            mock_recorder = Mock(spec=VideoRecorder)

            # Manually create a mock VideoRecordingWrapper
            video_env = Mock(spec=VideoRecordingWrapper)
            video_env.video_recoder = mock_recorder
            video_env.file_path = "/some/path.mp4"

            # Create outer env wrapper
            mock_env = Mock()
            mock_env.env = video_env

            # Test disabling video recording
            logger.video_init(mock_env, enable=False)

            # Check that file_path was set to None
            assert video_env.file_path is None

    @patch("mip.logger.wandb")
    def test_video_init_without_wrapper(self, mock_wandb):
        """Test video_init with environment without VideoRecordingWrapper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_config = LogConfig(
                log_dir=tmpdir,
                wandb_mode="disabled",
                project="test_project",
                group="test_group",
                exp_name="test_exp",
            )
            config = Config(
                optimization=OptimizationConfig(),
                network=NetworkConfig(),
                task=TaskConfig(),
                log=log_config,
            )

            logger = Logger(config)

            # Create a mock environment without VideoRecordingWrapper
            mock_env = Mock()
            mock_env.env = Mock()

            # This should not raise an error, just return early
            logger.video_init(mock_env, enable=True, video_id="test")


if __name__ == "__main__":
    import pytest

    # Run tests
    pytest.main([__file__, "-v"])
