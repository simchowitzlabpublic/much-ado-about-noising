"""Tests for TrainingAgent.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import torch

from mip.agent import TrainingAgent
from mip.config import Config, LogConfig, NetworkConfig, OptimizationConfig, TaskConfig


class TestTrainingAgent:
    """Test suite for the TrainingAgent class."""

    def test_agent_different_loss_types(self):
        """Test agent with different loss types."""
        act_dim = 2
        Ta = 4
        obs_dim = 3
        To = 1
        bs = 8

        network_config = NetworkConfig(emb_dim=32, num_layers=2)
        task_config = TaskConfig(
            act_dim=act_dim, obs_dim=obs_dim, act_steps=Ta, obs_steps=To, horizon=Ta
        )
        log_config = LogConfig(
            log_dir="./logs",
            wandb_mode="disabled",
            project="test",
            group="test",
            exp_name="test",
        )

        for loss_type in ["flow", "regression", "tsd", "mip"]:
            optimization_config = OptimizationConfig(
                loss_type=loss_type, num_steps=5, device="cpu"
            )
            config = Config(
                optimization=optimization_config,
                network=network_config,
                task=task_config,
                log=log_config,
            )
            agent = TrainingAgent(config)

            act = torch.randn(bs, Ta, act_dim)
            obs = torch.randn(bs, To, obs_dim)
            delta_t = torch.rand(bs)
            act_0 = torch.randn(bs, Ta, act_dim)

            # Test update
            info = agent.update(act, obs, delta_t)
            assert isinstance(info, dict)
            assert "loss" in info

            # Test sample
            sampled_act = agent.sample(act_0, obs)
            assert sampled_act.shape == act_0.shape
