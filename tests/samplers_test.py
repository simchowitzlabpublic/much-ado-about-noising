"""Tests for samplers.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import torch

from mip.config import OptimizationConfig
from mip.encoders import IdentityEncoder
from mip.flow_map import FlowMap
from mip.networks.mlp import MLP
from mip.samplers import (
    flow_map_sampler,
    mip_sampler,
    ode_sampler,
    regression_sampler,
)


class TestODESampler:
    """Test suite for ode_sampler function."""

    def test_ode_sampler_output_shape(self):
        """Test ode_sampler returns correct output shape."""
        # Setup
        act_dim = 2
        Ta = 4
        obs_dim = 2
        To = 1
        bs = 8
        num_steps = 10

        mlp = MLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=obs_dim,
            To=To,
            emb_dim=64,
            n_layers=3,
            timestep_emb_dim=32,
            dropout=0.0,
        )
        flow_map = FlowMap(mlp)
        encoder = IdentityEncoder()
        config = OptimizationConfig(num_steps=num_steps, sample_mode="zero")

        act_0 = torch.randn(bs, Ta, act_dim)
        obs = torch.randn(bs, To, obs_dim)

        # Execute
        act = ode_sampler(config, flow_map, encoder, act_0, obs)

        # Assert
        assert act.shape == act_0.shape


class TestFlowMapSampler:
    """Test suite for flow_map_sampler function."""

    def test_flow_map_sampler_output_shape(self):
        """Test flow_map_sampler returns correct output shape."""
        act_dim = 2
        Ta = 4
        obs_dim = 2
        To = 1
        bs = 8

        mlp = MLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=obs_dim,
            To=To,
            emb_dim=64,
            n_layers=3,
            timestep_emb_dim=32,
        )
        flow_map = FlowMap(mlp)
        encoder = IdentityEncoder()
        config = OptimizationConfig(num_steps=5, sample_mode="zero")

        act_0 = torch.randn(bs, Ta, act_dim)
        obs = torch.randn(bs, To, obs_dim)

        act = flow_map_sampler(config, flow_map, encoder, act_0, obs)

        assert act.shape == act_0.shape


class TestRegressionSampler:
    """Test suite for regression_sampler function."""

    def test_regression_sampler_output_shape(self):
        """Test regression_sampler returns correct output shape."""
        act_dim = 2
        Ta = 4
        obs_dim = 2
        To = 1
        bs = 8

        mlp = MLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=obs_dim,
            To=To,
            emb_dim=64,
            n_layers=3,
            timestep_emb_dim=32,
        )
        flow_map = FlowMap(mlp)
        encoder = IdentityEncoder()
        config = OptimizationConfig()

        act_0 = torch.randn(bs, Ta, act_dim)
        obs = torch.randn(bs, To, obs_dim)

        act = regression_sampler(config, flow_map, encoder, act_0, obs)

        assert act.shape == act_0.shape


class TestMIPSampler:
    """Test suite for mip_sampler function."""

    def test_mip_sampler_output_shape(self):
        """Test mip_sampler returns correct output shape."""
        act_dim = 2
        Ta = 4
        obs_dim = 2
        To = 1
        bs = 8

        mlp = MLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=obs_dim,
            To=To,
            emb_dim=64,
            n_layers=3,
            timestep_emb_dim=32,
        )
        flow_map = FlowMap(mlp)
        encoder = IdentityEncoder()
        config = OptimizationConfig(t_two_step=0.9)

        act_0 = torch.randn(bs, Ta, act_dim)
        obs = torch.randn(bs, To, obs_dim)

        act = mip_sampler(config, flow_map, encoder, act_0, obs)

        assert act.shape == act_0.shape
