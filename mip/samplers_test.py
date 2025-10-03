"""Tests for samplers.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import pytest
import torch

from mip.config import OptimizationConfig
from mip.encoders import IdentityEncoder
from mip.flow_map import FlowMap
from mip.networks.mlp import MLP, VanillaMLP
from mip.samplers import (
    flow_map_sampler,
    get_sampler,
    mip_sampler,
    ode_sampler,
    regression_sampler,
)


class TestGetSampler:
    """Test suite for get_sampler function"""

    def test_get_sampler_flow(self):
        """Test get_sampler returns ode_sampler for 'flow' loss type"""
        sampler = get_sampler("flow")
        assert sampler == ode_sampler

    def test_get_sampler_regression(self):
        """Test get_sampler returns regression_sampler for 'regression' loss type"""
        sampler = get_sampler("regression")
        assert sampler == regression_sampler

    def test_get_sampler_tsd(self):
        """Test get_sampler returns mip_sampler for 'tsd' loss type"""
        sampler = get_sampler("tsd")
        assert sampler == mip_sampler

    def test_get_sampler_mip(self):
        """Test get_sampler returns mip_sampler for 'mip' loss type"""
        sampler = get_sampler("mip")
        assert sampler == mip_sampler

    def test_get_sampler_lmd(self):
        """Test get_sampler returns flow_map_sampler for 'lmd' loss type"""
        sampler = get_sampler("lmd")
        assert sampler == flow_map_sampler

    def test_get_sampler_ctm(self):
        """Test get_sampler returns flow_map_sampler for 'ctm' loss type"""
        sampler = get_sampler("ctm")
        assert sampler == flow_map_sampler

    def test_get_sampler_invalid(self):
        """Test get_sampler raises NotImplementedError for invalid loss type"""
        with pytest.raises(NotImplementedError):
            get_sampler("invalid_loss_type")


class TestODESampler:
    """Test suite for ode_sampler function"""

    def test_ode_sampler_output_shape(self):
        """Test ode_sampler returns correct output shape"""
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

    def test_ode_sampler_zero_mode(self):
        """Test ode_sampler with zero initialization mode"""
        act_dim = 2
        Ta = 4
        obs_dim = 2
        To = 1
        bs = 8

        mlp = VanillaMLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=obs_dim,
            To=To,
            emb_dim=64,
            n_layers=3,
        )
        flow_map = FlowMap(mlp)
        encoder = IdentityEncoder()
        config = OptimizationConfig(num_steps=5, sample_mode="zero")

        act_0 = torch.randn(bs, Ta, act_dim)
        obs = torch.randn(bs, To, obs_dim)

        # Execute
        act = ode_sampler(config, flow_map, encoder, act_0, obs)

        # Assert - output shape matches input
        assert act.shape == act_0.shape

    def test_ode_sampler_stochastic_mode(self):
        """Test ode_sampler with stochastic initialization mode"""
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
        config = OptimizationConfig(num_steps=5, sample_mode="stochastic")

        act_0 = torch.randn(bs, Ta, act_dim)
        obs = torch.randn(bs, To, obs_dim)

        # Execute
        act = ode_sampler(config, flow_map, encoder, act_0, obs)

        # Assert
        assert act.shape == act_0.shape

    def test_ode_sampler_single_step(self):
        """Test ode_sampler with single step"""
        act_dim = 2
        Ta = 4
        obs_dim = 2
        To = 1
        bs = 8

        mlp = VanillaMLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=obs_dim,
            To=To,
            emb_dim=64,
            n_layers=3,
        )
        flow_map = FlowMap(mlp)
        encoder = IdentityEncoder()
        config = OptimizationConfig(num_steps=1, sample_mode="zero")

        act_0 = torch.randn(bs, Ta, act_dim)
        obs = torch.randn(bs, To, obs_dim)

        act = ode_sampler(config, flow_map, encoder, act_0, obs)

        assert act.shape == act_0.shape

    def test_ode_sampler_multiple_steps(self):
        """Test ode_sampler with multiple steps"""
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
        config = OptimizationConfig(num_steps=20, sample_mode="zero")

        act_0 = torch.randn(bs, Ta, act_dim)
        obs = torch.randn(bs, To, obs_dim)

        act = ode_sampler(config, flow_map, encoder, act_0, obs)

        assert act.shape == act_0.shape


class TestFlowMapSampler:
    """Test suite for flow_map_sampler function"""

    def test_flow_map_sampler_output_shape(self):
        """Test flow_map_sampler returns correct output shape"""
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

    def test_flow_map_sampler_zero_mode(self):
        """Test flow_map_sampler with zero initialization mode"""
        act_dim = 2
        Ta = 4
        obs_dim = 2
        To = 1
        bs = 8

        mlp = VanillaMLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=obs_dim,
            To=To,
            emb_dim=64,
            n_layers=3,
        )
        flow_map = FlowMap(mlp)
        encoder = IdentityEncoder()
        config = OptimizationConfig(num_steps=5, sample_mode="zero")

        act_0 = torch.randn(bs, Ta, act_dim)
        obs = torch.randn(bs, To, obs_dim)

        act = flow_map_sampler(config, flow_map, encoder, act_0, obs)

        assert act.shape == act_0.shape

    def test_flow_map_sampler_stochastic_mode(self):
        """Test flow_map_sampler with stochastic initialization mode"""
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
        config = OptimizationConfig(num_steps=5, sample_mode="stochastic")

        act_0 = torch.randn(bs, Ta, act_dim)
        obs = torch.randn(bs, To, obs_dim)

        act = flow_map_sampler(config, flow_map, encoder, act_0, obs)

        assert act.shape == act_0.shape

    def test_flow_map_sampler_single_step(self):
        """Test flow_map_sampler with single step"""
        act_dim = 2
        Ta = 4
        obs_dim = 2
        To = 1
        bs = 8

        mlp = VanillaMLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=obs_dim,
            To=To,
            emb_dim=64,
            n_layers=3,
        )
        flow_map = FlowMap(mlp)
        encoder = IdentityEncoder()
        config = OptimizationConfig(num_steps=1, sample_mode="zero")

        act_0 = torch.randn(bs, Ta, act_dim)
        obs = torch.randn(bs, To, obs_dim)

        act = flow_map_sampler(config, flow_map, encoder, act_0, obs)

        assert act.shape == act_0.shape


class TestRegressionSampler:
    """Test suite for regression_sampler function"""

    def test_regression_sampler_output_shape(self):
        """Test regression_sampler returns correct output shape"""
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

    def test_regression_sampler_with_vanilla_mlp(self):
        """Test regression_sampler with VanillaMLP"""
        act_dim = 2
        Ta = 4
        obs_dim = 2
        To = 1
        bs = 8

        mlp = VanillaMLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=obs_dim,
            To=To,
            emb_dim=64,
            n_layers=3,
        )
        flow_map = FlowMap(mlp)
        encoder = IdentityEncoder()
        config = OptimizationConfig()

        act_0 = torch.randn(bs, Ta, act_dim)
        obs = torch.randn(bs, To, obs_dim)

        act = regression_sampler(config, flow_map, encoder, act_0, obs)

        assert act.shape == act_0.shape


class TestMIPSampler:
    """Test suite for mip_sampler function"""

    def test_mip_sampler_output_shape(self):
        """Test mip_sampler returns correct output shape"""
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

    def test_mip_sampler_different_t_values(self):
        """Test mip_sampler with different t_two_step values"""
        act_dim = 2
        Ta = 4
        obs_dim = 2
        To = 1
        bs = 8

        mlp = VanillaMLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=obs_dim,
            To=To,
            emb_dim=64,
            n_layers=3,
        )
        flow_map = FlowMap(mlp)
        encoder = IdentityEncoder()

        act_0 = torch.randn(bs, Ta, act_dim)
        obs = torch.randn(bs, To, obs_dim)

        # Test with different t_two_step values
        for t_val in [0.5, 0.7, 0.9, 1.0]:
            config = OptimizationConfig(t_two_step=t_val)
            act = mip_sampler(config, flow_map, encoder, act_0, obs)
            assert act.shape == act_0.shape

    def test_mip_sampler_with_vanilla_mlp(self):
        """Test mip_sampler with VanillaMLP"""
        act_dim = 2
        Ta = 4
        obs_dim = 2
        To = 1
        bs = 8

        mlp = VanillaMLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=obs_dim,
            To=To,
            emb_dim=64,
            n_layers=3,
        )
        flow_map = FlowMap(mlp)
        encoder = IdentityEncoder()
        config = OptimizationConfig(t_two_step=0.9)

        act_0 = torch.randn(bs, Ta, act_dim)
        obs = torch.randn(bs, To, obs_dim)

        act = mip_sampler(config, flow_map, encoder, act_0, obs)

        assert act.shape == act_0.shape
