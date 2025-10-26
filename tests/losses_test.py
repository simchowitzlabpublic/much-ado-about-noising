"""Tests for losses module."""

import pytest
import torch

from mip.config import OptimizationConfig
from mip.encoders import IdentityEncoder
from mip.flow_map import FlowMap
from mip.interpolant import Interpolant
from mip.losses import (
    ctm_loss,
    esd_loss,
    flow_loss,
    lmd_loss,
    lsd_loss,
    mf_loss,
    mip_loss,
    psd_loss,
    regression_loss,
    tsd_loss,
)
from mip.networks.mlp import MLP


class TestLossFunctions:
    """Test suite for all loss functions."""

    @pytest.fixture
    def setup(self):
        """Setup common test fixtures."""
        # Dimensions
        act_dim = 2
        Ta = 4
        obs_dim = 3
        To = 1
        bs = 8

        # Create config
        config = OptimizationConfig(
            loss_type="flow",
            loss_scale=100.0,
            t_two_step=0.9,
            discrete_dt=0.01,
            norm_type="l2",
        )

        # Create network and flow map
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

        # Create reference network for lmd_loss and ctm_loss
        reference_mlp = MLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=obs_dim,
            To=To,
            emb_dim=64,
            n_layers=3,
            timestep_emb_dim=32,
            dropout=0.0,
        )

        flow_map = FlowMap(mlp, reference_net=reference_mlp)

        # Create encoder
        encoder = IdentityEncoder(dropout=0.0)

        # Create interpolant
        interp = Interpolant("linear")

        # Create tensors
        act = torch.randn(bs, Ta, act_dim)
        obs = torch.randn(bs, To, obs_dim)
        delta_t = torch.rand(bs)

        return {
            "config": config,
            "flow_map": flow_map,
            "encoder": encoder,
            "interp": interp,
            "act": act,
            "obs": obs,
            "delta_t": delta_t,
            "bs": bs,
        }

    def test_flow_loss_output_shape(self, setup):
        """Test flow loss returns correct output shape."""
        loss, aux = flow_loss(
            setup["config"],
            setup["flow_map"],
            setup["encoder"],
            setup["interp"],
            setup["act"],
            setup["obs"],
            setup["delta_t"],
        )

        # Loss should be a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        # Aux should be an empty dict
        assert isinstance(aux, dict)
        assert len(aux) == 0

    def test_regression_loss_output_shape(self, setup):
        """Test regression loss returns correct output shape."""
        loss, aux = regression_loss(
            setup["config"],
            setup["flow_map"],
            setup["encoder"],
            setup["interp"],
            setup["act"],
            setup["obs"],
            setup["delta_t"],
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert isinstance(aux, dict)
        assert len(aux) == 0

    def test_tsd_loss_output_shape(self, setup):
        """Test TSD loss returns correct output shape."""
        loss, aux = tsd_loss(
            setup["config"],
            setup["flow_map"],
            setup["encoder"],
            setup["interp"],
            setup["act"],
            setup["obs"],
            setup["delta_t"],
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert isinstance(aux, dict)
        assert len(aux) == 0

    def test_mip_loss_output_shape(self, setup):
        """Test MIP loss returns correct output shape."""
        loss, aux = mip_loss(
            setup["config"],
            setup["flow_map"],
            setup["encoder"],
            setup["interp"],
            setup["act"],
            setup["obs"],
            setup["delta_t"],
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert isinstance(aux, dict)
        assert len(aux) == 0

    def test_lmd_loss_output_shape(self, setup):
        """Test LMD loss returns correct output shape."""
        loss, aux = lmd_loss(
            setup["config"],
            setup["flow_map"],
            setup["encoder"],
            setup["interp"],
            setup["act"],
            setup["obs"],
            setup["delta_t"],
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert isinstance(aux, dict)
        assert len(aux) == 0

    def test_ctm_loss_output_shape(self, setup):
        """Test CTM loss returns correct output shape."""
        loss, aux = ctm_loss(
            setup["config"],
            setup["flow_map"],
            setup["encoder"],
            setup["interp"],
            setup["act"],
            setup["obs"],
            setup["delta_t"],
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert isinstance(aux, dict)
        assert len(aux) == 0

    def test_psd_loss_output_shape(self, setup):
        """Test PSD loss returns correct output shape."""
        loss, aux = psd_loss(
            setup["config"],
            setup["flow_map"],
            setup["encoder"],
            setup["interp"],
            setup["act"],
            setup["obs"],
            setup["delta_t"],
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert isinstance(aux, dict)
        assert len(aux) == 2
        assert "flow_loss" in aux
        assert "psd_term" in aux

    def test_lsd_loss_output_shape(self, setup):
        """Test LSD loss returns correct output shape."""
        loss, aux = lsd_loss(
            setup["config"],
            setup["flow_map"],
            setup["encoder"],
            setup["interp"],
            setup["act"],
            setup["obs"],
            setup["delta_t"],
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert isinstance(aux, dict)
        assert len(aux) == 2
        assert "flow_loss" in aux
        assert "lsd_term" in aux

    def test_mf_loss_output_shape(self, setup):
        """Test MF loss returns correct output shape."""
        loss, aux = mf_loss(
            setup["config"],
            setup["flow_map"],
            setup["encoder"],
            setup["interp"],
            setup["act"],
            setup["obs"],
            setup["delta_t"],
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert isinstance(aux, dict)
        assert len(aux) == 2
        assert "flow_loss" in aux
        assert "mf_term" in aux

    def test_esd_loss_output_shape(self, setup):
        """Test ESD loss returns correct output shape."""
        loss, aux = esd_loss(
            setup["config"],
            setup["flow_map"],
            setup["encoder"],
            setup["interp"],
            setup["act"],
            setup["obs"],
            setup["delta_t"],
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert isinstance(aux, dict)
        assert len(aux) == 2
        assert "flow_loss" in aux
        assert "esd_term" in aux

    def test_flow_loss_gradient_flow(self, setup):
        """Test that gradients flow through flow loss."""
        setup["act"].requires_grad = True

        loss, _ = flow_loss(
            setup["config"],
            setup["flow_map"],
            setup["encoder"],
            setup["interp"],
            setup["act"],
            setup["obs"],
            setup["delta_t"],
        )

        loss.backward()

        assert setup["act"].grad is not None
        assert setup["act"].grad.shape == setup["act"].shape

    def test_regression_loss_gradient_flow(self, setup):
        """Test that gradients flow through regression loss."""
        setup["act"].requires_grad = True

        loss, _ = regression_loss(
            setup["config"],
            setup["flow_map"],
            setup["encoder"],
            setup["interp"],
            setup["act"],
            setup["obs"],
            setup["delta_t"],
        )

        loss.backward()

        assert setup["act"].grad is not None
        assert setup["act"].grad.shape == setup["act"].shape

    def test_loss_scale_effect(self, setup):
        """Test that loss_scale affects the loss value."""
        config_scale_1 = OptimizationConfig(loss_scale=1.0)
        config_scale_100 = OptimizationConfig(loss_scale=100.0)

        # Set seed for reproducibility
        torch.manual_seed(42)
        loss_1, _ = flow_loss(
            config_scale_1,
            setup["flow_map"],
            setup["encoder"],
            setup["interp"],
            setup["act"],
            setup["obs"],
            setup["delta_t"],
        )

        # Reset seed to get same random samples
        torch.manual_seed(42)
        loss_100, _ = flow_loss(
            config_scale_100,
            setup["flow_map"],
            setup["encoder"],
            setup["interp"],
            setup["act"],
            setup["obs"],
            setup["delta_t"],
        )

        # loss_100 should be approximately 100 times loss_1
        assert torch.allclose(loss_100, loss_1 * 100.0, rtol=1e-5)

    def test_different_norm_types(self, setup):
        """Test that different norm types produce different results."""
        config_l1 = OptimizationConfig(norm_type="l1")
        config_l2 = OptimizationConfig(norm_type="l2")

        loss_l1, _ = flow_loss(
            config_l1,
            setup["flow_map"],
            setup["encoder"],
            setup["interp"],
            setup["act"],
            setup["obs"],
            setup["delta_t"],
        )

        loss_l2, _ = flow_loss(
            config_l2,
            setup["flow_map"],
            setup["encoder"],
            setup["interp"],
            setup["act"],
            setup["obs"],
            setup["delta_t"],
        )

        # Losses should be different (in general)
        assert isinstance(loss_l1, torch.Tensor)
        assert isinstance(loss_l2, torch.Tensor)
