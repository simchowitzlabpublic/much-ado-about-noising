"""Tests for FlowMap."""

import torch

from mip.flow_map import FlowMap
from mip.networks.mlp import MLP, VanillaMLP


class TestFlowMap:
    """Test suite for the FlowMap class."""

    def test_flowmap_creation(self):
        """Test creating a FlowMap with MLP network."""
        mlp = MLP(
            act_dim=2,
            Ta=4,
            obs_dim=2,
            To=1,
            emb_dim=64,
            n_layers=3,
            timestep_emb_dim=32,
        )
        flow_map = FlowMap(mlp)
        assert flow_map is not None

    def test_flowmap_forward(self):
        """Test FlowMap forward pass."""
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

        s = torch.randn(bs)
        t = torch.randn(bs)
        xs = torch.randn(bs, Ta, act_dim)
        label = torch.randn(bs, To, obs_dim)

        Xst = flow_map.forward(s, t, xs, label)

        assert Xst.shape == (bs, Ta, act_dim)

    def test_flowmap_forward_single(self):
        """Test FlowMap forward_single method."""
        act_dim = 2
        Ta = 4
        obs_dim = 2
        To = 1

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

        s = torch.tensor(0.0)
        t = torch.tensor(1.0)
        xs = torch.randn(Ta, act_dim)
        label = torch.randn(To, obs_dim)

        x = flow_map.forward_single(s, t, xs, label)

        assert x.shape == (Ta, act_dim)

    def test_flowmap_jvp_t_single(self):
        """Test FlowMap jvp_t_single method."""
        act_dim = 2
        Ta = 4
        obs_dim = 2
        To = 1

        mlp = MLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=obs_dim,
            To=To,
            emb_dim=64,
            n_layers=3,
            timestep_emb_dim=32,
            dropout=0.0,  # Disable dropout for jvp compatibility
        )
        flow_map = FlowMap(mlp)

        s = torch.tensor(0.0)
        t = torch.tensor(1.0)
        xs = torch.randn(Ta, act_dim)
        label = torch.randn(To, obs_dim)

        x, dx_dt = flow_map.jvp_t_single(s, t, xs, label)

        assert x.shape == (Ta, act_dim)
        assert dx_dt.shape == (Ta, act_dim)

    def test_flowmap_jvp_t(self):
        """Test FlowMap jvp_t method (batched version)."""
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
            dropout=0.0,  # Disable dropout for vmap compatibility
        )
        flow_map = FlowMap(mlp)

        s = torch.randn(bs)
        t = torch.randn(bs)
        xs = torch.randn(bs, Ta, act_dim)
        label = torch.randn(bs, To, obs_dim)

        x, dx_dt = flow_map.jvp_t(s, t, xs, label)

        assert x.shape == (bs, Ta, act_dim)
        assert dx_dt.shape == (bs, Ta, act_dim)

    def test_flowmap_get_velocity(self):
        """Test FlowMap get_velocity method."""
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

        t = torch.randn(bs)
        xs = torch.randn(bs, Ta, act_dim)
        label = torch.randn(bs, To, obs_dim)

        velocity = flow_map.get_velocity(t, xs, label)

        assert velocity.shape == (bs, Ta, act_dim)

    def test_flowmap_with_vanilla_mlp(self):
        """Test FlowMap with VanillaMLP network."""
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

        s = torch.randn(bs)
        t = torch.randn(bs)
        xs = torch.randn(bs, Ta, act_dim)
        label = torch.randn(bs, To, obs_dim)

        Xst = flow_map.forward(s, t, xs, label)

        assert Xst.shape == (bs, Ta, act_dim)

    def test_flowmap_gradient_flow(self):
        """Test that gradients flow through FlowMap."""
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

        s = torch.randn(bs, requires_grad=False)
        t = torch.randn(bs, requires_grad=False)
        xs = torch.randn(bs, Ta, act_dim, requires_grad=True)
        label = torch.randn(bs, To, obs_dim)

        Xst = flow_map.forward(s, t, xs, label)
        loss = Xst.sum()
        loss.backward()

        assert xs.grad is not None
        assert xs.grad.shape == xs.shape

    def test_flowmap_time_evolution(self):
        """Test that FlowMap evolves states correctly with time."""
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

        s = torch.zeros(bs)
        t = torch.ones(bs)
        xs = torch.randn(bs, Ta, act_dim)
        label = torch.randn(bs, To, obs_dim)

        mlp.eval()
        with torch.no_grad():
            Xst = flow_map.forward(s, t, xs, label)
            # When s=0 and t=1, Xst should be xs + 1 * F(xs, 0, 1)
            # Verify it's different from xs
            is_different = not torch.allclose(xs, Xst, atol=1e-6)
            assert is_different
