"""Tests for MLP networks."""

import torch

from mip.networks.mlp import MLP, VanillaMLP


class TestVanillaMLP:
    """Test suite for the VanillaMLP class."""

    def test_vanilla_mlp_creation(self):
        """Test creating a VanillaMLP network."""
        mlp = VanillaMLP(
            act_dim=2,
            Ta=4,
            obs_dim=2,
            To=1,
            emb_dim=512,
            n_layers=6,
            dropout=0.1,
        )
        assert mlp is not None

    def test_vanilla_mlp_output_shapes(self):
        """Test VanillaMLP output shapes."""
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
            emb_dim=512,
            n_layers=6,
            dropout=0.1,
        )

        x = torch.randn(bs, Ta, act_dim)
        s = torch.randn(bs)
        t = torch.randn(bs)
        condition = torch.randn(bs, To, obs_dim)

        y, scalar_out = mlp(x, s, t, condition)

        assert y.shape == (bs, Ta, act_dim)
        assert scalar_out.shape == (bs, 1)

    def test_vanilla_mlp_time_dependency(self):
        """Test that VanillaMLP produces different outputs for different time values."""
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
            emb_dim=512,
            n_layers=6,
            dropout=0.1,
        )

        x = torch.randn(bs, Ta, act_dim)
        s = torch.randn(bs)
        t = torch.randn(bs)
        condition = torch.randn(bs, To, obs_dim)

        s2 = torch.randn(bs)
        t2 = torch.randn(bs)

        mlp.eval()
        with torch.no_grad():
            y1, scalar_out1 = mlp(x, s, t, condition)
            y2, scalar_out2 = mlp(x, s2, t2, condition)

        is_different = not (
            torch.allclose(y1, y2, atol=1e-6)
            and torch.allclose(scalar_out1, scalar_out2, atol=1e-6)
        )
        assert is_different


class TestMLP:
    """Test suite for the MLP class."""

    def test_mlp_creation(self):
        """Test creating an MLP network."""
        mlp = MLP(
            act_dim=7,
            Ta=8,
            obs_dim=20,
            To=3,
            emb_dim=512,
            n_layers=6,
            timestep_emb_dim=128,
            max_freq=100.0,
            disable_time_embedding=False,
            dropout=0.1,
        )
        assert mlp is not None

    def test_mlp_output_shapes(self):
        """Test MLP output shapes."""
        act_dim = 7
        Ta = 8
        obs_dim = 20
        To = 3
        bs = 4

        mlp = MLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=obs_dim,
            To=To,
            emb_dim=512,
            n_layers=6,
            timestep_emb_dim=128,
            max_freq=100.0,
            disable_time_embedding=False,
            dropout=0.1,
        )

        x = torch.randn(bs, Ta, act_dim)
        s = torch.randn(bs)
        t = torch.randn(bs)
        condition = torch.randn(bs, To, obs_dim)

        y, scalar_out = mlp(x, s, t, condition)

        assert y.shape == (bs, Ta, act_dim)
        assert scalar_out.shape == (bs, 1)

    def test_mlp_time_dependency(self):
        """Test that MLP produces different outputs for different time values."""
        act_dim = 7
        Ta = 8
        obs_dim = 20
        To = 3
        bs = 4

        mlp = MLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=obs_dim,
            To=To,
            emb_dim=512,
            n_layers=6,
            timestep_emb_dim=128,
            max_freq=100.0,
            disable_time_embedding=False,
            dropout=0.1,
        )

        x = torch.randn(bs, Ta, act_dim)
        s = torch.randn(bs)
        t = torch.randn(bs)
        condition = torch.randn(bs, To, obs_dim)

        s2 = torch.randn(bs)
        t2 = torch.randn(bs)

        mlp.eval()
        with torch.no_grad():
            y1, scalar_out1 = mlp(x, s, t, condition)
            y2, scalar_out2 = mlp(x, s2, t2, condition)

        is_different = not (
            torch.allclose(y1, y2, atol=1e-6)
            and torch.allclose(scalar_out1, scalar_out2, atol=1e-6)
        )
        assert is_different

    def test_mlp_disable_time_embedding(self):
        """Test MLP with time embedding disabled."""
        act_dim = 7
        Ta = 8
        obs_dim = 20
        To = 3
        bs = 4

        mlp = MLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=obs_dim,
            To=To,
            emb_dim=512,
            n_layers=6,
            disable_time_embedding=True,
            dropout=0.1,
        )

        x = torch.randn(bs, Ta, act_dim)
        s = torch.randn(bs)
        t = torch.randn(bs)
        condition = torch.randn(bs, To, obs_dim)

        y, scalar_out = mlp(x, s, t, condition)

        assert y.shape == (bs, Ta, act_dim)
        assert scalar_out.shape == (bs, 1)
