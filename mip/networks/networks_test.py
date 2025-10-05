"""Comprehensive test suite for all network implementations.
Tests that all networks follow the BaseNetwork interface correctly.

Author: Chaoyi Pan
Date: 2025-10-04
"""

import pytest
import torch

from mip.networks.base import BaseNetwork
from mip.networks.chitfm import ChiTransformer
from mip.networks.chiunet import ChiUNet
from mip.networks.jannerunet import JannerUNet
from mip.networks.mlp import MLP, VanillaMLP
from mip.networks.rnn import RNN, VanillaRNN
from mip.networks.sudeepdit import SudeepDiT

# Test parameters
BATCH_SIZE = 4
ACT_DIM = 2
OBS_DIM = 3
TA = 8  # Must be power of 2 for U-Net architectures
TO = 2


class TestNetworkInterface:
    """Test that all networks follow the BaseNetwork interface."""

    @pytest.fixture(
        params=[
            MLP,
            VanillaMLP,
            ChiTransformer,
            ChiUNet,
            JannerUNet,
            RNN,
            VanillaRNN,
            SudeepDiT,
        ]
    )
    def network_class(self, request):
        return request.param

    def get_network_kwargs(self, network_class):
        """Get appropriate kwargs for each network class."""
        base_kwargs = {
            "act_dim": ACT_DIM,
            "Ta": TA,
            "obs_dim": OBS_DIM,
            "To": TO,
        }

        if network_class == MLP:
            return {
                **base_kwargs,
                "emb_dim": 128,
                "n_layers": 3,
                "timestep_emb_dim": 64,
                "dropout": 0.1,
            }
        elif network_class == VanillaMLP:
            return {
                **base_kwargs,
                "emb_dim": 128,
                "n_layers": 3,
                "dropout": 0.1,
            }
        elif network_class == ChiTransformer:
            return {
                **base_kwargs,
                "d_model": 128,
                "nhead": 4,
                "num_layers": 2,
            }
        elif network_class == ChiUNet:
            return {
                **base_kwargs,
                "model_dim": 64,
                "emb_dim": 64,
                "dim_mult": [1, 2],
            }
        elif network_class == JannerUNet:
            return {
                **base_kwargs,
                "model_dim": 32,
                "emb_dim": 32,
                "dim_mult": [1, 2],
                "attention": True,
            }
        elif network_class in [RNN, VanillaRNN]:
            return {
                **base_kwargs,
                "rnn_hidden_dim": 128,
                "rnn_num_layers": 2,
                "rnn_type": "LSTM",
                "dropout": 0.1,
            }
        elif network_class == SudeepDiT:
            return {
                **base_kwargs,
                "d_model": 128,
                "n_heads": 4,
                "depth": 2,
            }
        else:
            raise ValueError(f"Unknown network class: {network_class}")

    def test_inheritance(self, network_class):
        """Test that all networks inherit from BaseNetwork."""
        assert issubclass(network_class, BaseNetwork)

    def test_forward_signature(self, network_class):
        """Test that forward method has correct signature."""
        kwargs = self.get_network_kwargs(network_class)
        model = network_class(**kwargs)

        # Prepare inputs
        x = torch.randn(BATCH_SIZE, TA, ACT_DIM)
        s = torch.randn(BATCH_SIZE)
        t = torch.randn(BATCH_SIZE)
        condition = torch.randn(BATCH_SIZE, TO, OBS_DIM)

        # Test forward pass
        output = model(x, s, t, condition)

        # Check output format
        assert isinstance(output, tuple), "Output should be a tuple"
        assert len(output) == 2, "Output should have 2 elements (action, scalar)"

        action, scalar = output
        assert action.shape == (BATCH_SIZE, TA, ACT_DIM), (
            f"Action shape mismatch: {action.shape}"
        )
        assert scalar.shape == (BATCH_SIZE, 1), f"Scalar shape mismatch: {scalar.shape}"

    def test_without_condition(self, network_class):
        """Test that networks can handle None condition."""
        kwargs = self.get_network_kwargs(network_class)
        model = network_class(**kwargs)

        # Prepare inputs without condition
        x = torch.randn(BATCH_SIZE, TA, ACT_DIM)
        s = torch.randn(BATCH_SIZE)
        t = torch.randn(BATCH_SIZE)

        # Test forward pass without condition
        action, scalar = model(x, s, t, None)

        assert action.shape == (BATCH_SIZE, TA, ACT_DIM)
        assert scalar.shape == (BATCH_SIZE, 1)

    def test_disable_time_embedding(self, network_class):
        """Test networks with disable_time_embedding=True."""
        kwargs = self.get_network_kwargs(network_class)

        # Add disable_time_embedding if supported
        if network_class not in [VanillaMLP, VanillaRNN]:
            kwargs["disable_time_embedding"] = True
            model = network_class(**kwargs)

            # Prepare inputs
            x = torch.randn(BATCH_SIZE, TA, ACT_DIM)
            s = torch.randn(BATCH_SIZE)
            t = torch.randn(BATCH_SIZE)
            condition = torch.randn(BATCH_SIZE, TO, OBS_DIM)

            # Get outputs with different time values
            model.eval()
            with torch.no_grad():
                y1, scalar1 = model(x, s, t, condition)
                y2, scalar2 = model(
                    x, torch.randn(BATCH_SIZE), torch.randn(BATCH_SIZE), condition
                )

            # Should give same output when time embedding is disabled
            assert torch.allclose(y1, y2, atol=1e-6), "Outputs should be time-invariant"
            assert torch.allclose(scalar1, scalar2, atol=1e-6), (
                "Scalars should be time-invariant"
            )

    def test_gradient_flow(self, network_class):
        """Test that gradients flow through the network."""
        kwargs = self.get_network_kwargs(network_class)
        model = network_class(**kwargs)

        # Prepare inputs
        x = torch.randn(BATCH_SIZE, TA, ACT_DIM, requires_grad=True)
        s = torch.randn(BATCH_SIZE, requires_grad=True)
        t = torch.randn(BATCH_SIZE, requires_grad=True)
        condition = torch.randn(BATCH_SIZE, TO, OBS_DIM, requires_grad=True)

        # Forward pass
        action, scalar = model(x, s, t, condition)

        # Compute loss
        loss = action.mean() + scalar.mean()

        # Backward pass
        loss.backward()

        # Check gradients
        assert x.grad is not None, "x should have gradients"
        assert s.grad is not None, "s should have gradients"
        assert t.grad is not None, "t should have gradients"
        assert condition.grad is not None, "condition should have gradients"

        # Check that at least some model parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "Model parameters should have gradients"


class TestNetworkSpecific:
    """Test specific properties of individual networks."""

    def test_mlp_layers(self):
        """Test MLP specific properties."""
        model = MLP(
            act_dim=ACT_DIM,
            Ta=TA,
            obs_dim=OBS_DIM,
            To=TO,
            emb_dim=256,
            n_layers=4,
            timestep_emb_dim=64,
        )

        # Check number of residual blocks
        assert len(model.residual_blocks) == 4

    def test_rnn_types(self):
        """Test RNN with different RNN types."""
        for rnn_type in ["LSTM", "GRU"]:
            model = RNN(
                act_dim=ACT_DIM,
                Ta=TA,
                obs_dim=OBS_DIM,
                To=TO,
                rnn_hidden_dim=128,
                rnn_num_layers=2,
                rnn_type=rnn_type,
            )

            x = torch.randn(BATCH_SIZE, TA, ACT_DIM)
            s = torch.randn(BATCH_SIZE)
            t = torch.randn(BATCH_SIZE)
            condition = torch.randn(BATCH_SIZE, TO, OBS_DIM)

            action, scalar = model(x, s, t, condition)
            assert action.shape == (BATCH_SIZE, TA, ACT_DIM)

    def test_unet_dimensions(self):
        """Test U-Net architectures with different dimensions."""
        # Test ChiUNet
        model = ChiUNet(
            act_dim=ACT_DIM,
            Ta=16,  # Must be power of 2
            obs_dim=OBS_DIM,
            To=TO,
            model_dim=32,
            emb_dim=32,
            dim_mult=[1, 2, 4],
        )

        x = torch.randn(BATCH_SIZE, 16, ACT_DIM)
        s = torch.randn(BATCH_SIZE)
        t = torch.randn(BATCH_SIZE)
        condition = torch.randn(BATCH_SIZE, TO, OBS_DIM)

        action, scalar = model(x, s, t, condition)
        assert action.shape == (BATCH_SIZE, 16, ACT_DIM)

    def test_transformer_attention_heads(self):
        """Test transformer architectures with different attention heads."""
        for nhead in [1, 2, 4, 8]:
            if 128 % nhead != 0:
                continue

            model = ChiTransformer(
                act_dim=ACT_DIM,
                Ta=TA,
                obs_dim=OBS_DIM,
                To=TO,
                d_model=128,
                nhead=nhead,
                num_layers=2,
            )

            x = torch.randn(BATCH_SIZE, TA, ACT_DIM)
            s = torch.randn(BATCH_SIZE)
            t = torch.randn(BATCH_SIZE)
            condition = torch.randn(BATCH_SIZE, TO, OBS_DIM)

            action, scalar = model(x, s, t, condition)
            assert action.shape == (BATCH_SIZE, TA, ACT_DIM)


def test_network_creation_from_config():
    """Test that networks can be created from config using network_utils."""
    from mip.config import NetworkConfig, TaskConfig
    from mip.network_utils import get_network

    # Create task config
    task_config = TaskConfig(
        act_dim=ACT_DIM,
        obs_dim=OBS_DIM,
        horizon=TA,
        obs_steps=TO,
    )

    # Test each network type
    network_types = [
        "mlp",
        "vanilla_mlp",
        "chitransformer",
        "chiunet",
        "jannerunet",
        "rnn",
        "vanilla_rnn",
        "sudeepdit",
    ]

    for network_type in network_types:
        network_config = NetworkConfig(
            network_type=network_type,
            emb_dim=128,
            num_layers=2,
            dropout=0.1,
            timestep_emb_dim=64,
        )

        # Create network
        model = get_network(network_config, task_config)

        # Test forward pass
        x = torch.randn(BATCH_SIZE, TA, ACT_DIM)
        s = torch.randn(BATCH_SIZE)
        t = torch.randn(BATCH_SIZE)
        condition = torch.randn(BATCH_SIZE, TO, OBS_DIM)

        action, scalar = model(x, s, t, condition)
        assert action.shape == (BATCH_SIZE, TA, ACT_DIM)
        assert scalar.shape == (BATCH_SIZE, 1)


if __name__ == "__main__":
    # Run basic tests
    print("Testing all networks...")

    # Test each network individually
    networks = [
        ("MLP", MLP),
        ("VanillaMLP", VanillaMLP),
        ("ChiTransformer", ChiTransformer),
        ("ChiUNet", ChiUNet),
        ("JannerUNet", JannerUNet),
        ("RNN", RNN),
        ("VanillaRNN", VanillaRNN),
        ("SudeepDiT", SudeepDiT),
    ]

    for name, network_class in networks:
        print(f"\nTesting {name}...")

        # Get appropriate kwargs
        test = TestNetworkInterface()
        kwargs = test.get_network_kwargs(network_class)
        model = network_class(**kwargs)

        # Test forward pass
        x = torch.randn(BATCH_SIZE, TA, ACT_DIM)
        s = torch.randn(BATCH_SIZE)
        t = torch.randn(BATCH_SIZE)
        condition = torch.randn(BATCH_SIZE, TO, OBS_DIM)

        action, scalar = model(x, s, t, condition)
        print(f"  ✓ Forward pass: action={action.shape}, scalar={scalar.shape}")

        # Test without condition
        action, scalar = model(x, s, t, None)
        print(f"  ✓ Without condition: action={action.shape}, scalar={scalar.shape}")

        # Test disable_time_embedding if supported
        if network_class not in [VanillaMLP, VanillaRNN]:
            kwargs["disable_time_embedding"] = True
            model_no_time = network_class(**kwargs)
            model_no_time.eval()
            with torch.no_grad():
                y1, s1 = model_no_time(x, s, t, condition)
                y2, s2 = model_no_time(
                    x, torch.randn(BATCH_SIZE), torch.randn(BATCH_SIZE), condition
                )
            time_invariant = torch.allclose(y1, y2, atol=1e-6) and torch.allclose(
                s1, s2, atol=1e-6
            )
            print(f"  ✓ Time embedding disabled: time_invariant={time_invariant}")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  ✓ Parameters: {num_params:.2f}M")

    print("\n✅ All network tests passed!")
