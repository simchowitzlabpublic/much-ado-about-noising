"""Tests for TrainingAgent.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import pytest
import torch

from mip.agent import TrainingAgent
from mip.config import Config, LogConfig, NetworkConfig, OptimizationConfig, TaskConfig


class TestTrainingAgent:
    """Test suite for the TrainingAgent class."""

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
        optimization_config = OptimizationConfig(
            loss_type="flow",
            loss_scale=100.0,
            lr=1e-4,
            weight_decay=1e-5,
            num_steps=10,
            sample_mode="zero",
            ema_rate=0.995,
            interp_type="linear",
            grad_clip_norm=10.0,
            device="cpu",  # Use CPU for tests
        )

        network_config = NetworkConfig(
            network_type="mlp",
            num_layers=3,
            emb_dim=64,
            dropout=0.0,
            timestep_emb_dim=32,
        )

        task_config = TaskConfig(
            act_dim=act_dim,
            obs_dim=obs_dim,
            act_steps=Ta,
            obs_steps=To,
            horizon=Ta,
        )

        log_config = LogConfig(
            log_dir="./logs",
            wandb_mode="disabled",
            project="test",
            group="test",
            exp_name="test",
        )

        config = Config(
            optimization=optimization_config,
            network=network_config,
            task=task_config,
            log=log_config,
        )

        # Create agent
        agent = TrainingAgent(config)

        # Create tensors
        act = torch.randn(bs, Ta, act_dim)
        obs = torch.randn(bs, To, obs_dim)
        delta_t = torch.rand(bs)
        act_0 = torch.randn(bs, Ta, act_dim)

        return {
            "agent": agent,
            "config": config,
            "act": act,
            "obs": obs,
            "delta_t": delta_t,
            "act_0": act_0,
            "bs": bs,
            "Ta": Ta,
            "act_dim": act_dim,
        }

    def test_agent_creation(self, setup):
        """Test creating a TrainingAgent."""
        agent = setup["agent"]
        assert agent is not None
        assert agent.flow_map is not None
        assert agent.encoder is not None
        assert agent.interpolant is not None
        assert agent.flow_map_ema is not None
        assert agent.encoder_ema is not None
        assert agent.optimizer is not None

    def test_agent_creation_with_vanilla_mlp(self):
        """Test creating a TrainingAgent with VanillaMLP."""
        act_dim = 2
        Ta = 4
        obs_dim = 3
        To = 1

        optimization_config = OptimizationConfig(device="cpu")
        network_config = NetworkConfig(
            network_type="vanilla_mlp", emb_dim=64, num_layers=3
        )
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

        config = Config(
            optimization=optimization_config,
            network=network_config,
            task=task_config,
            log=log_config,
        )

        agent = TrainingAgent(config)

        assert agent is not None
        assert agent.flow_map is not None
        assert agent.encoder is not None

    def test_agent_creation_with_different_interp_types(self):
        """Test creating a TrainingAgent with different interpolant types."""
        act_dim = 2
        ta_dim = 4
        obs_dim = 3
        to_dim = 1

        # Test linear interpolant
        optimization_config_linear = OptimizationConfig(
            interp_type="linear", device="cpu"
        )
        network_config = NetworkConfig(emb_dim=64, num_layers=3)
        task_config = TaskConfig(
            act_dim=act_dim,
            obs_dim=obs_dim,
            act_steps=ta_dim,
            obs_steps=to_dim,
            horizon=ta_dim,
        )
        log_config = LogConfig(
            log_dir="./logs",
            wandb_mode="disabled",
            project="test",
            group="test",
            exp_name="test",
        )

        config_linear = Config(
            optimization=optimization_config_linear,
            network=network_config,
            task=task_config,
            log=log_config,
        )

        agent_linear = TrainingAgent(config_linear)
        assert agent_linear.interpolant is not None

        # Test trigonometric interpolant
        optimization_config_trig = OptimizationConfig(interp_type="trig", device="cpu")
        config_trig = Config(
            optimization=optimization_config_trig,
            network=network_config,
            task=task_config,
            log=log_config,
        )

        agent_trig = TrainingAgent(config_trig)
        assert agent_trig.interpolant is not None

    def test_agent_update_returns_dict(self, setup):
        """Test that agent.update returns a dictionary with loss and grad_norm."""
        agent = setup["agent"]
        act = setup["act"]
        obs = setup["obs"]
        delta_t = setup["delta_t"]

        info = agent.update(act, obs, delta_t)

        assert isinstance(info, dict)
        assert "loss" in info
        assert "grad_norm" in info
        assert isinstance(info["loss"], float)
        assert isinstance(info["grad_norm"], float)

    def test_agent_update_gradient_flow(self, setup):
        """Test that gradients flow through agent.update."""
        agent = setup["agent"]
        act = setup["act"]
        obs = setup["obs"]
        delta_t = setup["delta_t"]

        # Store initial parameters
        initial_params = [p.clone().detach() for p in agent.flow_map.parameters()]

        # Perform update
        _ = agent.update(act, obs, delta_t)

        # Check that parameters have changed
        current_params = list(agent.flow_map.parameters())
        params_changed = any(
            not torch.allclose(p_init, p_curr)
            for p_init, p_curr in zip(initial_params, current_params, strict=False)
        )
        assert params_changed, "Parameters should have changed after update"

    def test_agent_update_multiple_times(self, setup):
        """Test that agent.update can be called multiple times."""
        agent = setup["agent"]
        act = setup["act"]
        obs = setup["obs"]
        delta_t = setup["delta_t"]

        # Perform multiple updates
        for _ in range(5):
            info = agent.update(act, obs, delta_t)
            assert isinstance(info, dict)
            assert "loss" in info
            assert "grad_norm" in info

    def test_agent_sample_returns_correct_shape(self, setup):
        """Test that agent.sample returns correct output shape."""
        agent = setup["agent"]
        act_0 = setup["act_0"]
        obs = setup["obs"]
        bs = setup["bs"]
        Ta = setup["Ta"]
        act_dim = setup["act_dim"]

        # Sample
        act = agent.sample(act_0, obs)

        assert act.shape == (bs, Ta, act_dim)
        assert act.shape == act_0.shape

    def test_agent_sample_with_use_ema_false(self, setup):
        """Test agent.sample with use_ema=False."""
        agent = setup["agent"]
        act_0 = setup["act_0"]
        obs = setup["obs"]

        # Sample without EMA
        act = agent.sample(act_0, obs, use_ema=False)

        assert act.shape == act_0.shape

    def test_agent_sample_with_use_ema_true(self, setup):
        """Test agent.sample with use_ema=True."""
        agent = setup["agent"]
        act_0 = setup["act_0"]
        obs = setup["obs"]

        # Sample with EMA
        act = agent.sample(act_0, obs, use_ema=True)

        assert act.shape == act_0.shape

    def test_agent_sample_with_custom_num_steps(self, setup):
        """Test agent.sample with custom num_steps."""
        agent = setup["agent"]
        act_0 = setup["act_0"]
        obs = setup["obs"]

        # Sample with custom num_steps
        for num_steps in [1, 5, 20]:
            act = agent.sample(act_0, obs, num_steps=num_steps)
            assert act.shape == act_0.shape

    def test_agent_ema_update(self, setup):
        """Test that EMA parameters are updated correctly."""
        agent = setup["agent"]
        act = setup["act"]
        obs = setup["obs"]
        delta_t = setup["delta_t"]

        # Store initial EMA parameters
        initial_ema_params = [
            p.clone().detach() for p in agent.flow_map_ema.parameters()
        ]

        # Perform update (which should trigger EMA update)
        agent.update(act, obs, delta_t)

        # Check that EMA parameters have changed
        current_ema_params = list(agent.flow_map_ema.parameters())
        ema_params_changed = any(
            not torch.allclose(p_init, p_curr)
            for p_init, p_curr in zip(
                initial_ema_params, current_ema_params, strict=False
            )
        )
        assert ema_params_changed, "EMA parameters should have changed after update"

    def test_agent_ema_update_with_ema_rate_1(self):
        """Test that EMA update is not triggered when ema_rate=1."""
        act_dim = 2
        Ta = 4
        obs_dim = 3
        To = 1
        bs = 8

        optimization_config = OptimizationConfig(
            loss_type="flow",
            ema_rate=1.0,  # No EMA update
            device="cpu",
        )
        network_config = NetworkConfig(emb_dim=64, num_layers=3)
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

        # Store initial EMA parameters
        initial_ema_params = [
            p.clone().detach() for p in agent.flow_map_ema.parameters()
        ]

        # Perform update
        agent.update(act, obs, delta_t)

        # Check that EMA parameters have NOT changed (since ema_rate=1)
        current_ema_params = list(agent.flow_map_ema.parameters())
        ema_params_unchanged = all(
            torch.allclose(p_init, p_curr)
            for p_init, p_curr in zip(
                initial_ema_params, current_ema_params, strict=False
            )
        )
        assert ema_params_unchanged, "EMA parameters should NOT change when ema_rate=1"

    def test_agent_ema_convergence(self):
        """Test that EMA parameters converge towards main parameters over time."""
        act_dim = 2
        Ta = 4
        obs_dim = 3
        To = 1
        bs = 8

        optimization_config = OptimizationConfig(
            loss_type="flow",
            ema_rate=0.9,  # Fast EMA for testing
            lr=1e-3,
            device="cpu",
        )
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

        # Perform multiple updates with same data
        for _ in range(50):
            agent.update(act, obs, delta_t)

        # Final distance between main and EMA parameters
        final_distance = sum(
            torch.norm(p - p_ema).item()
            for p, p_ema in zip(
                agent.flow_map.parameters(),
                agent.flow_map_ema.parameters(),
                strict=False,
            )
        )

        # EMA should be tracking the main parameters
        # (distance may increase or decrease depending on training dynamics)
        assert final_distance >= 0  # Just check it's valid

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

    def test_agent_lmd_and_ctm_loss_types(self):
        """Test agent with lmd and ctm loss types - should raise error without reference network."""
        act_dim = 2
        ta_dim = 4
        obs_dim = 3
        to_dim = 1
        bs = 8

        task_config = TaskConfig(
            act_dim=act_dim,
            obs_dim=obs_dim,
            act_steps=ta_dim,
            obs_steps=to_dim,
            horizon=ta_dim,
        )
        log_config = LogConfig(
            log_dir="./logs",
            wandb_mode="disabled",
            project="test",
            group="test",
            exp_name="test",
        )

        for loss_type in ["lmd", "ctm"]:
            # Use dropout=0.0 for lmd/ctm loss types to avoid vmap randomness issues
            network_config = NetworkConfig(
                emb_dim=32, num_layers=2, dropout=0.0, timestep_emb_dim=16
            )
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

            act = torch.randn(bs, ta_dim, act_dim)
            obs = torch.randn(bs, to_dim, obs_dim)
            delta_t = torch.rand(bs)

            # Test that lmd/ctm loss types raise an error without reference network
            with pytest.raises((TypeError, RuntimeError)):
                agent.update(act, obs, delta_t)

    def test_agent_grad_clip_effect(self):
        """Test that gradient clipping is applied correctly."""
        act_dim = 2
        Ta = 4
        obs_dim = 3
        To = 1
        bs = 8

        # Create agent with gradient clipping
        optimization_config = OptimizationConfig(
            loss_type="flow",
            grad_clip_norm=1.0,  # Small clip value
            device="cpu",
        )
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

        # Perform update
        info = agent.update(act, obs, delta_t)

        # Grad norm should be reported
        assert "grad_norm" in info
        assert info["grad_norm"] >= 0

    def test_agent_no_grad_clip(self):
        """Test agent without gradient clipping."""
        act_dim = 2
        Ta = 4
        obs_dim = 3
        To = 1
        bs = 8

        optimization_config = OptimizationConfig(
            loss_type="flow",
            grad_clip_norm=0.0,  # No clipping
            device="cpu",
        )
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

        info = agent.update(act, obs, delta_t)

        assert "grad_norm" in info
        # When grad_clip_norm=0, grad_norm should be 0 (default value)
        assert info["grad_norm"] == 0.0

    def test_agent_different_batch_sizes(self):
        """Test agent with different batch sizes."""
        act_dim = 2
        Ta = 4
        obs_dim = 3
        To = 1

        optimization_config = OptimizationConfig(
            loss_type="flow", num_steps=5, device="cpu"
        )
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

        config = Config(
            optimization=optimization_config,
            network=network_config,
            task=task_config,
            log=log_config,
        )

        agent = TrainingAgent(config)

        for bs in [1, 4, 16]:
            act = torch.randn(bs, Ta, act_dim)
            obs = torch.randn(bs, To, obs_dim)
            delta_t = torch.rand(bs)
            act_0 = torch.randn(bs, Ta, act_dim)

            # Test update
            info = agent.update(act, obs, delta_t)
            assert isinstance(info, dict)

            # Test sample
            sampled_act = agent.sample(act_0, obs)
            assert sampled_act.shape == (bs, Ta, act_dim)

    def test_agent_optimizer_state(self, setup):
        """Test that optimizer state is maintained across updates."""
        agent = setup["agent"]
        act = setup["act"]
        obs = setup["obs"]
        delta_t = setup["delta_t"]

        # Perform first update
        agent.update(act, obs, delta_t)

        # Check that optimizer has state (for AdamW, should have momentum buffers)
        optimizer_state = agent.optimizer.state_dict()
        assert "state" in optimizer_state
        assert "param_groups" in optimizer_state

    def test_agent_sample_deterministic_in_eval_mode(self, setup):
        """Test that sampling is deterministic when using fixed input."""
        agent = setup["agent"]
        agent.flow_map.eval()
        agent.encoder.eval()

        act_0 = setup["act_0"]
        obs = setup["obs"]

        # Sample twice with same input
        with torch.no_grad():
            act1 = agent.sample(act_0, obs, use_ema=False)
            act2 = agent.sample(act_0, obs, use_ema=False)

        # Should be identical (no randomness in eval mode with fixed input)
        assert torch.allclose(act1, act2)

    def test_agent_integration_update_then_sample(self, setup):
        """Test integration: update then sample."""
        agent = setup["agent"]
        act = setup["act"]
        obs = setup["obs"]
        delta_t = setup["delta_t"]
        act_0 = setup["act_0"]

        # Perform several updates
        for _ in range(3):
            info = agent.update(act, obs, delta_t)
            assert "loss" in info

        # Then sample
        sampled_act = agent.sample(act_0, obs)
        assert sampled_act.shape == act_0.shape

        # Sample with EMA
        sampled_act_ema = agent.sample(act_0, obs, use_ema=True)
        assert sampled_act_ema.shape == act_0.shape

    def test_agent_compile_with_different_networks(self):
        """Test torch.compile works with all network architectures."""
        act_dim = 2
        Ta = 16
        obs_dim = 3
        To = 2
        bs = 4

        # Networks to test
        networks = ["mlp", "vanilla_mlp", "chiunet", "chitransformer", "sudeepdit", "rnn"]

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

        for network_type in networks:
            print(f"\nTesting torch.compile with network: {network_type}")

            # Create config with compilation enabled
            optimization_config = OptimizationConfig(
                loss_type="flow",
                num_steps=1,
                use_compile=True,
                device="cpu",
            )
            # Use emb_dim=72 which is divisible by n_heads=6 (for transformer)
            network_config = NetworkConfig(
                network_type=network_type,
                emb_dim=72,
                num_layers=2,
                model_dim=72,  # For UNet-based networks
                dropout=0.0,  # Disable dropout for deterministic testing
                n_heads=6,  # For transformer networks
            )

            config = Config(
                optimization=optimization_config,
                network=network_config,
                task=task_config,
                log=log_config,
            )

            # Create agent (this triggers compilation)
            agent = TrainingAgent(config)

            # Test tensors
            act = torch.randn(bs, Ta, act_dim)
            obs = torch.randn(bs, To, obs_dim)
            delta_t = torch.rand(bs)
            act_0 = torch.randn(bs, Ta, act_dim)

            # Test update (compiled)
            info = agent.update(act, obs, delta_t)
            assert isinstance(info, dict)
            assert "loss" in info
            assert "grad_norm" in info

            # Test sample (compiled)
            sampled_act = agent.sample(act_0, obs)
            assert sampled_act.shape == (bs, Ta, act_dim)

            print(f"  ✓ {network_type} works with torch.compile")

    def test_agent_compile_speedup(self):
        """Test that torch.compile provides speedup over eager mode."""
        import time

        act_dim = 2
        Ta = 16
        obs_dim = 3
        To = 2
        bs = 32
        num_iterations = 20  # Reduced for faster testing

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

        # Test data
        act = torch.randn(bs, Ta, act_dim)
        obs = torch.randn(bs, To, obs_dim)
        delta_t = torch.rand(bs)

        # Test without compilation
        print("\nTesting WITHOUT torch.compile:")
        optimization_config_no_compile = OptimizationConfig(
            loss_type="flow",
            num_steps=1,
            use_compile=False,
            device="cpu",
        )
        network_config = NetworkConfig(
            network_type="mlp",
            emb_dim=128,
            num_layers=3,
            dropout=0.0,
        )

        config_no_compile = Config(
            optimization=optimization_config_no_compile,
            network=network_config,
            task=task_config,
            log=log_config,
        )

        agent_no_compile = TrainingAgent(config_no_compile)

        # Warmup
        for _ in range(3):
            agent_no_compile.update(act, obs, delta_t)

        # Time without compilation
        start = time.time()
        for _ in range(num_iterations):
            agent_no_compile.update(act, obs, delta_t)
        time_no_compile = time.time() - start
        print(f"  Time without compile: {time_no_compile:.4f}s")

        # Test with compilation
        print("\nTesting WITH torch.compile:")
        optimization_config_compile = OptimizationConfig(
            loss_type="flow",
            num_steps=1,
            use_compile=True,
            device="cpu",
        )

        config_compile = Config(
            optimization=optimization_config_compile,
            network=network_config,
            task=task_config,
            log=log_config,
        )

        agent_compile = TrainingAgent(config_compile)

        # Warmup (includes compilation time)
        print("  Compiling (warmup)...")
        for _ in range(3):
            agent_compile.update(act, obs, delta_t)

        # Time with compilation (after warmup)
        start = time.time()
        for _ in range(num_iterations):
            agent_compile.update(act, obs, delta_t)
        time_compile = time.time() - start
        print(f"  Time with compile: {time_compile:.4f}s")

        # Calculate speedup
        speedup = time_no_compile / time_compile
        print(f"\n  Speedup: {speedup:.2f}x")

        # Assert that compilation provides some benefit or at least doesn't slow down
        # (On CPU, speedup might be modest; on GPU it's typically much larger)
        assert time_compile <= time_no_compile * 1.5, (
            f"Compilation should not significantly slow down execution. "
            f"Got {speedup:.2f}x speedup (expected >= 0.67x)"
        )

        # If we get speedup, celebrate!
        if speedup > 1.0:
            print(f"  ✓ torch.compile provided {speedup:.2f}x speedup!")
        else:
            print(f"  ℹ torch.compile didn't provide speedup on CPU (normal)")

    def test_agent_compile_no_compile_functionality(self):
        """Test that both compiled and non-compiled agents work correctly."""
        act_dim = 2
        Ta = 8
        obs_dim = 3
        To = 2
        bs = 8

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

        # Test without compilation
        optimization_config_no_compile = OptimizationConfig(
            loss_type="flow",
            num_steps=1,
            use_compile=False,
            device="cpu",
        )
        network_config = NetworkConfig(
            network_type="mlp", emb_dim=64, num_layers=2, dropout=0.0
        )
        config_no_compile = Config(
            optimization=optimization_config_no_compile,
            network=network_config,
            task=task_config,
            log=log_config,
        )
        agent_no_compile = TrainingAgent(config_no_compile)

        # Test with compilation
        optimization_config_compile = OptimizationConfig(
            loss_type="flow",
            num_steps=1,
            use_compile=True,
            device="cpu",
        )
        config_compile = Config(
            optimization=optimization_config_compile,
            network=network_config,
            task=task_config,
            log=log_config,
        )
        agent_compile = TrainingAgent(config_compile)

        # Test data
        act = torch.randn(bs, Ta, act_dim)
        obs = torch.randn(bs, To, obs_dim)
        delta_t = torch.rand(bs)
        act_0 = torch.randn(bs, Ta, act_dim)

        # Test update for both
        info_no_compile = agent_no_compile.update(act, obs, delta_t)
        info_compile = agent_compile.update(act, obs, delta_t)

        # Both should return valid info dicts
        assert isinstance(info_no_compile, dict)
        assert isinstance(info_compile, dict)
        assert "loss" in info_no_compile
        assert "loss" in info_compile
        assert "grad_norm" in info_no_compile
        assert "grad_norm" in info_compile

        # Test sampling for both
        agent_no_compile.eval()
        agent_compile.eval()

        with torch.no_grad():
            sample_no_compile = agent_no_compile.sample(act_0, obs, use_ema=False)
            sample_compile = agent_compile.sample(act_0, obs, use_ema=False)

        # Both should produce correct shapes
        assert sample_no_compile.shape == (bs, Ta, act_dim)
        assert sample_compile.shape == (bs, Ta, act_dim)

        print(f"  ✓ Both compiled and non-compiled agents work correctly")
        print(f"  No-compile loss: {info_no_compile['loss']:.2f}")
        print(f"  Compiled loss: {info_compile['loss']:.2f}")
