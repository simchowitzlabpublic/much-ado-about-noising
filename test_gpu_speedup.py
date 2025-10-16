"""Test torch.compile speedup on GPU.

Temporary test file to benchmark GPU speedup.
"""

import time

import torch

from mip.agent import TrainingAgent
from mip.config import Config, LogConfig, NetworkConfig, OptimizationConfig, TaskConfig


def test_gpu_speedup():
    """Test that torch.compile provides speedup on GPU."""
    act_dim = 2
    Ta = 16
    obs_dim = 3
    To = 2
    bs = 32
    num_iterations = 50  # More iterations for better timing

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

    # Test data on GPU
    device = "cuda"
    act = torch.randn(bs, Ta, act_dim, device=device)
    obs = torch.randn(bs, To, obs_dim, device=device)
    delta_t = torch.rand(bs, device=device)

    # Test without compilation
    print("\n" + "=" * 60)
    print("Testing WITHOUT torch.compile on GPU:")
    print("=" * 60)
    optimization_config_no_compile = OptimizationConfig(
        loss_type="flow",
        num_steps=1,
        use_compile=False,
        device=device,
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
    print("Warming up (no compile)...")
    for _ in range(5):
        agent_no_compile.update(act, obs, delta_t)
    torch.cuda.synchronize()

    # Time without compilation
    print(f"Timing {num_iterations} iterations...")
    start = time.time()
    for _ in range(num_iterations):
        agent_no_compile.update(act, obs, delta_t)
    torch.cuda.synchronize()
    time_no_compile = time.time() - start
    print(f"Time without compile: {time_no_compile:.4f}s")
    print(f"Average per iteration: {time_no_compile/num_iterations*1000:.2f}ms")

    # Test with compilation
    print("\n" + "=" * 60)
    print("Testing WITH torch.compile on GPU:")
    print("=" * 60)
    optimization_config_compile = OptimizationConfig(
        loss_type="flow",
        num_steps=1,
        use_compile=True,
        device=device,
    )

    config_compile = Config(
        optimization=optimization_config_compile,
        network=network_config,
        task=task_config,
        log=log_config,
    )

    agent_compile = TrainingAgent(config_compile)

    # Warmup (includes compilation time)
    print("Warming up + compiling...")
    for _ in range(5):
        agent_compile.update(act, obs, delta_t)
    torch.cuda.synchronize()

    # Time with compilation (after warmup)
    print(f"Timing {num_iterations} iterations...")
    start = time.time()
    for _ in range(num_iterations):
        agent_compile.update(act, obs, delta_t)
    torch.cuda.synchronize()
    time_compile = time.time() - start
    print(f"Time with compile: {time_compile:.4f}s")
    print(f"Average per iteration: {time_compile/num_iterations*1000:.2f}ms")

    # Calculate speedup
    speedup = time_no_compile / time_compile
    print("\n" + "=" * 60)
    print(f"SPEEDUP: {speedup:.2f}x")
    print("=" * 60)

    if speedup > 1.0:
        print(f"✓ torch.compile provided {speedup:.2f}x speedup on GPU!")
    else:
        print(f"⚠ torch.compile was {1/speedup:.2f}x slower")

    # Show memory usage
    print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

    return speedup


if __name__ == "__main__":
    speedup = test_gpu_speedup()
