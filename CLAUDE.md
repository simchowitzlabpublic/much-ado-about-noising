# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MIP (Minimum Iterative Policy) is a PyTorch-based reinforcement learning framework focused on behavior cloning with flow matching. The project integrates with the Robomimic dataset and environment for training robotic manipulation policies.

## Commands

### Setup
```bash
# Install dependencies
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

### Training
```bash
# Train with default configuration
python examples/train_robomimic.py

# Train with custom configuration
python examples/train_robomimic.py task=lift_image  # For image-based observations
python examples/train_robomimic.py optimization.batch_size=32  # Override batch size
python examples/train_robomimic.py network=mlp  # Select network architecture
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest mip/agent_test.py

# Run tests with verbose output
pytest -v

# Run tests for a specific network
pytest mip/networks/mlp_test.py
```

### Code Quality
```bash
# Format code
ruff format .

# Lint code
ruff check .

# Fix linting issues automatically
ruff check --fix .
```

## Architecture

### Core Components

1. **Agent System** (`mip/agent.py`)
   - `TrainingAgent`: Manages flow matching model training with EMA updates
   - Handles observation encoding, flow map learning, and sampling
   - Automatically selects encoder based on network config

2. **Flow Matching Framework**
   - `mip/flow_map.py`: Neural network that learns the velocity field
   - `mip/interpolant.py`: Interpolation between noise and data distributions
   - `mip/samplers.py`: Various sampling strategies (Euler, Heun, etc.)
   - `mip/losses.py`: Loss functions for flow matching training

3. **Network Architecture** (`mip/networks/`)
   - All networks inherit from `BaseNetwork` and follow a consistent interface
   - Networks take (x, s, t, condition) and return (action, scalar) tuple
   - Available networks: `mlp`, `vanilla_mlp` (more coming: `chitransformer`, `chiunet`, `dit`, `jannerunet`, `rnn`)
   - Networks are configured via `network_utils.get_network()`

4. **Encoders** (`mip/encoders.py`)
   - `IdentityEncoder`: Pass-through for state observations
   - `MLPEncoder`: Multi-layer encoding for state observations
   - `MultiImageObsEncoder`: CNN-based encoder for image observations
   - Encoder selection is automatic based on `obs_type` and `encoder_type`

5. **Dataset Pipeline** (`mip/datasets/`)
   - `robomimic_dataset.py`: Handles Robomimic HDF5 dataset loading
   - Supports both state and image observations
   - Includes action/observation normalization
   - Can load from HuggingFace or local paths

6. **Environment Wrappers** (`mip/envs/`)
   - `robomimic_env.py`: Base environment wrapper
   - `robomimic_lowdim_wrapper.py`: State-based observations
   - `robomimic_image_wrapper.py`: Image-based observations with frame stacking

7. **Configuration System**
   - Uses Hydra for hierarchical configuration
   - Config files in `examples/configs/`
   - Main config structure: `task`, `network`, `optimization`, `log`
   - Config dataclasses in `mip/config.py`

### Training Pipeline

The training flow (`examples/train_robomimic.py`):
1. Loads Hydra configuration
2. Creates vectorized environments
3. Initializes dataset with normalizers
4. Creates `TrainingAgent` with flow map and encoder
5. Runs training loop with:
   - Batch sampling from dataloader
   - Flow matching loss computation
   - Gradient updates with optional EMA
   - Periodic evaluation on environment
   - Metric logging to Weights & Biases

### Key Design Patterns

- **Modular Networks**: Networks (`mip/networks/`) are separate from agents
- **Encoder-Decoder**: Separate observation encoder from action decoder (flow map)
- **Dual Output**: Networks output both action predictions and scalar values for enhanced learning
- **Normalization**: All observations/actions normalized during training via dataset normalizers
- **Vectorized Environments**: Supports parallel environment execution with `make_vec_env()`
- **Configurable Sampling**: Multiple ODE solvers for inference
- **EMA Updates**: Exponential moving average for stable training

## Configuration Structure

The configuration uses Hydra with these main groups:
- `task/`: Environment and dataset settings (lift, lift_image, etc.)
- `network/`: Neural network architectures (mlp, vanilla_mlp, etc.)
- `optimization/`: Training hyperparameters
- `log/`: Logging and evaluation settings

Key configuration parameters:
- `task.obs_type`: "state" or "image" (determines encoder type)
- `task.horizon`: Action prediction horizon (typically 10)
- `task.obs_steps`: Observation history length (typically 2)
- `task.act_steps`: Action execution steps (typically 8)
- `optimization.loss_type`: Flow matching loss variant ("flow" default)
- `optimization.gradient_steps`: Total training steps
- `optimization.batch_size`: Training batch size
- `network.network_type`: Network architecture selection
- `network.emb_dim`: Embedding dimension for networks
- `network.num_layers`: Number of network layers