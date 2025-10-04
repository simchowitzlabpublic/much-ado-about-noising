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
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest mip/agent_test.py
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

2. **Flow Matching Framework**
   - `mip/flow_map.py`: Neural network that learns the velocity field
   - `mip/interpolant.py`: Interpolation between noise and data distributions
   - `mip/samplers.py`: Various sampling strategies (Euler, Heun, etc.)
   - `mip/losses.py`: Loss functions for flow matching training

3. **Dataset Pipeline** (`mip/datasets/`)
   - `robomimic_dataset.py`: Handles Robomimic HDF5 dataset loading
   - Supports both state and image observations
   - Includes action/observation normalization

4. **Environment Wrappers** (`mip/envs/`)
   - `robomimic_env.py`: Base environment wrapper
   - `robomimic_lowdim_wrapper.py`: State-based observations
   - `robomimic_image_wrapper.py`: Image-based observations with frame stacking

5. **Configuration System**
   - Uses Hydra for hierarchical configuration
   - Config files in `examples/configs/`
   - Main config structure: `task`, `network`, `optimization`, `log`

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
- **Normalization**: All observations/actions normalized during training
- **Vectorized Environments**: Supports parallel environment execution
- **Configurable Sampling**: Multiple ODE solvers for inference

## Configuration Structure

The configuration uses Hydra with these main groups:
- `task/`: Environment and dataset settings (lift, lift_image, etc.)
- `network/`: Neural network architectures (mlp, etc.)
- `optimization/`: Training hyperparameters
- `log/`: Logging and evaluation settings

Key configuration parameters:
- `task.obs_type`: "state" or "image"
- `task.horizon`: Action prediction horizon
- `optimization.loss_type`: Flow matching loss variant
- `optimization.gradient_steps`: Total training steps