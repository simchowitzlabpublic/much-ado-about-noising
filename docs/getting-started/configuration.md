# Configuration Guide

This repository uses [Hydra](https://hydra.cc/) for hierarchical configuration management. This guide explains how to configure training runs.

## Configuration Structure

```
examples/configs/
├── main.yaml              # Main config with defaults
├── task/                  # Environment and dataset configs
│   ├── lift_ph_state.yaml
│   ├── lift_ph_image.yaml
│   └── ...
├── network/               # Network architecture configs
│   ├── mlp.yaml
│   ├── chiunet.yaml
│   └── ...
├── optimization/          # Training hyperparameters
│   └── default.yaml
├── log/                   # Logging and evaluation
│   └── default.yaml
├── launcher/              # Multi-run launchers
│   ├── basic.yaml
│   └── cluster.yaml
└── exps/                  # Experiment presets
    └── debug.yaml
```

## Quick Start

### Basic Usage

```bash
# Use default configuration
uv run examples/train_robomimic.py

# Select different task and network
uv run examples/train_robomimic.py task=lift_ph_state network=chiunet

# Override parameters
uv run examples/train_robomimic.py task.horizon=16 optimization.batch_size=256

# Training with specific loss type
uv run examples/train_robomimic.py optimization.loss_type=flow

# Image-based task
uv run examples/train_robomimic.py task=lift_ph_image network=chiunet

# Debug mode (quick test)
uv run examples/train_robomimic.py -cn exps/debug.yaml

# Multi-run (sweep multiple configs)
uv run examples/train_robomimic.py task=lift_ph_state,can_ph_state --multirun
```

## Configuration Groups

### Task Configuration

**Key Parameters**:
- `obs_type`: "state" or "image" (determines encoder)
- `obs_steps`: Number of observations to stack (typically 2)
- `act_steps`: Actions executed per prediction (typically 8)
- `horizon`: Total action sequence length (typically 10-16)
- `abs_action`: true for absolute actions, false for delta

### Network Configuration

**Key Parameters**:
- `network_type`: Architecture selection
- `emb_dim`: Embedding dimension (512 typical)
- `num_layers`: Network depth
- `num_encoder_layers`: 0 for identity, >0 for MLP encoder

**Available Networks**:
- `mlp`: Multi-layer perceptron
- `vanilla_mlp`: Simple MLP baseline
- `chiunet`: U-Net from Diffusion Policy
- `jannerunet`: U-Net from Decision Diffuser
- `chitransformer`: Transformer architecture
- `sudeepdit`: Diffusion Transformer (DiT)
- `rnn`: LSTM/GRU recurrent network

### Optimization Configuration

**Key Parameters**:
- `loss_type`: Training objective
- `batch_size`: 1024 for state, 256 for image
- `gradient_steps`: Total training iterations
- `ema_rate`: EMA decay (0.995 typical)
- `use_compile`: Enable torch.compile
- `use_cudagraphs`: Enable CUDA graphs (state only)

## Next Steps

- [Architecture & Design](design.md) - Understand the framework
- [Quick Start](quick_start.md) - Get started with training
- [Troubleshooting](../help/troubleshooting.md) - Common issues
