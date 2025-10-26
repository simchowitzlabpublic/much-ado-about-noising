# Miniumm Flow Policies Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the code for the paper **"Much Ado About Noising: Dispelling the Myths of Generative Robotic Control"**. MIP is a PyTorch-based framework for behavior cloning with flow matching and related generative models, incorporating best practices from diffusion model training.

## Features

- 🧩 **Clean & Modular**: Composable components for losses, samplers, networks, and encoders
- ⚡ **Fast**: Optimized with torch.compile and CUDA graphs for maximum throughput
- 📊 **Best Practices**: EMA, warmup scheduling, auto-resume, and proven training techniques
- 🎯 **Diverse Algorithms**: Support for flow matching, consistency models, shortcut models, and regression
- 🤖 **Robot-Ready**: Pre-configured for Robomimic, Kitchen, and PushT tasks

## Documentation

- 📚 **[Full Documentation](docs/)** - Comprehensive guides and API reference
- 🚀 **[Quick Start](docs/getting-started/quick_start.md)** - Get running in minutes
- 🏗️ **[Architecture Guide](docs/getting-started/design.md)** - Framework internals
- 🔧 **[Configuration](docs/getting-started/configuration.md)** - Detailed configuration guide
- 🛠️ **[Development Guides](docs/development/)** - Extend MIP with custom tasks and methods
- 💡 **[FAQ](docs/help/faq.md)** - Frequently asked questions
- 🔍 **[Troubleshooting](docs/help/troubleshooting.md)** - Common issues and solutions

## Installation

```bash
uv sync
# install robomimic dependencies
uv sync --extras robomimic
# install kitchen dependencies
uv sync --extras kitchen
```

## Quick Start

### Training

```bash
# Train Robomimic (state observations)
uv run examples/train_robomimic.py \
    task=lift_ph_state \
    network=chiunet \
    optimization.loss_type=flow \
    log.wandb_mode=online

# Train Robomimic (image observations)
uv run examples/train_robomimic.py \
    task=lift_ph_image \
    network=chiunet \
    optimization.batch_size=256

# Train Kitchen
uv run examples/train_kitchen.py task=kitchen_state

# Train PushT
uv run examples/train_pusht.py task=pusht_state
```

### Evaluation

```bash
# Evaluate trained model
uv run examples/train_robomimic.py \
    mode=eval \
    optimization.model_path="/path/to/checkpoint.pt"
```

### Configuration

```bash
# Debug mode (quick test)
uv run examples/train_robomimic.py -cn exps/debug.yaml

# Override parameters
uv run examples/train_robomimic.py task.horizon=16 optimization.batch_size=512

# Multi-run (sweep multiple configs)
uv run examples/train_robomimic.py task=lift_ph_state,can_ph_state --multirun
```

See the [Configuration Guide](docs/getting-started/configuration.md) for more details.

## Supported Training Objectives

MIP supports multiple training objectives:

- **Flow Matching** (`flow`): Standard continuous normalizing flow
- **Regression** (`regression`): Direct supervised learning baseline
- **MIP** (`mip`): Minimum Iterative Policy with two-step sampling
- **TSD** (`tsd`): Two-Stage Denoising
- **CTM** (`ctm`): Consistency Trajectory Model
- **PSD** (`psd`): Progressive Self-Distillation
- **LSD** (`lsd`): Lagrangian Self-Distillation
- **ESD** (`esd`): Euler Self-Distillation
- **MF** (`mf`): Mean Flow

## Known Issues

- CUDA graphs not supported for image-based tasks (requires static tensor shapes)
- Kitchen tasks require MuJoCo 3.1.6: `uv pip install "mujoco==3.1.6"`

See [Troubleshooting](docs/help/troubleshooting.md) for more issues and solutions.

## Citation

```

```
