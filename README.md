<div align="center">

# Minimum Flow Policies Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2406.12345-b31b1b.svg)](https://arxiv.org/abs/2512.01809)

<br />

[**Project Website**](https://simchowitzlabpublic.github.io/much-ado-about-noising-project/) •
[**Documentation**](https://simchowitzlabpublic.github.io/much-ado-about-noising/) •
[**Dataset**](https://huggingface.co/datasets/ChaoyiPan/mip-dataset) •
[**Checkpoints**](https://huggingface.co/ChaoyiPan/mip-checkpoints)

</div>

This repository contains the code for the paper **"Much Ado About Noising: Dispelling the Myths of Generative Robotic Control"**. This repository is a PyTorch-based framework for behavior cloning with flow matching and related generative models, incorporating best practices from diffusion model training.

## Features

- 🧩 **Clean & Modular**: Composable components for losses, samplers, networks, and encoders
- ⚡ **Fast**: Optimized with torch.compile and CUDA graphs for maximum throughput
- 📊 **Best Practices**: EMA, warmup scheduling, auto-resume, and proven training techniques
- 🎯 **Diverse Algorithms**: Support for flow matching, consistency models, shortcut models, and regression
- 🤖 **Robot-Ready**: Pre-configured for Robomimic, Kitchen, and PushT tasks

## Documentation

Please refer to the [documentation](https://simchowitzlabpublic.github.io/much-ado-about-noising/) for more details.

## Installation

```bash
uv sync
# install for development
uv sync --extra dev
```

## Quick Start

### Training

```bash
# on headless machine
export MUJOCO_GL=egl
# on ubuntu machine without mujoco installed
sudo apt-get install -y libglew-dev libosmesa6-dev patchelf
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

You can download checkpoints from [Hugging Face](https://huggingface.co/ChaoyiPan/mip-checkpoints).

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

This repository supports multiple training objectives:

- **Flow Matching** (`flow`): Standard continuous normalizing flow
- **Regression** (`regression`): Direct supervised learning baseline
- **MIP** (`mip`): Minimum Iterative Policy with two-step sampling
- **TSD** (`tsd`): Two-Stage Denoising
- **CTM** (`ctm`): Consistency Trajectory Model
- **PSD** (`psd`): Progressive Self-Distillation
- **LSD** (`lsd`): Lagrangian Self-Distillation
- **ESD** (`esd`): Euler Self-Distillation
- **MF** (`mf`): Mean Flow

## Porting MIP to Your Pipeline

If you are interested in porting MIP to your own pipeline, you can refer to [examples/pi0_pytorch.diff](examples/pi0_pytorch.diff). This file demonstrates how to modify the loss function and integrator of a standard flow policy to implement MIP.

> **⚠️ Note on Reproducibility** > Since we have migrated to the latest versions of dependencies (e.g., robomimic) and performed significant code cleanup, the results obtained from this repository may not *exactly* replicate the specific numbers reported in the paper. However, the overall performance trends and conclusions remain the same. Original results is obtained with [cleandiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) robomimic environment and dataset.

## Known Issues

- CUDA graphs not supported for image-based tasks (requires static tensor shapes)
- Kitchen tasks require MuJoCo 3.1.6: `uv pip install "mujoco==3.1.6"`

See [Troubleshooting](docs/help/troubleshooting.md) for more issues and solutions.

## Citation

```
@article{pan2025adonoisingdispellingmyths,
      title={Much Ado About Noising: Dispelling the Myths of Generative Robotic Control},
      author={Chaoyi Pan and Giri Anantharaman and Nai-Chieh Huang and Claire Jin and Daniel Pfrommer and Chenyang Yuan and Frank Permenter and Guannan Qu and Nicholas Boffi and Guanya Shi and Max Simchowitz},
      year={2025},
      eprint={2512.01809},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2512.01809},
}
```
