---
layout: home

hero:
  name: "Much Ado About Noising: Dispelling the Myths of Generative Robotic Control"
  text: "Minimum Implementation of flow-based policies"
  tagline: A PyTorch framework for behavior cloning with flow matching and generative models
  actions:
    - theme: brand
      text: Get Started
      link: /getting-started/quick_start
    - theme: alt
      text: View on GitHub
      link: https://github.com/ChaoyiPan/mip

features:
  - icon: ⚡
    title: Fast & Efficient
    details: Optimized with torch.compile and CUDA graphs for maximum training throughput
  - icon: 🧩
    title: Modular Design
    details: Clean, composable architecture with separate components for losses, samplers, networks, and encoders
  - icon: 🎯
    title: Diverse Algorithms
    details: Support for flow matching, consistency models, flow map models, shortcut models, and regression
  - icon: 🤖
    title: Robot Learning Ready
    details: Pre-configured for Robomimic, Kitchen, and PushT tasks with state and image observations
  - icon: 📊
    title: Best Practices
    details: SoTA architectures and training recipes from flow training.
  - icon: 🔬
    title: Research-Grade
    details: Based on "Much Ado About Noising - Dispelling the Myths of Generative Robotic Control"
---

## Quick Example

```bash
# Install dependencies
uv sync --extras robomimic

# Train a flow matching policy on Robomimic Lift task
uv run examples/train_robomimic.py \
    task=lift_ph_state \
    network=chiunet \
    optimization.loss_type=flow \
    log.wandb_mode=online
```

## Key Features

### Multiple Training Objectives

MIP supports a variety of training objectives for generative behavior cloning:

- **Flow Matching** (`flow`): Standard continuous normalizing flow objective
- **Regression** (`regression`): Direct supervised learning baseline with the same architecture as flow matching.
- **MIP** (`mip`): Minimum Iterative Policy with two-step sampling (from Much Ado About Noising)
- **TSD** (`tsd`): Two-Stage Denoising (from Much Ado About Noising)
- **CTM** (`ctm`): Consistency Trajectory Model
- **PSD** (`psd`): Progressive Self-Distillation
- **LSD** (`lsd`): Lagrangian Self-Distillation (only support differentiable networks)

### Flexible Network Architectures

Choose from multiple proven architectures:

- **MLP**: Enhanced MLP with Film conditioning, residual connection and layer normalization.
- **ChiUNet**: U-Net from Diffusion Policy (Chi et al.), good for smooth control tasks.
- **ChiTransformer**: Transformer architecture from Diffusion Policy, more expressive than ChiUNet but less stable.
- **SudeepDiT**: Diffusion Transformer (DiT) architecture, contains best practices from [Sudeeep et al.](https://dit-policy.github.io/)
- **RNN**: LSTM/GRU-based recurrent networks, only for baseline purposes.

### Comprehensive Task Support

Pre-configured environments and datasets:

- **Robomimic**: Manipulation tasks (Lift, Can, Square, Transport, Tool Hang)
- **Kitchen**: Multi-task kitchen environment
- **PushT**: 2D pushing task with multiple observation modalities

## What's Next?

- [Quick Start Guide](/getting-started/quick_start) - Get up and running in minutes
- [Architecture & Design](/getting-started/design) - Understand the framework internals
- [Adding a New Task](/development/add_new_task) - Extend MIP to your own environments

## Citation

If you use this repository in your research, please cite:

```bibtex
```
