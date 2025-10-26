# Frequently Asked Questions

## General Questions

### Which training method should I use?

- **Regression** (`regression`): Simplest baseline, give strong performance on most tasks with correct architecture.
- **MIP** (`mip`): Minimum Iterative Policy, reserve most of the benefits of flow matching while inference faster.
- **Flow matching** (`flow`): Good default choice, best performance on most tasks.
- **CTM/PSD/LSD** (`ctm`/`psd`/`lsd`): Self distillation, slower to train. Useful for learning full aciton distribution.

See the paper for detailed comparisons and recommendations.

### Which network architecture should I use?

- **State observations**: `mlp` (simple, fast) or `chiunet` (better for long horizons)
- **Image observations**: `chiunet` (proven architecture) or `chitransformer` (for complex scenes)
- **Long horizons** (>16): `chitransformer` or `sudeepdit`
- **Limited compute**: `mlp`

## Installation & Setup

### Do I need to use UV package manager?

While UV is recommended for its speed, you can use pip instead:
```bash
pip install -e .
```

### Can I pause and resume training?

Yes! MIP supports auto-resume:
```bash
# Training will automatically resume from last checkpoint if it exists
uv run examples/train_robomimic.py task=lift_ph_state optimization.auto_resume=true
```

## Performance

### How can I speed up training?

1. **Enable torch.compile** to speed up training:
   ```bash
   uv run examples/train_robomimic.py optimization.use_compile=true
   ```

2. **Use CUDA graphs** to reduce cpu overhead (state observations only):
   ```bash
   uv run examples/train_robomimic.py optimization.use_cudagraphs=true
   ```

You can use both to get the best performance.
