# Troubleshooting

This page covers common issues and their solutions when working with MIP.

### CUDA Graphs Not Working

**Problem**: `use_cudagraphs=true` has no effect or crashes

**Reasons**:

1. **Image observations**: CUDA graphs require static tensor shapes, not supported for image observations
   - Solution: Use state observations or disable CUDA graphs

2. **Dynamic shapes in network**: Some networks may have dynamic computation
   - Solution: Use simpler networks like MLP

3. **Inconsistent batch sizes**: DataLoader must have `drop_last=True`
   - Already set in training scripts, verify if using custom script

## Dataset Issues

### Dataset Not Found

**Problem**: `FileNotFoundError: Dataset not found`

**Solutions**:

1. **For HuggingFace datasets**: Ensure you're connected to internet and have huggingface_hub installed

2. **For local datasets**: Download and process dataset first:
   ```bash
   uv run python examples/process_robomimic_dataset.py --skip_upload
   ```

3. **Override dataset path**:
   ```bash
   uv run examples/train_robomimic.py +task.dataset_path=/path/to/dataset.hdf5
   ```
