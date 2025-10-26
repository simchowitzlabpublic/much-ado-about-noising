# Instructions for Porting a New Task

This guide provides step-by-step instructions for adding a new task to the MIP framework. Follow these steps to ensure consistency and completeness.

## Overview

Porting a new task involves four main steps:
1. **Dataset Preparation**: Download/prepare dataset and create dataset loaders
2. **Environment Setup**: Create environment wrappers for evaluation
3. **Training Configuration**: Create training scripts and configuration files
4. **Testing**: Verify the implementation with test scripts

## 1. Dataset Preparation

### 1.1 Dataset Download and Upload (Optional)

We use HuggingFace to manage datasets. If you want to share your dataset:

1. Upload your dataset to the HuggingFace dataset repo: `ChaoyiPan/mip-dataset`
2. Refer to `examples/process_single_robomimic_dataset.py` for examples
3. Consider creating a dataset uploader script in `examples/` for convenience

Example: For PushT, you can download from `lerobot/pusht` and then upload, or use an existing zarr file locally.

### 1.2 Create Dataset Loader

Create a new dataset loader file: `mip/datasets/<task_name>_dataset.py`

**Required components:**

1. **`make_dataset(task_config, mode="train")` function**: Factory function that creates the appropriate dataset based on configuration
   ```python
   def make_dataset(task_config, mode="train"):
       """Create dataset based on configuration."""
       if task_config.obs_type == "state":
           return StateDataset(...)
       elif task_config.obs_type == "image":
           return ImageDataset(...)
       else:
           raise ValueError(f"Invalid observation type: {task_config.obs_type}")
   ```

2. **Dataset classes**: Inherit from `BaseDataset` and implement:
   - `__init__()`: Initialize dataset, replay buffer, sampler, and normalizer
   - `__len__()`: Return dataset length
   - `__getitem__(idx)`: Return a sample with structure:
     ```python
     {
         "obs": {...},  # Observations (dict for image, tensor for state)
         "action": tensor  # Actions
     }
     ```
   - `get_normalizer()`: Return normalizers for observations and actions

**Key considerations:**
- Support different observation types (state, image, keypoint, etc.)
- Use `ReplayBuffer` for data storage
- Use `SequenceSampler` for sequence sampling with padding
- Use appropriate normalizers (`MinMaxNormalizer`, `ImageNormalizer`)
- Handle `horizon`, `pad_before`, `pad_after` correctly

**Example structure:**
```python
class TaskStateDataset(BaseDataset):
    def __init__(self, dataset_path, horizon=1, pad_before=0, pad_after=0):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(dataset_path, keys=...)
        self.sampler = SequenceSampler(...)
        self.normalizer = self.get_normalizer()

    def get_normalizer(self):
        return {
            "obs": {"state": MinMaxNormalizer(...)},
            "action": MinMaxNormalizer(...)
        }

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)
        # Process and normalize sample
        return torch_data
```

### 1.3 Create Dataset Tests

Create test file: `mip/datasets/<task_name>_dataset_test.py`

**Required tests:**
- `test_initialization`: Verify dataset can be initialized
- `test_len`: Check `__len__` returns correct length
- `test_getitem`: Verify `__getitem__` returns correct structure and shapes
- `test_normalizer`: Check normalizer is created correctly
- Tests for each observation type (state, image, etc.)

Use `pytest` fixtures to create temporary test data (zarr or hdf5).

**Reference files:**
- Dataset: `mip/datasets/robomimic_dataset.py`, `mip/datasets/pusht_dataset.py`
- Tests: `mip/datasets/robomimic_dataset_test.py`, `mip/datasets/pusht_dataset_test.py`

## 2. Environment Setup

### 2.1 Create Environment Wrapper

Create environment wrapper file: `mip/envs/<task_name>/<task_name>_env_wrapper.py`

**Required function:**

```python
def make_vec_env(task_config: TaskConfig, seed=None):
    """Create vectorized environments.

    Args:
        task_config: Task configuration
        seed: Random seed for environments

    Returns:
        Vectorized gym environment
    """
    # Choose vectorization strategy
    if task_config.num_envs == 1 or task_config.save_video:
        vnc_env_class = gym.vector.SyncVectorEnv
    else:
        vnc_env_class = gym.vector.AsyncVectorEnv

    # Create vectorized environments
    envs = vnc_env_class([
        make_task_env(task_config, idx, False, seed=seed)
        for idx in range(task_config.num_envs)
    ])
    return envs
```

**Helper function structure:**

```python
def make_task_env(task_config: TaskConfig, idx, render=False, seed=None):
    """Create a single environment instance."""
    def thunk():
        # 1. Create base environment (import locally to avoid dependency issues)
        env = create_base_env(task_config)

        # 2. Add observation wrappers if needed
        if task_config.obs_type == "image":
            env = ImageWrapper(env)

        # 3. Add video recording wrapper
        video_recoder = VideoRecorder.create_h264(...)
        env = VideoRecordingWrapper(env, video_recoder, ...)

        # 4. Add multi-step wrapper (important!)
        env = MultiStepWrapper(
            env,
            n_obs_steps=task_config.obs_steps,
            n_action_steps=task_config.act_steps,
            max_episode_steps=task_config.max_episode_steps,
        )

        # 5. Set seed
        if seed is not None:
            env.seed(seed + idx)

        return env
    return thunk
```

**Key wrappers:**
- `VideoRecordingWrapper`: For video recording
- `MultiStepWrapper`: Handles observation stacking and action repetition
- Custom wrappers for observation preprocessing

### 2.2 Update __init__.py

Update `mip/envs/<task_name>/__init__.py`:
```python
from mip.envs.<task_name>.<task_name>_env_wrapper import make_vec_env

__all__ = ["make_vec_env"]
```

**Reference files:**
- `mip/envs/robomimic/robomimic_env.py`
- `mip/envs/pusht/pusht_env_wrapper.py`

## 3. Training Configuration

### 3.1 Create Training Script

Create training script: `examples/train_<task_name>.py`

**Required functions:**

1. **`train(config, envs, dataset, agent, logger)`**: Main training loop
   - Load data with DataLoader
   - Setup LR scheduler and warmup scheduler
   - Training loop with periodic logging, saving, and evaluation
   - Track best and average metrics

2. **`eval(config, envs, dataset, agent, logger, num_steps=1)`**: Evaluation function
   - Reset environments
   - Collect episodes
   - Normalize observations
   - Sample actions from agent
   - Unnormalize actions
   - Execute actions and collect rewards
   - Return metrics (mean_step, mean_reward, mean_success)

3. **`main(config)`**: Main entry point
   - Setup logger, environment, dataset, agent
   - Auto-configure network encoder based on observation type
   - Call train or eval based on mode

**Important details:**
- Handle different observation types (state, image, keypoint)
- Properly normalize/unnormalize observations and actions
- Use appropriate action slicing: `act[:, start:end, :]` where `start = obs_steps - 1`
- Support video recording during evaluation

### 3.2 Create Task Configurations

Create task config files: `examples/configs/task/<task_name>_<obs_type>.yaml`

**Required fields:**
```yaml
# @package task
defaults:
  - _self_

# Task identification
env_name: "task_name"
obs_type: "state"  # or "image", "keypoint", etc.
abs_action: false  # true for absolute actions, false for delta actions

# Dataset configuration
dataset_path: "path/to/dataset"
# OR for HuggingFace:
# dataset_repo: "ChaoyiPan/mip-dataset"
# dataset_filename: "task/variant/dataset.hdf5"

# Environment parameters
max_episode_steps: 400

# Task parameters
obs_steps: 2        # Number of observation steps to stack
act_steps: 8        # Number of action steps to execute
horizon: 16         # Prediction horizon
num_envs: 1         # Number of parallel environments
save_video: false   # Whether to save videos
val_dataset_percentage: 0.0  # Validation split percentage

# Dimensions
act_dim: 10         # Action dimension
obs_dim: 20         # Observation dimension
```

**For image observations, add:**
```yaml
shape_meta:
  action:
    shape: [10]
  obs:
    image_key:
      shape: [3, 96, 96]
      type: rgb
    lowdim_key:
      shape: [10]
      type: low_dim

# Image encoder settings
rgb_model: resnet18
resize_shape: null
crop_shape: [84, 84]
random_crop: true
use_group_norm: true
use_seq: true
```

Create configs for each observation type:
- `<task_name>_state.yaml`
- `<task_name>_image.yaml`
- `<task_name>_keypoint.yaml` (if applicable)

### 3.3 Create Test Script

Create test script: `examples/test_<task_name>.py`

**Purpose:** Verify all task configurations work correctly

**Structure:**
```python
TASKS = {
    "task_name": ["state", "image", "keypoint"],
}

TEST_STEPS = 10  # Quick test
BATCH_SIZE = 4

def test_task(task_name, obs_type):
    """Test a single configuration."""
    cmd = [
        "uv", "run", "python",
        "examples/train_task.py",
        "-cn", "exps/debug.yaml",
        f"task={task_name}_{obs_type}",
        f"optimization.gradient_steps={TEST_STEPS}",
        # ... other overrides
    ]
    # Run and verify

def main():
    """Test all configurations."""
    # Run tests and report results
```

**Reference files:**
- Training: `examples/train_robomimic.py`, `examples/train_pusht.py`
- Config: `examples/configs/task/lift_ph_state.yaml`, `examples/configs/task/pusht_state.yaml`
- Test: `examples/test_robomimic.py`, `examples/test_pusht.py`

## 4. Testing and Verification

### 4.1 Download and Process Dataset

First, download and process the dataset from the source:

```bash
# Download dataset (will download to data/<task_name>/)
uv run python examples/process_<task_name>_dataset.py --skip_upload

# Or upload to HuggingFace (requires authentication)
uv run python examples/process_<task_name>_dataset.py
```

### 4.2 Quick Dataset Test

Verify the dataset loader works correctly:

```bash
# Quick test to verify dataset loading
uv run python -c "from mip.datasets.<task_name>_dataset import <Task>StateDataset; \
ds = <Task>StateDataset('data/<task_name>/<dataset_dir>', horizon=16, pad_before=1, pad_after=7); \
print(f'✅ Dataset loaded: {len(ds)} samples'); \
item = ds[0]; \
print(f'✅ State shape: {item[\"obs\"][\"state\"].shape}'); \
print(f'✅ Action shape: {item[\"action\"].shape}')"
```

### 4.3 Run Dataset Tests

```bash
# Test dataset loader (may skip if HuggingFace download not set up)
uv run pytest mip/datasets/<task_name>_dataset_test.py -v

# Test specific observation type
uv run pytest mip/datasets/<task_name>_dataset_test.py::Test<Task>StateDataset -v
```

### 4.4 Run Training Test

```bash
# Test with debug config using local dataset
uv run python examples/train_<task_name>.py -cn exps/debug.yaml task=<task_name>_state +task.dataset_path=data/<task_name>/<dataset_dir>

# Test with HuggingFace dataset (if uploaded)
uv run python examples/train_<task_name>.py -cn exps/debug.yaml task=<task_name>_state

# Run full test suite
uv run python examples/test_<task_name>.py
```

### 4.5 Verification Checklist

- [ ] Dataset download script works
- [ ] Dataset loader works for all observation types
- [ ] Dataset tests pass (or skip gracefully)
- [ ] Environment can be created and reset
- [ ] Training loop runs without errors
- [ ] Evaluation works correctly
- [ ] Video recording works (if enabled)
- [ ] All configurations in test script pass
- [ ] Metrics are logged correctly

### 4.6 Common Testing Issues

**Dataset not found:**
- Make sure to run `process_<task_name>_dataset.py` first
- Check the dataset path in the config matches the downloaded location
- Use `+task.dataset_path=<path>` to override the config

**Import errors:**
- Ensure all dependencies are installed: `uv sync`
- For environments using legacy APIs, check compatibility wrappers

**Gym vs Gymnasium:**
- This repo uses `gymnasium` (modern replacement for `gym`)
- For environments using old `gym` API (like adept_envs), create wrapper to convert:
  - `reset()` → `reset(seed, options)` returns `(obs, info)`
  - `step()` returns 5-tuple `(obs, reward, terminated, truncated, info)`
  - `render(mode)` → `render()` returns array

## 5. Common Patterns and Tips

### 5.1 Observation Processing

**State observations:**
```python
# In dataset
obs_dict = {"state": state_tensor}

# In eval
obs = dataset.normalizer["obs"]["state"].normalize(obs)
obs = torch.tensor(obs, device=device)
```

**Image observations:**
```python
# In dataset
obs_dict = {
    "image": image_tensor,  # (T, C, H, W)
    "agent_pos": pos_tensor
}

# In eval
for k in obs_raw:
    obs[k] = dataset.normalizer["obs"][k].normalize(obs[k])
    obs[k] = torch.tensor(obs[k], device=device)
```

### 5.2 Action Processing

```python
# In eval loop
start = config.task.obs_steps - 1
end = start + config.task.act_steps
action = action_pred[:, start:end, :]

# For absolute actions with rotation
if config.task.abs_action:
    action = dataset.undo_transform_action(action)
```

### 5.3 Directory Structure

```
mip/
├── mip/
│   ├── datasets/
│   │   ├── task_dataset.py
│   │   └── task_dataset_test.py
│   └── envs/
│       └── task/
│           ├── __init__.py
│           ├── task_env.py
│           └── task_env_wrapper.py
└── examples/
    ├── train_task.py
    ├── test_task.py
    └── configs/
        └── task/
            ├── task_state.yaml
            ├── task_image.yaml
            └── task_keypoint.yaml
```

## 6. Reference Examples

### Complete Examples
- **Robomimic**: State and image observations with rotation transformations
  - Dataset: `mip/datasets/robomimic_dataset.py`
  - Environment: `mip/envs/robomimic/robomimic_env.py`
  - Training: `examples/train_robomimic.py`

- **PushT**: State, keypoint, and image observations
  - Dataset: `mip/datasets/pusht_dataset.py`
  - Environment: `mip/envs/pusht/pusht_env_wrapper.py`
  - Training: `examples/train_pusht.py`

### External References
- Adaptive Diffusion pusht_launcher: For comparison and insights
- CleanDiffuser pusht_dataset: For dataset format reference

## 7. Troubleshooting

### Common Issues

1. **Import errors**: Make sure to import heavy dependencies (like robomimic) inside functions to avoid unnecessary dependencies
2. **Shape mismatches**: Verify observation and action dimensions match between dataset, environment, and config
3. **Normalization issues**: Ensure consistent normalization between training and evaluation
4. **Seed issues**: Set seeds for both environments and random number generators
5. **Video recording**: Ensure VideoRecordingWrapper is added before MultiStepWrapper

### Debug Tips

- Use `debug.yaml` config for quick testing with small gradient_steps
- Print shapes during development to verify dimensions
- Test each component independently (dataset, environment, training)
- Check normalizer statistics to ensure reasonable ranges
