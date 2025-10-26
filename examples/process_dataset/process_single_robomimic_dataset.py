#!/usr/bin/env python
"""Modern HuggingFace-based dataset management for Robomimic datasets.

This script handles:
1. Downloading datasets from HuggingFace Hub
2. Processing demo datasets to image datasets
3. Uploading processed datasets back to HuggingFace Hub
4. Loading datasets directly with load_dataset

Author: Chaoyi Pan
Date: 2025-10-04
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import h5py
import loguru
from huggingface_hub import HfApi, hf_hub_download, upload_file
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# HuggingFace repo for processed datasets
PROCESSED_REPO_ID = "ChaoyiPan/mip-dataset"

# Path to task config directory
SCRIPT_DIR = Path(__file__).parent
CONFIG_DIR = SCRIPT_DIR / "configs"


def load_task_config(task: str, source: str, obs_type: str = "image") -> DictConfig:
    """Load task configuration using Hydra.

    Args:
        task: Task name (e.g., 'lift', 'can', 'square')
        source: Data source (e.g., 'ph', 'mh')
        obs_type: Observation type ('image' or 'state')

    Returns:
        Hydra DictConfig with resolved configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_name = f"{task}_{source}_{obs_type}"
    config_file = CONFIG_DIR / "task" / f"{config_name}.yaml"

    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_file}\n"
            f"Expected path: {config_file.absolute()}"
        )

    # Initialize Hydra with the config directory
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.absolute()),
        version_base="1.3",
    ):
        # Compose the configuration
        cfg = compose(config_name=f"task/{config_name}")

    return cfg


def get_camera_names_from_config(task: str, source: str) -> list[str]:
    """Extract camera names from task configuration file using Hydra.

    Args:
        task: Task name (e.g., 'lift', 'can', 'square')
        source: Data source (e.g., 'ph', 'mh')

    Returns:
        List of camera names (without '_image' suffix)

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is missing required fields or has no RGB observations
    """
    cfg = load_task_config(task, source, obs_type="image")

    # Access the task config (Hydra wraps it under 'task' key)
    task_cfg = cfg.task

    # Extract image observation keys from shape_meta
    if "shape_meta" not in task_cfg:
        raise ValueError(f"Config for {task}/{source} is missing 'shape_meta' field")

    if "obs" not in task_cfg.shape_meta:
        raise ValueError(
            f"Config for {task}/{source} is missing 'shape_meta.obs' field"
        )

    obs_config = task_cfg.shape_meta.obs

    # Find all observation keys with type 'rgb' (image observations)
    camera_names = []
    for key, value in obs_config.items():
        if isinstance(value, DictConfig) and value.get("type") == "rgb":
            # Remove '_image' suffix to get camera name
            if key.endswith("_image"):
                camera_name = key[:-6]  # Remove '_image'
                camera_names.append(camera_name)
            else:
                camera_names.append(key)

    if not camera_names:
        raise ValueError(
            f"No RGB observations found in config for {task}/{source}\n"
            f"Available observation keys: {list(obs_config.keys())}"
        )

    loguru.logger.info(f"Loaded camera names from config: {camera_names}")
    return camera_names


def get_camera_dimensions_from_config(task: str, source: str) -> tuple[int, int]:
    """Extract camera dimensions from task configuration file using Hydra.

    Args:
        task: Task name (e.g., 'lift', 'can', 'square')
        source: Data source (e.g., 'ph', 'mh')

    Returns:
        Tuple of (height, width)

    Raises:
        ValueError: If camera dimensions cannot be determined
    """
    cfg = load_task_config(task, source, obs_type="image")

    # Access the task config (Hydra wraps it under 'task' key)
    task_cfg = cfg.task

    if "shape_meta" not in task_cfg or "obs" not in task_cfg.shape_meta:
        raise ValueError(f"Config for {task}/{source} is missing observation metadata")

    obs_config = task_cfg.shape_meta.obs

    # Find first RGB observation and extract dimensions
    for _key, value in obs_config.items():
        if isinstance(value, DictConfig) and value.get("type") == "rgb":
            if "shape" in value:
                shape = OmegaConf.to_object(value.shape)
                # Shape is [channels, height, width]
                if len(shape) == 3:
                    _, height, width = shape
                    loguru.logger.info(
                        f"Loaded camera dimensions from config: {height}x{width}"
                    )
                    return height, width

    raise ValueError(
        f"Could not determine camera dimensions from config for {task}/{source}"
    )


def download_original_dataset(
    task: str = "lift",
    source: str = "ph",
    dataset_type: str = "low_dim",
) -> str:
    """Download original robomimic dataset from HuggingFace.

    Args:
        task: Task name (e.g., 'lift', 'can', 'square')
        source: Data source (e.g., 'ph', 'mh', 'mg')
        dataset_type: Type of dataset ('low_dim' or 'demo')

    Returns:
        Path to downloaded HDF5 file
    """
    repo_id = "amandlek/robomimic"

    if dataset_type == "low_dim":
        filename = f"v1.5/{task}/{source}/low_dim_v15.hdf5"
    elif dataset_type == "demo":
        filename = f"v1.5/{task}/{source}/demo_v15.hdf5"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")

    loguru.logger.info(f"Downloading {filename} from {repo_id}...")
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
    )
    loguru.logger.info(f"Downloaded to: {file_path}")
    return file_path


def process_to_image_dataset(
    demo_path: str,
    task: str = "lift",
    source: str = "ph",
    n_demos: int = -1,  # -1 means all demos
    camera_names: list | None = None,
    camera_height: int | None = None,
    camera_width: int | None = None,
) -> str:
    """Process demo dataset to image dataset.

    Args:
        demo_path: Path to demo.hdf5 file
        task: Task name for camera configuration
        source: Data source
        n_demos: Number of demos to process (-1 for all demos)
        camera_names: List of camera names (loaded from config if None)
        camera_height: Camera image height (loaded from config if None)
        camera_width: Camera image width (loaded from config if None)

    Returns:
        Path to processed image dataset
    """
    if camera_names is None:
        # Load camera names from config file
        camera_names = get_camera_names_from_config(task, source)

    if camera_height is None or camera_width is None:
        # Load camera dimensions from config file
        height, width = get_camera_dimensions_from_config(task, source)
        if camera_height is None:
            camera_height = height
        if camera_width is None:
            camera_width = width

    # Create temp directory for processing
    temp_dir = tempfile.mkdtemp()
    temp_demo_path = os.path.join(temp_dir, "demo.hdf5")

    # Determine output name based on whether we're processing all demos
    if n_demos == -1:
        output_name = f"image_{task}_{source}.hdf5"
        loguru.logger.info("Processing ALL demos to image dataset...")
    else:
        output_name = f"image_{task}_{source}_{n_demos}demos.hdf5"
        loguru.logger.info(f"Processing {n_demos} demos to image dataset...")

    output_path = os.path.join(temp_dir, output_name)
    shutil.copy2(demo_path, temp_demo_path)

    # Build command
    cmd = [
        "python",
        "-m",
        "robomimic.scripts.dataset_states_to_obs",
        f"--dataset={temp_demo_path}",
        f"--output_name={output_name}",
        "--done_mode=2",
        f"--camera_height={camera_height}",
        f"--camera_width={camera_width}",
    ]

    # Only add -n flag if not processing all demos
    if n_demos != -1:
        cmd.append(f"--n={n_demos}")

    # Add camera names - all camera names should follow a single --camera_names flag
    if camera_names:
        cmd.append("--camera_names")
        cmd.extend(camera_names)

    # Set environment for EGL rendering
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"

    loguru.logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to process dataset:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    loguru.logger.info(f"Processed dataset saved to: {output_path}")
    return output_path


def upload_to_hub(
    local_path: str,
    task: str,
    source: str,
    dataset_type: str,
    n_demos: int | None = None,
    repo_id: str = PROCESSED_REPO_ID,
) -> str:
    """Upload processed dataset to HuggingFace Hub.

    Args:
        local_path: Path to local HDF5 file
        task: Task name
        source: Data source
        dataset_type: Type ('low_dim' or 'image')
        n_demos: Number of demos (for image datasets, -1 or None for all)
        repo_id: HuggingFace repo ID

    Returns:
        Path in the repository
    """
    # Determine path in repo
    if dataset_type == "image":
        if n_demos and n_demos != -1:
            repo_path = f"robomimic/{task}/{source}/image_{n_demos}demos.hdf5"
        else:
            repo_path = f"robomimic/{task}/{source}/image.hdf5"
    elif dataset_type == "low_dim":
        repo_path = f"robomimic/{task}/{source}/low_dim.hdf5"
    else:
        repo_path = f"robomimic/{task}/{source}/{dataset_type}.hdf5"

    # Check if authenticated
    try:
        api = HfApi()
        # Try to get user info to check authentication
        api.whoami()
    except Exception:
        loguru.logger.error("Not authenticated with HuggingFace Hub. Skipping upload.")
        print("   To enable uploading, run: huggingface-cli login")
        return repo_path

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        loguru.logger.error(f"Could not create/access repo: {e}")
        print("   Dataset saved locally but not uploaded.")
        return repo_path

    loguru.logger.info(f"Uploading to {repo_id}/{repo_path}...")

    try:
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
        )
        loguru.logger.info(f"Uploaded successfully to {repo_id}/{repo_path}")
        return repo_path
    except Exception as e:
        loguru.logger.error(f"Upload failed: {e}")
        return repo_path


def get_or_create_image_dataset(
    task: str = "lift",
    source: str = "ph",
    n_demos: int = -1,
    force_recreate: bool = False,
) -> str:
    """Get image dataset from Hub or create and upload if needed.

    Args:
        task: Task name
        source: Data source
        n_demos: Number of demos (-1 for all)
        force_recreate: Force recreation even if exists

    Returns:
        Path to local HDF5 file
    """
    # Local cache directory
    cache_dir = Path.home() / ".cache" / "mip" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine cache filename based on demo count
    if n_demos == -1:
        local_cache_path = cache_dir / f"{task}_{source}_image.hdf5"
        repo_path = f"robomimic/{task}/{source}/image.hdf5"
    else:
        local_cache_path = cache_dir / f"{task}_{source}_image_{n_demos}demos.hdf5"
        repo_path = f"robomimic/{task}/{source}/image_{n_demos}demos.hdf5"

    if not force_recreate:
        # Check local cache first
        if local_cache_path.exists():
            loguru.logger.info(f"Found cached dataset: {local_cache_path}")
            return str(local_cache_path)

        # Try to download from processed repo
        try:
            loguru.logger.info(
                f"Checking for existing dataset at {PROCESSED_REPO_ID}/{repo_path}"
            )
            hub_path = hf_hub_download(
                repo_id=PROCESSED_REPO_ID,
                filename=repo_path,
                repo_type="dataset",
            )
            # Copy to local cache
            shutil.copy2(hub_path, local_cache_path)
            loguru.logger.info(f"Downloaded and cached dataset: {local_cache_path}")
            return str(local_cache_path)
        except Exception:
            loguru.logger.info("Dataset not found in Hub, will process locally")

    # Download demo dataset
    demo_path = download_original_dataset(task=task, source=source, dataset_type="demo")

    # Process to image dataset
    image_path = process_to_image_dataset(
        demo_path=demo_path,
        task=task,
        source=source,
        n_demos=n_demos,
    )

    # Copy to local cache
    shutil.copy2(image_path, local_cache_path)
    loguru.logger.info(f"Cached processed dataset: {local_cache_path}")

    # Try to upload to Hub (optional)
    upload_to_hub(
        local_path=str(local_cache_path),
        task=task,
        source=source,
        dataset_type="image",
        n_demos=n_demos,
    )

    return str(local_cache_path)


def get_or_upload_lowdim_dataset(
    task: str = "lift",
    source: str = "ph",
) -> str:
    """Get low-dim dataset and ensure it's in our Hub repo.

    Args:
        task: Task name
        source: Data source

    Returns:
        Path to local HDF5 file
    """
    repo_path = f"robomimic/{task}/{source}/low_dim.hdf5"

    # Try to download from our processed repo first
    try:
        loguru.logger.info(f"Checking for dataset at {PROCESSED_REPO_ID}/{repo_path}")
        local_path = hf_hub_download(
            repo_id=PROCESSED_REPO_ID,
            filename=repo_path,
            repo_type="dataset",
        )
        loguru.logger.info(f"Found dataset: {local_path}")
        return local_path
    except Exception:
        loguru.logger.info(
            "Dataset not in our repo, downloading from original and uploading..."
        )

    # Download from original repo
    local_path = download_original_dataset(
        task=task, source=source, dataset_type="low_dim"
    )

    # Upload to our repo
    upload_to_hub(
        local_path=local_path,
        task=task,
        source=source,
        dataset_type="low_dim",
    )

    return local_path


def validate_dataset(hdf5_path: str) -> dict:
    """Validate and get info about an HDF5 dataset.

    Args:
        hdf5_path: Path to HDF5 file

    Returns:
        Dictionary with dataset info
    """
    with h5py.File(hdf5_path, "r") as f:
        info = {
            "path": hdf5_path,
            "keys": list(f.keys()),
            "n_demos": len(f["data"]) if "data" in f else 0,
        }

        if info["n_demos"] > 0:
            demo_0 = f["data"]["demo_0"]
            info["demo_keys"] = list(demo_0.keys())

            if "obs" in demo_0:
                info["obs_keys"] = list(demo_0["obs"].keys())
                info["image_keys"] = [k for k in info["obs_keys"] if "image" in k]

                # Get shapes
                info["obs_shapes"] = {}
                for key in info["obs_keys"]:
                    info["obs_shapes"][key] = demo_0["obs"][key].shape

        return info


def main():
    """Example usage of the modern dataset workflow."""
    image_path = get_or_create_image_dataset(
        task="lift",
        source="ph",
        force_recreate=False,  # Use cached if available
    )

    image_info = validate_dataset(image_path)
    loguru.logger.info(image_info)


if __name__ == "__main__":
    main()
