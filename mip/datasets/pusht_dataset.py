"""PushT dataset.

Author: Chaoyi Pan
Date: 2025-10-15
"""

import os
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from loguru import logger

from mip.dataset_utils import (
    ImageNormalizer,
    MinMaxNormalizer,
    ReplayBuffer,
    SequenceSampler,
    dict_apply,
)
from mip.datasets.base import BaseDataset


def download_pusht_dataset(
    dataset_filename: str = "pusht/pusht_cchi_v7_replay.zarr.zip",
    repo_id: str = "ChaoyiPan/mip-dataset",
) -> str:
    """Download PushT dataset from HuggingFace and extract locally.

    Args:
        dataset_filename: Filename in the HuggingFace dataset repo (should be a .zip file)
        repo_id: HuggingFace repository ID

    Returns:
        Path to the extracted zarr dataset
    """
    logger.info(f"Downloading PushT dataset from {repo_id}/{dataset_filename}")
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=dataset_filename,
        repo_type="dataset",
    )
    logger.info(f"Downloaded zip file to: {zip_path}")

    # Extract the zip file in the same directory
    zip_path_obj = Path(zip_path)
    extract_dir = zip_path_obj.parent

    logger.info(f"Extracting dataset to {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # Find the extracted zarr directory
    # Assuming the zip contains a single .zarr directory
    zarr_name = zip_path_obj.stem  # Remove .zip extension
    if zarr_name.endswith(".zarr"):
        zarr_path = extract_dir / zarr_name
    else:
        # Look for .zarr directories
        zarr_dirs = list(extract_dir.glob("*.zarr"))
        if not zarr_dirs:
            raise FileNotFoundError(
                f"No .zarr directory found after extracting {zip_path}"
            )
        zarr_path = zarr_dirs[0]

    logger.info(f"Extracted dataset to: {zarr_path}")
    return str(zarr_path)


def make_dataset(task_config, mode="train"):
    """Create PushT dataset based on configuration.

    Args:
        task_config: Task configuration with dataset parameters
        mode: Dataset mode (train/val), currently unused for PushT

    Returns:
        Dataset instance for PushT task
    """
    _ = mode  # Currently unused for PushT, but kept for API consistency

    # Check if we should download from HuggingFace
    if hasattr(task_config, "dataset_repo") and hasattr(
        task_config, "dataset_filename"
    ):
        # Auto-download from HuggingFace (handles zip extraction)
        dataset_filename = task_config.dataset_filename

        # If filename ends with .zip, download and extract
        if dataset_filename.endswith(".zip"):
            dataset_path = download_pusht_dataset(
                dataset_filename=dataset_filename,
                repo_id=task_config.dataset_repo,
            )
        else:
            # Direct download (backward compatibility for non-zip files)
            logger.info(
                f"Downloading dataset from {task_config.dataset_repo}/{dataset_filename}"
            )
            dataset_path = hf_hub_download(
                repo_id=task_config.dataset_repo,
                filename=dataset_filename,
                repo_type="dataset",
            )
            logger.info(f"Downloaded dataset to: {dataset_path}")
    elif hasattr(task_config, "dataset_path"):
        # Use explicit path if provided
        dataset_path = os.path.expanduser(task_config.dataset_path)
    else:
        raise ValueError(
            "Either dataset_repo/dataset_filename or dataset_path must be provided"
        )

    if task_config.obs_type == "state":
        return PushTStateDataset(
            dataset_path=dataset_path,
            horizon=task_config.horizon,
            pad_before=task_config.obs_steps - 1,
            pad_after=task_config.act_steps - 1,
        )
    elif task_config.obs_type == "keypoint":
        return PushTKeypointDataset(
            dataset_path=dataset_path,
            horizon=task_config.horizon,
            pad_before=task_config.obs_steps - 1,
            pad_after=task_config.act_steps - 1,
        )
    elif task_config.obs_type == "image":
        return PushTImageDataset(
            dataset_path=dataset_path,
            shape_meta=task_config.shape_meta,
            n_obs_steps=task_config.obs_steps,
            horizon=task_config.horizon,
            pad_before=task_config.obs_steps - 1,
            pad_after=task_config.act_steps - 1,
        )
    else:
        raise ValueError(f"Invalid observation type: {task_config.obs_type}")


class PushTStateDataset(BaseDataset):
    """PushT dataset with state observations.

    State observation contains: [agent_x, agent_y, block_x, block_y, block_angle]
    """

    def __init__(
        self,
        dataset_path,
        obs_keys=("state", "action"),
        horizon=1,
        pad_before=0,
        pad_after=0,
    ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(dataset_path, keys=obs_keys)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.normalizer = self.get_normalizer()

    def get_normalizer(self, **kwargs):
        state_normalizer = MinMaxNormalizer(self.replay_buffer["state"][:])
        action_normalizer = MinMaxNormalizer(self.replay_buffer["action"][:])
        return {
            "obs": {"state": state_normalizer},
            "action": action_normalizer,
        }

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        state = sample["state"].astype(np.float32)
        state = self.normalizer["obs"]["state"].normalize(state)

        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)

        data = {
            "obs": {"state": state},
            "action": action,
        }
        torch_data = dict_apply(data, torch.tensor)
        return torch_data


class PushTKeypointDataset(BaseDataset):
    """PushT dataset with keypoint observations.

    Keypoint observation contains 9 keypoints (18 values) plus agent position (2 values).
    """

    def __init__(
        self,
        dataset_path,
        obs_keys=("keypoint", "state", "action"),
        horizon=1,
        pad_before=0,
        pad_after=0,
    ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(dataset_path, keys=obs_keys)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.normalizer = self.get_normalizer()

    def get_normalizer(self, **kwargs):
        agent_pos_normalizer = MinMaxNormalizer(self.replay_buffer["state"][:, :2])
        keypoint_normalizer = MinMaxNormalizer(self.replay_buffer["keypoint"][:])
        action_normalizer = MinMaxNormalizer(self.replay_buffer["action"][:])
        return {
            "obs": {
                "keypoint": keypoint_normalizer,
                "agent_pos": agent_pos_normalizer,
            },
            "action": action_normalizer,
        }

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        # keypoint: (T, 9, 2) -> (T, 18)
        data_size = sample["keypoint"].shape[0]
        keypoint = (
            sample["keypoint"]
            .reshape(-1, sample["keypoint"].shape[-1])
            .astype(np.float32)
        )
        keypoint = self.normalizer["obs"]["keypoint"].normalize(keypoint)
        keypoint = keypoint.reshape(data_size, -1)

        # agent_pos: (T, 2)
        agent_pos = sample["state"][:, :2].astype(np.float32)
        agent_pos = self.normalizer["obs"]["agent_pos"].normalize(agent_pos)

        # action: (T, 2)
        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)

        data = {
            "obs": {
                "keypoint": keypoint,
                "agent_pos": agent_pos,
            },
            "action": action,
        }
        torch_data = dict_apply(data, torch.tensor)
        return torch_data


class PushTImageDataset(BaseDataset):
    """PushT dataset with image observations.

    Image observation contains RGB image plus agent position.
    """

    def __init__(
        self,
        dataset_path,
        shape_meta: dict,
        n_obs_steps=None,
        horizon=1,
        pad_before=0,
        pad_after=0,
    ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            dataset_path, keys=["img", "state", "action"]
        )

        # Parse shape_meta to get rgb and lowdim keys
        rgb_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type_ = attr.get("type", "low_dim")
            if type_ == "rgb":
                rgb_keys.append(key)
            elif type_ == "low_dim":
                lowdim_keys.append(key)

        key_first_k = {}
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            key_first_k=key_first_k,
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

        self.normalizer = self.get_normalizer()

    def get_normalizer(self, **kwargs):
        normalizer = defaultdict(dict)
        # For PushT, we map 'img' to 'image' and state[:,:2] to 'agent_pos'
        normalizer["obs"]["image"] = ImageNormalizer()
        normalizer["obs"]["agent_pos"] = MinMaxNormalizer(
            self.replay_buffer["state"][..., :2]
        )
        normalizer["action"] = MinMaxNormalizer(self.replay_buffer["action"][:])
        return normalizer

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps)

        obs_dict = {}

        # Image: (T, H, W, C) -> (T, C, H, W)
        image = np.moveaxis(sample["img"][T_slice], -1, 1).astype(np.float32) / 255.0
        del sample["img"]
        obs_dict["image"] = self.normalizer["obs"]["image"].normalize(image)

        # Agent position: (T, 2)
        agent_pos = sample["state"][T_slice, :2].astype(np.float32)
        obs_dict["agent_pos"] = self.normalizer["obs"]["agent_pos"].normalize(agent_pos)

        # Action: (T, 2)
        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.tensor),
            "action": torch.tensor(action),
        }
        return torch_data
