"""Kitchen dataset.
Port from CleanDiffuser https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/dataset/kitchen_dataset.py

Author: Chaoyi Pan
Date: 2025-10-17
"""

import os
import pathlib
import zipfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from loguru import logger

from mip.dataset_utils import (
    MinMaxNormalizer,
    ReplayBuffer,
    SequenceSampler,
    dict_apply,
)
from mip.datasets.base import BaseDataset


def download_kitchen_dataset(
    dataset_filename: str = "kitchen/kitchen_demos_multitask.zip",
    repo_id: str = "ChaoyiPan/mip-dataset",
) -> str:
    """Download Kitchen dataset from HuggingFace and extract locally.

    Args:
        dataset_filename: Filename in the HuggingFace dataset repo (should be a .zip file)
        repo_id: HuggingFace repository ID

    Returns:
        Path to the extracted dataset directory
    """
    logger.info(f"Downloading Kitchen dataset from {repo_id}/{dataset_filename}")
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

    # Find the extracted directory
    # The zip should contain a directory with observations_seq.npy, actions_seq.npy, existence_mask.npy
    dataset_name = zip_path_obj.stem  # Remove .zip extension
    dataset_path = extract_dir / dataset_name

    if not dataset_path.exists():
        # Look for the directory that was extracted
        extracted_dirs = [
            d
            for d in extract_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        if extracted_dirs:
            dataset_path = extracted_dirs[0]
        else:
            raise FileNotFoundError(f"No directory found after extracting {zip_path}")

    logger.info(f"Extracted dataset to: {dataset_path}")
    return str(dataset_path)


def make_dataset(task_config, mode="train"):
    """Create Kitchen dataset based on configuration.

    Args:
        task_config: Task configuration with dataset parameters
        mode: Dataset mode (train/val), currently unused for Kitchen

    Returns:
        Dataset instance for Kitchen task
    """
    _ = mode  # Currently unused for Kitchen, but kept for API consistency

    # Check if we should download from HuggingFace
    if hasattr(task_config, "dataset_repo") and hasattr(
        task_config, "dataset_filename"
    ):
        # Auto-download from HuggingFace (handles zip extraction)
        dataset_filename = task_config.dataset_filename

        # If filename ends with .zip, download and extract
        if dataset_filename.endswith(".zip"):
            dataset_path = download_kitchen_dataset(
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
        return KitchenStateDataset(
            dataset_path=dataset_path,
            horizon=task_config.horizon,
            pad_before=task_config.obs_steps - 1,
            pad_after=task_config.act_steps - 1,
        )
    else:
        raise ValueError(f"Invalid observation type: {task_config.obs_type}")


class KitchenStateDataset(BaseDataset):
    """Kitchen dataset with state observations.

    The dataset contains demonstrations of a robot performing kitchen tasks.
    State observation contains 60-dimensional observation vector:
    - robot_qpos: 9-dimensional joint positions
    - object_qpos: 21-dimensional object positions
    - robot_qvel: 30-dimensional (zeros in processed data)
    """

    def __init__(
        self,
        dataset_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
    ):
        super().__init__()

        data_directory = pathlib.Path(dataset_path)
        observations = np.load(data_directory / "observations_seq.npy")
        actions = np.load(data_directory / "actions_seq.npy")
        masks = np.load(data_directory / "existence_mask.npy")

        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        for i in range(len(masks)):
            eps_len = int(masks[i].sum())
            obs = observations[i, :eps_len].astype(np.float32)
            action = actions[i, :eps_len].astype(np.float32)
            data = {
                "state": obs,
                "action": action,
            }
            self.replay_buffer.add_episode(data)

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
