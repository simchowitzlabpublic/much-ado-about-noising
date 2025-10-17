"""Kitchen dataset test.

Tests Kitchen dataset loading with real data from HuggingFace.

Author: Chaoyi Pan
Date: 2025-10-17
"""

import os

import pytest
import torch

from mip.datasets.kitchen_dataset import (
    KitchenStateDataset,
    download_kitchen_dataset,
    make_dataset,
)


@pytest.fixture(scope="module")
def real_data_path():
    """Download real Kitchen dataset from HuggingFace once for all tests."""
    try:
        print("\n🔄 Downloading Kitchen dataset from HuggingFace...")
        path = download_kitchen_dataset()
        print(f"✅ Dataset downloaded to: {path}")
        assert os.path.exists(path)
        return path
    except Exception as e:
        pytest.skip(f"❌ Failed to download real data: {e}")


class TestKitchenStateDataset:
    """Test KitchenStateDataset class with real HuggingFace data."""

    def test_initialization(self, real_data_path):
        """Test that the dataset can be initialized."""
        dataset = KitchenStateDataset(
            dataset_path=real_data_path,
            horizon=16,
            pad_before=1,
            pad_after=7,
        )

        assert dataset is not None
        assert dataset.horizon == 16
        assert dataset.pad_before == 1
        assert dataset.pad_after == 7

    def test_len(self, real_data_path):
        """Test that __len__ returns correct length."""
        dataset = KitchenStateDataset(
            dataset_path=real_data_path,
            horizon=16,
            pad_before=1,
            pad_after=7,
        )

        length = len(dataset)
        assert length > 0
        assert isinstance(length, int)
        print(f"\n📊 Dataset length: {length}")

    def test_getitem(self, real_data_path):
        """Test that __getitem__ returns correct data structure."""
        dataset = KitchenStateDataset(
            dataset_path=real_data_path,
            horizon=16,
            pad_before=1,
            pad_after=7,
        )

        item = dataset[0]

        # Check structure
        assert isinstance(item, dict)
        assert "obs" in item
        assert "action" in item
        assert "state" in item["obs"]

        # Check tensor types
        assert isinstance(item["obs"]["state"], torch.Tensor)
        assert isinstance(item["action"], torch.Tensor)

        # Check shapes
        # Kitchen: state_dim=60 (9 robot qpos + 21 object qpos + 30 zeros)
        # action_dim=9 (robot joint actions)
        assert item["obs"]["state"].shape == (16, 60)  # horizon=16, state_dim=60
        assert item["action"].shape == (16, 9)  # horizon=16, action_dim=9

        print(f"\n📦 State shape: {item['obs']['state'].shape}")
        print(f"📦 Action shape: {item['action'].shape}")

    def test_normalizer(self, real_data_path):
        """Test that normalizer is created correctly."""
        dataset = KitchenStateDataset(
            dataset_path=real_data_path,
            horizon=16,
        )

        assert "obs" in dataset.normalizer
        assert "action" in dataset.normalizer
        assert "state" in dataset.normalizer["obs"]

    def test_multiple_samples(self, real_data_path):
        """Test loading multiple samples."""
        dataset = KitchenStateDataset(
            dataset_path=real_data_path,
            horizon=16,
            pad_before=1,
            pad_after=7,
        )

        # Test first 10 samples
        for i in range(min(10, len(dataset))):
            item = dataset[i]
            assert "obs" in item
            assert "action" in item
            assert item["obs"]["state"].shape == (16, 60)
            assert item["action"].shape == (16, 9)

        print(f"\n✅ Successfully loaded {min(10, len(dataset))} samples")


class TestMakeDataset:
    """Test make_dataset factory function."""

    def test_make_dataset_with_huggingface(self, real_data_path):
        """Test make_dataset function with HuggingFace config."""
        from omegaconf import OmegaConf

        # Test state dataset
        config_dict = {
            "obs_type": "state",
            "dataset_path": real_data_path,
            "horizon": 16,
            "obs_steps": 2,
            "act_steps": 8,
        }
        config = OmegaConf.create(config_dict)

        dataset = make_dataset(config)
        assert dataset is not None
        assert len(dataset) > 0
        assert isinstance(dataset, KitchenStateDataset)

        # Check we can get an item
        item = dataset[0]
        assert "obs" in item
        assert "action" in item

        print(f"\n✅ make_dataset(state) - Dataset length: {len(dataset)}")

    def test_make_dataset_with_repo_config(self):
        """Test make_dataset with HuggingFace repo configuration."""
        from omegaconf import OmegaConf

        config_dict = {
            "obs_type": "state",
            "dataset_repo": "ChaoyiPan/mip-dataset",
            "dataset_filename": "kitchen/kitchen_demos_multitask.zip",
            "horizon": 16,
            "obs_steps": 2,
            "act_steps": 8,
        }
        config = OmegaConf.create(config_dict)

        try:
            dataset = make_dataset(config)
            assert dataset is not None
            assert len(dataset) > 0
            assert isinstance(dataset, KitchenStateDataset)

            print(f"\n✅ make_dataset(HF repo) - Dataset length: {len(dataset)}")
        except Exception as e:
            pytest.skip(f"❌ Failed to download from HuggingFace: {e}")
