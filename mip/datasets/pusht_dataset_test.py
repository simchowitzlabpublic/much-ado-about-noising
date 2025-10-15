"""PushT dataset test.

Tests PushT dataset loading with real data from HuggingFace.

Author: Chaoyi Pan
Date: 2025-10-15
"""

import os

import pytest
import torch

from mip.datasets.pusht_dataset import (
    PushTImageDataset,
    PushTKeypointDataset,
    PushTStateDataset,
    download_pusht_dataset,
    make_dataset,
)


@pytest.fixture(scope="module")
def real_data_path():
    """Download real PushT dataset from HuggingFace once for all tests."""
    try:
        print("\n🔄 Downloading PushT dataset from HuggingFace...")
        path = download_pusht_dataset()
        print(f"✅ Dataset downloaded to: {path}")
        assert os.path.exists(path)
        return path
    except Exception as e:
        pytest.skip(f"❌ Failed to download real data: {e}")


class TestPushTStateDataset:
    """Test PushTStateDataset class with real HuggingFace data."""

    def test_initialization(self, real_data_path):
        """Test that the dataset can be initialized."""
        dataset = PushTStateDataset(
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
        dataset = PushTStateDataset(
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
        dataset = PushTStateDataset(
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
        assert item["obs"]["state"].shape == (16, 5)  # horizon=16, state_dim=5
        assert item["action"].shape == (16, 2)  # horizon=16, action_dim=2

        print(f"\n📦 State shape: {item['obs']['state'].shape}")
        print(f"📦 Action shape: {item['action'].shape}")

    def test_normalizer(self, real_data_path):
        """Test that normalizer is created correctly."""
        dataset = PushTStateDataset(
            dataset_path=real_data_path,
            horizon=16,
        )

        assert "obs" in dataset.normalizer
        assert "action" in dataset.normalizer
        assert "state" in dataset.normalizer["obs"]

    def test_multiple_samples(self, real_data_path):
        """Test loading multiple samples."""
        dataset = PushTStateDataset(
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
            assert item["obs"]["state"].shape == (16, 5)
            assert item["action"].shape == (16, 2)

        print(f"\n✅ Successfully loaded {min(10, len(dataset))} samples")


class TestPushTKeypointDataset:
    """Test PushTKeypointDataset class with real HuggingFace data."""

    def test_initialization(self, real_data_path):
        """Test that the dataset can be initialized."""
        dataset = PushTKeypointDataset(
            dataset_path=real_data_path,
            horizon=16,
            pad_before=1,
            pad_after=7,
        )

        assert dataset is not None
        assert dataset.horizon == 16

    def test_getitem(self, real_data_path):
        """Test that __getitem__ returns correct data structure."""
        dataset = PushTKeypointDataset(
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
        assert "keypoint" in item["obs"]
        assert "agent_pos" in item["obs"]

        # Check tensor types
        assert isinstance(item["obs"]["keypoint"], torch.Tensor)
        assert isinstance(item["obs"]["agent_pos"], torch.Tensor)
        assert isinstance(item["action"], torch.Tensor)

        # Check shapes
        assert item["obs"]["keypoint"].shape == (16, 18)  # horizon=16, 9 keypoints * 2
        assert item["obs"]["agent_pos"].shape == (16, 2)  # horizon=16, agent_pos_dim=2
        assert item["action"].shape == (16, 2)  # horizon=16, action_dim=2

        print(f"\n📦 Keypoint shape: {item['obs']['keypoint'].shape}")
        print(f"📦 Agent pos shape: {item['obs']['agent_pos'].shape}")
        print(f"📦 Action shape: {item['action'].shape}")

    def test_normalizer(self, real_data_path):
        """Test that normalizer is created correctly."""
        dataset = PushTKeypointDataset(
            dataset_path=real_data_path,
            horizon=16,
        )

        assert "obs" in dataset.normalizer
        assert "action" in dataset.normalizer
        assert "keypoint" in dataset.normalizer["obs"]
        assert "agent_pos" in dataset.normalizer["obs"]


class TestPushTImageDataset:
    """Test PushTImageDataset class with real HuggingFace data."""

    @pytest.fixture
    def shape_meta(self):
        """Shape metadata for image dataset."""
        return {
            "action": {"shape": [2]},
            "obs": {
                "image": {"shape": [3, 96, 96], "type": "rgb"},
                "agent_pos": {"shape": [2], "type": "low_dim"},
            },
        }

    def test_initialization(self, real_data_path, shape_meta):
        """Test that the image dataset can be initialized."""
        dataset = PushTImageDataset(
            dataset_path=real_data_path,
            shape_meta=shape_meta,
            n_obs_steps=2,
            horizon=16,
            pad_before=1,
            pad_after=7,
        )

        assert dataset is not None
        assert dataset.horizon == 16
        assert dataset.n_obs_steps == 2
        assert len(dataset.rgb_keys) == 1
        assert len(dataset.lowdim_keys) == 1

    def test_getitem(self, real_data_path, shape_meta):
        """Test that __getitem__ returns correct data structure."""
        dataset = PushTImageDataset(
            dataset_path=real_data_path,
            shape_meta=shape_meta,
            n_obs_steps=2,
            horizon=16,
            pad_before=1,
            pad_after=7,
        )

        item = dataset[0]

        # Check structure
        assert isinstance(item, dict)
        assert "obs" in item
        assert "action" in item
        assert "image" in item["obs"]
        assert "agent_pos" in item["obs"]

        # Check tensor types
        assert isinstance(item["obs"]["image"], torch.Tensor)
        assert isinstance(item["obs"]["agent_pos"], torch.Tensor)
        assert isinstance(item["action"], torch.Tensor)

        # Check shapes (T, C, H, W) for images
        assert item["obs"]["image"].shape == (2, 3, 96, 96)  # n_obs_steps=2
        assert item["obs"]["agent_pos"].shape == (2, 2)  # n_obs_steps=2
        assert item["action"].shape == (16, 2)  # horizon=16

        print(f"\n📦 Image shape: {item['obs']['image'].shape}")
        print(f"📦 Agent pos shape: {item['obs']['agent_pos'].shape}")
        print(f"📦 Action shape: {item['action'].shape}")

    def test_image_normalization(self, real_data_path, shape_meta):
        """Test that images are properly normalized to [-1, 1]."""
        dataset = PushTImageDataset(
            dataset_path=real_data_path,
            shape_meta=shape_meta,
            horizon=16,
        )

        item = dataset[0]

        # Images should be in range [-1, 1] after normalization by ImageNormalizer
        image = item["obs"]["image"]
        assert image.min() >= -1.0
        assert image.max() <= 1.0
        print(f"\n📊 Image range: [{image.min():.3f}, {image.max():.3f}]")

    def test_normalizer(self, real_data_path, shape_meta):
        """Test that normalizer is created correctly."""
        dataset = PushTImageDataset(
            dataset_path=real_data_path,
            shape_meta=shape_meta,
            horizon=16,
        )

        assert "obs" in dataset.normalizer
        assert "action" in dataset.normalizer
        assert "image" in dataset.normalizer["obs"]
        assert "agent_pos" in dataset.normalizer["obs"]


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
        assert isinstance(dataset, PushTStateDataset)

        # Check we can get an item
        item = dataset[0]
        assert "obs" in item
        assert "action" in item

        print(f"\n✅ make_dataset(state) - Dataset length: {len(dataset)}")

    def test_make_dataset_keypoint(self, real_data_path):
        """Test make_dataset with keypoint observation type."""
        from omegaconf import OmegaConf

        config_dict = {
            "obs_type": "keypoint",
            "dataset_path": real_data_path,
            "horizon": 16,
            "obs_steps": 2,
            "act_steps": 8,
        }
        config = OmegaConf.create(config_dict)

        dataset = make_dataset(config)
        assert isinstance(dataset, PushTKeypointDataset)
        assert len(dataset) > 0

        print(f"\n✅ make_dataset(keypoint) - Dataset length: {len(dataset)}")

    def test_make_dataset_image(self, real_data_path):
        """Test make_dataset with image observation type."""
        from omegaconf import OmegaConf

        config_dict = {
            "obs_type": "image",
            "dataset_path": real_data_path,
            "horizon": 16,
            "obs_steps": 2,
            "act_steps": 8,
            "shape_meta": {
                "action": {"shape": [2]},
                "obs": {
                    "image": {"shape": [3, 96, 96], "type": "rgb"},
                    "agent_pos": {"shape": [2], "type": "low_dim"},
                },
            },
        }
        config = OmegaConf.create(config_dict)

        dataset = make_dataset(config)
        assert isinstance(dataset, PushTImageDataset)
        assert len(dataset) > 0

        print(f"\n✅ make_dataset(image) - Dataset length: {len(dataset)}")
