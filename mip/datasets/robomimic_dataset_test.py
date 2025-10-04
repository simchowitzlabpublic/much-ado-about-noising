"""Test file for robomimic_dataset.py.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import os
import tempfile

import h5py
import numpy as np
import pytest
import torch

from mip.dataset_utils import RotationTransformer
from mip.datasets.robomimic_dataset import (
    RobomimicDataset,
    RobomimicImageDataset,
    _convert_actions,
    _data_to_obs,
    download_and_process_image_dataset,
    download_robomimic_dataset,
    process_demo_to_image_dataset,
)


def create_mock_robomimic_hdf5(
    file_path, n_demos=3, episode_length=50, include_images=True
):
    """Create a mock robomimic HDF5 file for testing."""
    with h5py.File(file_path, "w") as f:
        data_group = f.create_group("data")

        for i in range(n_demos):
            demo = data_group.create_group(f"demo_{i}")

            # Create actions (episode_length, 7) - pos(3) + rot(3) + gripper(1)
            actions = np.random.randn(episode_length, 7).astype(np.float32)
            demo.create_dataset("actions", data=actions)

            # Create dones
            dones = np.zeros(episode_length, dtype=bool)
            dones[-1] = True
            demo.create_dataset("dones", data=dones)

            # Create states
            states = np.random.randn(episode_length, 71).astype(np.float32)
            demo.create_dataset("states", data=states)

            # Create rewards
            rewards = np.random.randn(episode_length).astype(np.float32)
            demo.create_dataset("rewards", data=rewards)

            # Create obs group
            obs_group = demo.create_group("obs")

            # Low-dim observations
            obs_group.create_dataset(
                "object", data=np.random.randn(episode_length, 14).astype(np.float32)
            )
            obs_group.create_dataset(
                "robot0_eef_pos",
                data=np.random.randn(episode_length, 3).astype(np.float32),
            )
            obs_group.create_dataset(
                "robot0_eef_quat",
                data=np.random.randn(episode_length, 4).astype(np.float32),
            )
            obs_group.create_dataset(
                "robot0_gripper_qpos",
                data=np.random.randn(episode_length, 2).astype(np.float32),
            )

            if include_images:
                # RGB observations (H, W, C)
                obs_group.create_dataset(
                    "agentview_image",
                    data=np.random.randint(
                        0, 255, (episode_length, 84, 84, 3), dtype=np.uint8
                    ),
                )
                obs_group.create_dataset(
                    "robot0_eye_in_hand_image",
                    data=np.random.randint(
                        0, 255, (episode_length, 84, 84, 3), dtype=np.uint8
                    ),
                )


class TestRobomimicDataset:
    """Test RobomimicDataset class."""

    @pytest.fixture
    def temp_hdf5_file(self):
        """Create a temporary HDF5 file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            tmp_path = tmp.name

        create_mock_robomimic_hdf5(
            tmp_path, n_demos=3, episode_length=50, include_images=False
        )
        yield tmp_path

        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    def test_initialization(self, temp_hdf5_file):
        """Test that the dataset can be initialized."""
        dataset = RobomimicDataset(
            dataset_dir=temp_hdf5_file,
            horizon=10,
            pad_before=2,
            pad_after=2,
            obs_keys=(
                "object",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            ),
            abs_action=False,
            rotation_rep="rotation_6d",
            use_key_state_for_val=False,
        )

        assert dataset is not None
        assert dataset.horizon == 10
        assert dataset.pad_before == 2
        assert dataset.pad_after == 2
        assert not dataset.abs_action

    def test_len(self, temp_hdf5_file):
        """Test that __len__ returns correct length."""
        dataset = RobomimicDataset(
            dataset_dir=temp_hdf5_file,
            horizon=10,
            pad_before=2,
            pad_after=2,
            use_key_state_for_val=False,
        )

        length = len(dataset)
        assert length > 0
        # With 3 demos of 50 steps each, we should have 150 total steps
        # The sampler will determine exact indexable length
        assert isinstance(length, int)

    def test_getitem(self, temp_hdf5_file):
        """Test that __getitem__ returns correct data structure."""
        dataset = RobomimicDataset(
            dataset_dir=temp_hdf5_file,
            horizon=10,
            pad_before=2,
            pad_after=2,
            use_key_state_for_val=False,
        )

        # Get first item
        item = dataset[0]

        # Check structure
        assert isinstance(item, dict)
        assert "obs" in item
        assert "action" in item

        # Check that obs contains state
        assert "state" in item["obs"]

        # Check tensor types
        assert isinstance(item["obs"]["state"], torch.Tensor)
        assert isinstance(item["action"], torch.Tensor)

        # Check shapes
        assert item["obs"]["state"].shape[0] == 10  # horizon
        assert item["action"].shape[0] == 10  # horizon

    def test_abs_action(self, temp_hdf5_file):
        """Test with absolute action mode."""
        dataset = RobomimicDataset(
            dataset_dir=temp_hdf5_file,
            horizon=5,
            abs_action=True,
            rotation_rep="rotation_6d",
            use_key_state_for_val=False,
        )

        item = dataset[0]

        # With rotation_6d, action should have dimension: pos(3) + rot(6) + gripper(1) = 10
        assert item["action"].shape[-1] == 10

    def test_train_val_split(self, temp_hdf5_file):
        """Test train/val split functionality."""
        # Train dataset (use 50% split to ensure both sets have data)
        train_dataset = RobomimicDataset(
            dataset_dir=temp_hdf5_file,
            horizon=5,
            val_dataset_percentage=0.5,
            mode="train",
            use_key_state_for_val=False,
        )

        # Val dataset
        val_dataset = RobomimicDataset(
            dataset_dir=temp_hdf5_file,
            horizon=5,
            val_dataset_percentage=0.5,
            mode="val",
            use_key_state_for_val=False,
        )

        # Both should be non-empty
        assert len(train_dataset) > 0
        assert len(val_dataset) > 0

        # Can index both
        train_item = train_dataset[0]
        val_item = val_dataset[0]

        assert train_item is not None
        assert val_item is not None

    def test_undo_transform_action(self, temp_hdf5_file):
        """Test action transformation and inverse."""
        dataset = RobomimicDataset(
            dataset_dir=temp_hdf5_file,
            horizon=5,
            abs_action=True,
            rotation_rep="rotation_6d",
            use_key_state_for_val=False,
        )

        item = dataset[0]
        action = item["action"].numpy()

        # Denormalize
        action_denorm = dataset.normalizer["action"].unnormalize(action)

        # Undo transform
        original_action = dataset.undo_transform_action(action_denorm)

        # Should have original dimensionality: pos(3) + rot(3) + gripper(1) = 7
        assert original_action.shape[-1] == 7


class TestRobomimicImageDataset:
    """Test RobomimicImageDataset class."""

    @pytest.fixture
    def temp_hdf5_file_with_images(self):
        """Create a temporary HDF5 file with images for testing."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            tmp_path = tmp.name

        create_mock_robomimic_hdf5(
            tmp_path, n_demos=2, episode_length=30, include_images=True
        )
        yield tmp_path

        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    @pytest.fixture
    def shape_meta(self):
        """Shape metadata for image dataset."""
        return {
            "action": {"shape": [10]},
            "obs": {
                "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
                "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
                "robot0_eef_pos": {"shape": [3], "type": "low_dim"},
                "robot0_eef_quat": {"shape": [4], "type": "low_dim"},
                "robot0_gripper_qpos": {"shape": [2], "type": "low_dim"},
            },
        }

    def test_initialization(self, temp_hdf5_file_with_images, shape_meta):
        """Test that the image dataset can be initialized."""
        dataset = RobomimicImageDataset(
            dataset_dir=temp_hdf5_file_with_images,
            shape_meta=shape_meta,
            n_obs_steps=2,
            horizon=10,
            pad_before=2,
            pad_after=2,
            abs_action=True,
            rotation_rep="rotation_6d",
        )

        assert dataset is not None
        assert dataset.horizon == 10
        assert dataset.n_obs_steps == 2
        assert len(dataset.rgb_keys) == 2
        assert len(dataset.lowdim_keys) == 3

    def test_len(self, temp_hdf5_file_with_images, shape_meta):
        """Test that __len__ returns correct length."""
        dataset = RobomimicImageDataset(
            dataset_dir=temp_hdf5_file_with_images,
            shape_meta=shape_meta,
            horizon=5,
            abs_action=True,  # Use abs_action=True to match expected action shape of 10
        )

        length = len(dataset)
        assert length > 0
        assert isinstance(length, int)

    def test_getitem(self, temp_hdf5_file_with_images, shape_meta):
        """Test that __getitem__ returns correct data structure."""
        dataset = RobomimicImageDataset(
            dataset_dir=temp_hdf5_file_with_images,
            shape_meta=shape_meta,
            n_obs_steps=2,
            horizon=10,
            abs_action=True,  # Use abs_action=True to match expected action shape of 10
        )

        # Get first item
        item = dataset[0]

        # Check structure
        assert isinstance(item, dict)
        assert "obs" in item
        assert "action" in item

        # Check that obs contains images and low-dim
        assert "agentview_image" in item["obs"]
        assert "robot0_eye_in_hand_image" in item["obs"]
        assert "robot0_eef_pos" in item["obs"]
        assert "robot0_eef_quat" in item["obs"]
        assert "robot0_gripper_qpos" in item["obs"]

        # Check tensor types
        for key in dataset.rgb_keys + dataset.lowdim_keys:
            assert isinstance(item["obs"][key], torch.Tensor)
        assert isinstance(item["action"], torch.Tensor)

        # Check image shapes (T, C, H, W)
        assert item["obs"]["agentview_image"].shape == (2, 3, 84, 84)  # n_obs_steps=2
        assert item["obs"]["robot0_eye_in_hand_image"].shape == (2, 3, 84, 84)

        # Check low-dim shapes (T, D)
        assert item["obs"]["robot0_eef_pos"].shape == (2, 3)  # n_obs_steps=2

        # Check action shape
        assert item["action"].shape[0] == 10  # horizon

    def test_image_normalization(self, temp_hdf5_file_with_images, shape_meta):
        """Test that images are properly normalized to [0, 1]."""
        dataset = RobomimicImageDataset(
            dataset_dir=temp_hdf5_file_with_images,
            shape_meta=shape_meta,
            horizon=5,
            abs_action=True,  # Use abs_action=True to match expected action shape of 10
        )

        item = dataset[0]

        # Images should be in range [-1, 1] after normalization by ImageNormalizer
        for key in dataset.rgb_keys:
            image = item["obs"][key]
            assert image.min() >= -1.0
            assert image.max() <= 1.0


class TestHelperFunctions:
    """Test helper functions."""

    def test_data_to_obs_relative_action(self):
        """Test _data_to_obs with relative action."""
        raw_obs = {
            "robot0_eef_pos": np.random.randn(10, 3).astype(np.float32),
            "robot0_gripper_qpos": np.random.randn(10, 2).astype(np.float32),
        }
        raw_actions = np.random.randn(10, 7).astype(np.float32)
        obs_keys = ("robot0_eef_pos", "robot0_gripper_qpos")
        rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep="rotation_6d"
        )

        result = _data_to_obs(
            raw_obs=raw_obs,
            raw_actions=raw_actions,
            obs_keys=obs_keys,
            abs_action=False,
            rotation_transformer=rotation_transformer,
        )

        assert "obs" in result
        assert "action" in result
        assert result["obs"].shape == (10, 5)  # 3 + 2
        assert result["action"].shape == (10, 7)  # unchanged for relative action

    def test_data_to_obs_absolute_action(self):
        """Test _data_to_obs with absolute action."""
        raw_obs = {
            "robot0_eef_pos": np.random.randn(10, 3).astype(np.float32),
        }
        raw_actions = np.random.randn(10, 7).astype(np.float32)
        obs_keys = ("robot0_eef_pos",)
        rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep="rotation_6d"
        )

        result = _data_to_obs(
            raw_obs=raw_obs,
            raw_actions=raw_actions,
            obs_keys=obs_keys,
            abs_action=True,
            rotation_transformer=rotation_transformer,
        )

        assert "obs" in result
        assert "action" in result
        assert result["obs"].shape == (10, 3)
        assert result["action"].shape == (10, 10)  # pos(3) + rot(6) + gripper(1)

    def test_convert_actions(self):
        """Test _convert_actions function."""
        raw_actions = np.random.randn(10, 7).astype(np.float32)
        rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep="rotation_6d"
        )

        # Test with abs_action=False
        result = _convert_actions(
            raw_actions=raw_actions,
            abs_action=False,
            rotation_transformer=rotation_transformer,
        )
        assert result.shape == (10, 7)

        # Test with abs_action=True
        result = _convert_actions(
            raw_actions=raw_actions,
            abs_action=True,
            rotation_transformer=rotation_transformer,
        )
        assert result.shape == (10, 10)  # pos(3) + rot(6) + gripper(1)


class TestWithRealData:
    """Test with real robomimic data from Hugging Face."""

    @pytest.fixture(scope="class")
    def real_lowdim_data_path(self):
        """Download real low-dim robomimic dataset from Hugging Face."""
        try:
            # Download low-dim dataset
            path = download_robomimic_dataset(
                task="lift",
                source="ph",
                dataset_type="low_dim",
            )
            return path
        except Exception as e:
            pytest.skip(f"Failed to download real data: {e}")

    @pytest.fixture(scope="class")
    def real_demo_data_path(self):
        """Download real demo robomimic dataset from Hugging Face."""
        try:
            # Download demo dataset
            path = download_robomimic_dataset(
                task="lift",
                source="ph",
                dataset_type="demo",
            )
            return path
        except Exception as e:
            pytest.skip(f"Failed to download demo data: {e}")

    def test_download_lowdim_dataset(self):
        """Test downloading low-dim dataset from Hugging Face."""
        path = download_robomimic_dataset(
            task="lift",
            source="ph",
            dataset_type="low_dim",
        )
        assert os.path.exists(path)
        assert path.endswith(".hdf5")

        # Verify it's a valid HDF5 file
        with h5py.File(path, "r") as f:
            assert "data" in f

    def test_download_demo_dataset(self):
        """Test downloading demo dataset from Hugging Face."""
        path = download_robomimic_dataset(
            task="lift",
            source="ph",
            dataset_type="demo",
        )
        assert os.path.exists(path)
        assert path.endswith(".hdf5")

        # Verify it's a valid HDF5 file with states
        with h5py.File(path, "r") as f:
            assert "data" in f
            demo_0 = f["data"]["demo_0"]
            assert "states" in demo_0

    def test_with_real_lowdim_data(self, real_lowdim_data_path):
        """Test dataset with real low-dim robomimic data from HF."""
        dataset = RobomimicDataset(
            dataset_dir=real_lowdim_data_path,
            horizon=10,
            pad_before=2,
            pad_after=2,
            obs_keys=(
                "object",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            ),
            abs_action=True,
            rotation_rep="rotation_6d",
            use_key_state_for_val=False,
        )

        # Check dataset can be initialized
        assert len(dataset) > 0

        # Check we can index it
        item = dataset[0]
        assert item is not None
        assert "obs" in item
        assert "action" in item

        # Print some info
        print(f"\nDataset info: {dataset}")
        print(f"Dataset length: {len(dataset)}")
        print(f"Obs state shape: {item['obs']['state'].shape}")
        print(f"Action shape: {item['action'].shape}")

    @pytest.mark.skipif(
        os.environ.get("DISPLAY") is None and os.environ.get("MUJOCO_GL") != "osmesa",
        reason="Requires display or osmesa for rendering",
    )
    def test_process_demo_to_image(self, real_demo_data_path):
        """Test processing demo file to image dataset."""
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            output_path = tmp.name

        try:
            # Process demo to image dataset
            processed_path = process_demo_to_image_dataset(
                demo_path=real_demo_data_path,
                output_path=output_path,
                task="lift",
                camera_height=84,
                camera_width=84,
            )

            assert os.path.exists(processed_path)

            # Verify it's a valid image dataset
            with h5py.File(processed_path, "r") as f:
                assert "data" in f
                demo_0 = f["data"]["demo_0"]
                assert "obs" in demo_0
                # Print available keys for debugging
                obs_keys = list(demo_0["obs"].keys())
                print(f"\nAvailable observation keys: {obs_keys}")
                # Should have image observations
                # Note: robomimic appends "_image" to camera names
                # Check for at least one image observation
                image_keys = [k for k in obs_keys if k.endswith("_image")]
                assert len(image_keys) > 0, (
                    f"No image observations found. Available keys: {obs_keys}"
                )
                assert "robot0_eye_in_hand_image" in demo_0["obs"]
                print(f"Successfully found image observations: {image_keys}")

        finally:
            # Cleanup
            if os.path.exists(output_path):
                os.remove(output_path)

    @pytest.mark.skipif(
        os.environ.get("DISPLAY") is None and os.environ.get("MUJOCO_GL") != "osmesa",
        reason="Requires display or osmesa for rendering",
    )
    def test_download_and_process_image_dataset(self):
        """Test convenience function to download and process image dataset."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            output_path = tmp.name

        try:
            # Use convenience function
            processed_path = download_and_process_image_dataset(
                task="lift",
                source="ph",
                output_path=output_path,
                camera_height=84,
                camera_width=84,
            )

            assert os.path.exists(processed_path)

            # Verify it's a valid image dataset
            with h5py.File(processed_path, "r") as f:
                assert "data" in f
                demo_0 = f["data"]["demo_0"]
                assert "obs" in demo_0
                # Should have image observations
                assert "agentview_image" in demo_0["obs"]
                assert "robot0_eye_in_hand_image" in demo_0["obs"]

            print(f"\nSuccessfully created image dataset at: {processed_path}")

        finally:
            # Cleanup
            if os.path.exists(output_path):
                os.remove(output_path)

    @pytest.mark.skipif(
        os.environ.get("DISPLAY") is None and os.environ.get("MUJOCO_GL") != "osmesa",
        reason="Requires display or osmesa for rendering",
    )
    def test_with_real_image_data(self, real_demo_data_path):
        """Test RobomimicImageDataset with real data from HF."""
        # Process demo to image dataset
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            output_path = tmp.name

        try:
            processed_path = process_demo_to_image_dataset(
                demo_path=real_demo_data_path,
                output_path=output_path,
                task="lift",
                camera_height=84,
                camera_width=84,
            )

            # Define shape metadata
            shape_meta = {
                "action": {"shape": [10]},  # rotation_6d: pos(3) + rot(6) + gripper(1)
                "obs": {
                    "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
                    "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
                    "robot0_eef_pos": {"shape": [3], "type": "low_dim"},
                    "robot0_eef_quat": {"shape": [4], "type": "low_dim"},
                    "robot0_gripper_qpos": {"shape": [2], "type": "low_dim"},
                },
            }

            # Create image dataset
            dataset = RobomimicImageDataset(
                dataset_dir=processed_path,
                shape_meta=shape_meta,
                n_obs_steps=2,
                horizon=10,
                pad_before=2,
                pad_after=2,
                abs_action=True,
                rotation_rep="rotation_6d",
            )

            # Check dataset can be initialized
            assert len(dataset) > 0

            # Check we can index it
            item = dataset[0]
            assert item is not None
            assert "obs" in item
            assert "action" in item

            # Check image observations
            assert "agentview_image" in item["obs"]
            assert "robot0_eye_in_hand_image" in item["obs"]
            assert item["obs"]["agentview_image"].shape == (2, 3, 84, 84)

            # Print some info
            print(f"\nImage Dataset info: {dataset}")
            print(f"Dataset length: {len(dataset)}")
            print(f"Agentview image shape: {item['obs']['agentview_image'].shape}")
            print(f"Action shape: {item['action'].shape}")

        finally:
            # Cleanup
            if os.path.exists(output_path):
                os.remove(output_path)
