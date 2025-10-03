"""Example script showing how to download and use Robomimic datasets from HuggingFace.

Author: Chaoyi Pan
Date: 2025-10-03
"""

from mip.datasets.robomimic_dataset import (
    RobomimicDataset,
    RobomimicImageDataset,
    download_and_process_image_dataset,
    download_robomimic_dataset,
)


def example_lowdim_dataset():
    """Example: Download and use low-dim dataset."""
    print("=" * 80)
    print("Example 1: Low-dim dataset")
    print("=" * 80)

    # Download low-dim dataset from HuggingFace
    dataset_path = download_robomimic_dataset(
        task="lift",
        source="ph",
        dataset_type="low_dim",
    )

    # Create dataset
    dataset = RobomimicDataset(
        dataset_dir=dataset_path,
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

    print(f"Dataset: {dataset}")
    print(f"Length: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print(f"Observation shape: {sample['obs']['state'].shape}")
    print(f"Action shape: {sample['action'].shape}")
    print()


def example_image_dataset():
    """Example: Download and process image dataset."""
    print("=" * 80)
    print("Example 2: Image dataset (from demo)")
    print("=" * 80)

    # Download and process demo to image dataset
    # Note: This can take several minutes as it renders images
    print(
        "Downloading and processing demo to image dataset (this may take a few minutes)..."
    )
    image_path = download_and_process_image_dataset(
        task="lift",
        source="ph",
        camera_height=84,
        camera_width=84,
    )

    # Define shape metadata for image dataset
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
        dataset_dir=image_path,
        shape_meta=shape_meta,
        n_obs_steps=2,
        horizon=10,
        pad_before=2,
        pad_after=2,
        abs_action=True,
        rotation_rep="rotation_6d",
    )

    print(f"Dataset: {dataset}")
    print(f"Length: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print(f"Agentview image shape: {sample['obs']['agentview_image'].shape}")
    print(
        f"Robot eye-in-hand image shape: {sample['obs']['robot0_eye_in_hand_image'].shape}"
    )
    print(f"Action shape: {sample['action'].shape}")
    print()


if __name__ == "__main__":
    # Run low-dim example
    example_lowdim_dataset()

    # Note: Uncomment to run image example (it takes longer due to image rendering)
    # example_image_dataset()

    print("All examples completed!")
