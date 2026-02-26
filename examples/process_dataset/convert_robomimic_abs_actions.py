#!/usr/bin/env python
"""Convert robomimic delta actions to absolute actions.

Downloads a low_dim.hdf5 dataset from HuggingFace, replays each demo through
the simulator to compute absolute goal positions/orientations, and saves/uploads
the result as low_dim_abs.hdf5.

Ported from diffusion_policy's RobomimicAbsoluteActionConverter.

Usage:
    MUJOCO_GL=egl uv run examples/process_dataset/convert_robomimic_abs_actions.py \
        --task lift --source ph
"""

import argparse
import copy
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
from huggingface_hub import hf_hub_download
from loguru import logger
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory

# Reuse constants from the existing processing script
PROCESSED_REPO_ID = "ChaoyiPan/mip-dataset"

# Tasks that have been validated for conversion
SUPPORTED_TASKS = ["lift", "can", "square", "transport", "tool_hang"]


class AbsoluteActionConverter:
    """Converts delta actions to absolute actions by replaying through the simulator."""

    def __init__(self, dataset_path, algo_name="bc"):
        config = config_factory(algo_name=algo_name)
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        abs_env_meta = copy.deepcopy(env_meta)
        abs_env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        assert len(env.env.robots) in (1, 2)

        abs_env = EnvUtils.create_env_from_metadata(
            env_meta=abs_env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )

        self.env = env
        self.abs_env = abs_env

    @staticmethod
    def _get_arm_controller(robot):
        """Get the arm controller from a robot, handling both robosuite API versions."""
        # robosuite v1.5+: part_controllers dict
        if hasattr(robot, "part_controllers"):
            # Single arm: 'right', Dual arm: 'right' and 'left'
            for name in ("right", "left"):
                if name in robot.part_controllers:
                    return robot.part_controllers[name]
        # robosuite < v1.5: direct controller attribute
        if hasattr(robot, "controller"):
            return robot.controller
        raise AttributeError(f"Cannot find arm controller on {type(robot)}")

    def convert_actions(self, states, actions):
        """Convert delta actions to absolute goal positions/orientations.

        For each timestep: reset env to the recorded state, feed the delta action
        through the controller, then read controller.goal_pos and controller.goal_ori.

        Args:
            states: (N, state_dim) array of simulator states
            actions: (N, 7) or (N, 14) array of delta actions

        Returns:
            abs_actions: same shape as actions, with absolute targets
        """
        # Handle multi-robot: reshape (N,14) to (N,2,7) or (N,7) to (N,1,7)
        stacked_actions = actions.reshape(*actions.shape[:-1], -1, 7)

        env = self.env
        action_goal_pos = np.zeros(
            stacked_actions.shape[:-1] + (3,), dtype=stacked_actions.dtype
        )
        action_goal_ori = np.zeros(
            stacked_actions.shape[:-1] + (3,), dtype=stacked_actions.dtype
        )
        action_gripper = stacked_actions[..., [-1]]

        for i in range(len(states)):
            _ = env.reset_to({"states": states[i]})

            for idx, robot in enumerate(env.env.robots):
                robot.control(stacked_actions[i, idx], policy_step=True)

                controller = self._get_arm_controller(robot)
                action_goal_pos[i, idx] = controller.goal_pos
                action_goal_ori[i, idx] = Rotation.from_matrix(
                    controller.goal_ori
                ).as_rotvec()

        stacked_abs_actions = np.concatenate(
            [action_goal_pos, action_goal_ori, action_gripper], axis=-1
        )
        abs_actions = stacked_abs_actions.reshape(actions.shape)
        return abs_actions

    def convert_and_eval(self, states, actions, robot0_eef_pos, robot0_eef_quat):
        """Convert actions and evaluate rollout error for verification."""
        abs_actions = self.convert_actions(states, actions)

        eval_skip_steps = 1

        delta_error = self._evaluate_rollout_error(
            self.env,
            states,
            actions,
            robot0_eef_pos,
            robot0_eef_quat,
            metric_skip_steps=eval_skip_steps,
        )
        abs_error = self._evaluate_rollout_error(
            self.abs_env,
            states,
            abs_actions,
            robot0_eef_pos,
            robot0_eef_quat,
            metric_skip_steps=eval_skip_steps,
        )

        return abs_actions, {"delta_max_error": delta_error, "abs_max_error": abs_error}

    @staticmethod
    def _evaluate_rollout_error(
        env, states, actions, robot0_eef_pos, robot0_eef_quat, metric_skip_steps=1
    ):
        rollout_next_eef_pos = []
        rollout_next_eef_quat = []

        for i in range(len(states)):
            _ = env.reset_to({"states": states[i]})
            obs, _, _, _ = env.step(actions[i])
            obs = env.get_observation()
            rollout_next_eef_pos.append(obs["robot0_eef_pos"])
            rollout_next_eef_quat.append(obs["robot0_eef_quat"])

        rollout_next_eef_pos = np.array(rollout_next_eef_pos)
        rollout_next_eef_quat = np.array(rollout_next_eef_quat)

        next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
        next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
        max_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

        next_eef_rot_diff = Rotation.from_quat(
            robot0_eef_quat[1:]
        ) * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
        next_eef_rot_dist = next_eef_rot_diff.magnitude()
        max_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()

        return {"pos": float(max_pos_dist), "rot": float(max_rot_dist)}


def download_dataset(task, source, dataset_type="low_dim"):
    """Download an HDF5 dataset from HuggingFace."""
    repo_path = f"robomimic/{task}/{source}/{dataset_type}.hdf5"
    logger.info(f"Downloading {PROCESSED_REPO_ID}/{repo_path}...")
    local_path = hf_hub_download(
        repo_id=PROCESSED_REPO_ID,
        filename=repo_path,
        repo_type="dataset",
    )
    logger.info(f"Downloaded to: {local_path}")
    return local_path


def convert_dataset(input_path, output_path, lowdim_path=None, eval_first_n=2):
    """Convert all demos in an HDF5 dataset from delta to absolute actions.

    Args:
        input_path: Path to input HDF5 with delta actions
        output_path: Path to write output HDF5 with absolute actions
        lowdim_path: Path to low_dim.hdf5 for env metadata (needed for image
            datasets which may not have env metadata). If None, uses input_path.
        eval_first_n: Number of demos to run full eval on (for verification)
    """
    # Copy the original file so we preserve all metadata
    shutil.copy2(input_path, output_path)

    # For image datasets, we need the low_dim file for env metadata (states).
    # The converter creates envs from the dataset's env_meta.
    # Image HDF5 files also contain states and actions, so we can use them directly.
    converter = AbsoluteActionConverter(lowdim_path or input_path)

    with h5py.File(input_path, "r") as f_in:
        n_demos = len(f_in["data"])
        logger.info(f"Converting {n_demos} demos from delta to absolute actions")

    with h5py.File(output_path, "r+") as f_out:
        demos = f_out["data"]

        for i in tqdm(range(n_demos), desc="Converting demos"):
            demo = demos[f"demo_{i}"]
            states = demo["states"][:]
            actions = demo["actions"][:]

            if i < eval_first_n:
                # Run full eval on first few demos to verify correctness
                robot0_eef_pos = demo["obs"]["robot0_eef_pos"][:]
                robot0_eef_quat = demo["obs"]["robot0_eef_quat"][:]

                abs_actions, info = converter.convert_and_eval(
                    states, actions, robot0_eef_pos, robot0_eef_quat
                )
                logger.info(f"Demo {i} eval: {info}")
            else:
                abs_actions = converter.convert_actions(states, actions)

            # Overwrite actions in place
            demo["actions"][...] = abs_actions

    # Log action range statistics for verification
    with h5py.File(output_path, "r") as f:
        all_actions = []
        for i in range(min(10, n_demos)):
            all_actions.append(f["data"][f"demo_{i}"]["actions"][:])
        all_actions = np.concatenate(all_actions, axis=0)

        logger.info(f"Absolute action stats (first 10 demos):")
        logger.info(f"  Position (dims 0-2): min={all_actions[:, :3].min():.4f}, "
                     f"max={all_actions[:, :3].max():.4f}, "
                     f"mean={all_actions[:, :3].mean():.4f}")
        logger.info(f"  Rotation (dims 3-5): min={all_actions[:, 3:6].min():.4f}, "
                     f"max={all_actions[:, 3:6].max():.4f}")
        logger.info(f"  Gripper  (dim 6):    min={all_actions[:, 6].min():.4f}, "
                     f"max={all_actions[:, 6].max():.4f}")


def upload_to_hub(local_path, task, source, dataset_type="low_dim", repo_id=PROCESSED_REPO_ID):
    """Upload the converted dataset to HuggingFace Hub."""
    from huggingface_hub import HfApi, upload_file

    repo_path = f"robomimic/{task}/{source}/{dataset_type}_abs.hdf5"

    try:
        api = HfApi()
        api.whoami()
    except Exception:
        logger.error("Not authenticated with HuggingFace Hub. Skipping upload.")
        logger.info("To enable uploading, run: huggingface-cli login")
        return repo_path

    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        logger.error(f"Could not create/access repo: {e}")
        return repo_path

    logger.info(f"Uploading to {repo_id}/{repo_path}...")
    try:
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
        )
        logger.info(f"Uploaded successfully to {repo_id}/{repo_path}")
    except Exception as e:
        logger.error(f"Upload failed: {e}")

    return repo_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert robomimic delta actions to absolute actions"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=SUPPORTED_TASKS,
        help="Task name (e.g., lift, can, square)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Data source (e.g., ph, mh)",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="low_dim",
        choices=["low_dim", "image"],
        help="Dataset type to convert (low_dim or image)",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Path to input HDF5. If not provided, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to a temp directory.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading to HuggingFace",
    )
    parser.add_argument(
        "--eval-demos",
        type=int,
        default=2,
        help="Number of demos to run full evaluation on (for verification)",
    )
    args = parser.parse_args()

    dataset_type = args.dataset_type

    # Download or use provided input
    if args.input_path:
        input_path = args.input_path
    else:
        input_path = download_dataset(args.task, args.source, dataset_type)

    # For image datasets, we also need low_dim for env metadata
    lowdim_path = None
    if dataset_type == "image":
        lowdim_path = download_dataset(args.task, args.source, "low_dim")

    # Set up output path
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(tempfile.mkdtemp())

    output_path = str(output_dir / f"{dataset_type}_abs.hdf5")
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")

    # Convert
    convert_dataset(input_path, output_path, lowdim_path=lowdim_path, eval_first_n=args.eval_demos)

    # Upload
    if not args.no_upload:
        upload_to_hub(output_path, args.task, args.source, dataset_type=dataset_type)
    else:
        logger.info(f"Skipping upload. Output at: {output_path}")

    logger.info("Done!")
    return output_path


if __name__ == "__main__":
    main()
