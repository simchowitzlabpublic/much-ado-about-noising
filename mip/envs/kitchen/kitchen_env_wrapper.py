"""Kitchen environment wrapper for observation preprocessing and multi-step control.

Author: Chaoyi Pan
Date: 2025-10-17
"""

import gymnasium as gym

from mip.config import TaskConfig
from mip.env_utils import MultiStepWrapper, VideoRecorder, VideoRecordingWrapper
from mip.envs.kitchen.kitchen_thirdparty.kitchen_lowdim_wrapper import (
    KitchenLowdimWrapper,
)


def make_vec_env(task_config: TaskConfig, seed=None):
    """Create vectorized Kitchen environments using CleanDiffuser.

    Args:
        task_config: Task configuration
        seed: Random seed for environments

    Returns:
        Vectorized gym environment
    """
    # Use SyncVectorEnv when num_envs=1 or save_video=True
    if task_config.num_envs == 1 or task_config.save_video:
        vnc_env_class = gym.vector.SyncVectorEnv
    else:
        vnc_env_class = gym.vector.AsyncVectorEnv

    envs = vnc_env_class(
        [
            make_kitchen_env(task_config, idx, task_config.save_video, seed=seed)
            for idx in range(task_config.num_envs)
        ],
    )
    return envs


def make_kitchen_env(task_config: TaskConfig, idx, render=False, seed=None):
    """Create a single Kitchen environment using CleanDiffuser.

    Args:
        task_config: Task configuration
        idx: Environment index for seed offset
        render: Whether to render
        seed: Random seed

    Returns:
        Callable that creates the environment
    """

    def thunk():
        # Import old gym (not gymnasium) for CleanDiffuser compatibility
        import gym as old_gym

        # Create CleanDiffuser kitchen environment
        # The env_name should be one of: kitchen-all-v0, kitchen-microwave-kettle-light-slider-v0, etc.
        # use_abs_action=True for absolute actions (as used in CleanDiffuser's MJL dataset)
        env = old_gym.make(task_config.env_name, use_abs_action=True)

        # Wrap with CleanDiffuser's KitchenLowdimWrapper
        render_hw = (240, 360) if render else (240, 360)
        env = KitchenLowdimWrapper(
            env=env, init_qpos=None, init_qvel=None, render_hw=render_hw
        )

        # Add video recording wrapper
        video_recorder = VideoRecorder.create_h264(
            fps=12.5,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=22,
            thread_type="FRAME",
            thread_count=1,
        )
        file_path = None if not render else "results/video.mp4"
        env = VideoRecordingWrapper(
            env, video_recorder, file_path=file_path, steps_per_render=1
        )

        # Add multi-step wrapper (IMPORTANT: must be after video recording)
        env = MultiStepWrapper(
            env,
            n_obs_steps=task_config.obs_steps,
            n_action_steps=task_config.act_steps,
            max_episode_steps=task_config.max_episode_steps,
        )

        # Set seed
        if seed is not None:
            env.seed(seed + idx)

        return env

    return thunk
