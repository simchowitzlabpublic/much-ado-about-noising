"""Kitchen environment wrapper for observation preprocessing and multi-step control.

Uses Gymnasium-Robotics FrankaKitchen-v1 environment.
Reference: https://robotics.farama.org/envs/franka_kitchen/

Author: Chaoyi Pan
Date: 2025-10-17
"""

import gymnasium as gym

from mip.config import TaskConfig
from mip.env_utils import MultiStepWrapper, VideoRecorder, VideoRecordingWrapper

# Map from task config env_name to Gymnasium-Robotics tasks
KITCHEN_TASK_MAPPING = {
    "kitchen-all-v0": None,  # All tasks (default)
    "kitchen-microwave-kettle-light-slider-v0": [
        "microwave",
        "kettle",
        "light switch",
        "slide cabinet",
    ],
    "kitchen-microwave-kettle-burner-light-v0": [
        "microwave",
        "kettle",
        "bottom burner",
        "light switch",
    ],
    "kitchen-kettle-microwave-light-slider-v0": [
        "kettle",
        "microwave",
        "light switch",
        "slide cabinet",
    ],
}


def make_vec_env(task_config: TaskConfig, seed=None):
    """Create vectorized Kitchen environments using Gymnasium-Robotics.

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
            make_kitchen_env(task_config, idx, False, seed=seed)
            for idx in range(task_config.num_envs)
        ],
    )
    return envs


def make_kitchen_env(task_config: TaskConfig, idx, render=False, seed=None):
    """Create a single Kitchen environment using Gymnasium-Robotics.

    Args:
        task_config: Task configuration
        idx: Environment index for seed offset
        render: Whether to render
        seed: Random seed

    Returns:
        Callable that creates the environment
    """

    def thunk():
        # Import and register gymnasium_robotics environments
        import gymnasium_robotics

        gym.register_envs(gymnasium_robotics)

        # Get task list from mapping
        tasks = KITCHEN_TASK_MAPPING.get(task_config.env_name)

        # Create Gymnasium-Robotics FrankaKitchen environment
        if tasks is not None:
            env = gym.make(
                "FrankaKitchen-v1",
                tasks_to_complete=tasks,
                max_episode_steps=task_config.max_episode_steps,
            )
        else:
            # Use all tasks (default)
            env = gym.make(
                "FrankaKitchen-v1",
                max_episode_steps=task_config.max_episode_steps,
            )

        # Wrap to extract observation from dict and handle info
        env = KitchenObservationWrapper(env)

        # Add video recording wrapper
        video_recoder = VideoRecorder.create_h264(
            fps=10,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=22,
            thread_type="FRAME",
            thread_count=1,
        )
        file_path = None if not render else "results/video.mp4"
        env = VideoRecordingWrapper(
            env, video_recoder, file_path=file_path, steps_per_render=1
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
            env.reset(seed=seed + idx)

        return env

    return thunk


class KitchenObservationWrapper(gym.ObservationWrapper):
    """Wrapper to extract flat observation from Gymnasium-Robotics dict observation.

    The FrankaKitchen-v1 environment returns a dict observation with keys:
    - observation: (59,) array = qpos(9) + qvel(9) + obj_qpos(21) + obj_qvel(20)
    - achieved_goal: dict with current task states
    - desired_goal: dict with target task states

    This wrapper extracts and reformats to match the old adept_envs dataset format:
    - robot_qpos: 9 dims (indices 0:9)
    - object_qpos: 21 dims (indices 18:39)
    - padding: 30 dims (zeros for compatibility with dataset)
    Total: 60 dimensions

    The dataset was collected with adept_envs which had: qpos(9) + obj_qpos(21) + zeros(30).
    """

    def __init__(self, env):
        super().__init__(env)
        import numpy as np
        from gymnasium.spaces import Box

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(60,),
            dtype=np.float32,
        )

    def observation(self, obs):
        """Extract and reformat observation to match dataset (60 dimensions)."""
        import numpy as np

        gym_obs = obs["observation"]  # 59 dims from Gymnasium-Robotics

        # Extract the relevant parts to match adept_envs format
        robot_qpos = gym_obs[0:9]       # robot joint positions
        obj_qpos = gym_obs[18:39]       # object states (21 dims)

        # Construct observation matching dataset: qpos(9) + obj_qpos(21) + zeros(30)
        reformatted_obs = np.concatenate([
            robot_qpos,    # 9 dims
            obj_qpos,      # 21 dims
            np.zeros(30),  # 30 dims padding
        ], axis=0).astype(np.float32)

        return reformatted_obs

    def step(self, action):
        """Step environment and add completed_tasks to info."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add completed_tasks list to info for compatibility
        # Gymnasium-Robotics uses 'episode_task_completions' key
        if "episode_task_completions" in info:
            info["completed_tasks"] = [info["episode_task_completions"]]
        else:
            info["completed_tasks"] = [[]]

        return self.observation(obs), reward, terminated, truncated, info
