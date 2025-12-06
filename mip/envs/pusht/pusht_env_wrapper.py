"""PushT environment wrapper for observation preprocessing and multi-step control.

Author: Chaoyi Pan
Date: 2025-10-15
"""

import gymnasium as gym
import numpy as np

from mip.config import TaskConfig
from mip.env_utils import MultiStepWrapper, VideoRecorder, VideoRecordingWrapper


def make_vec_env(task_config: TaskConfig, seed=None):
    """Create vectorized PushT environments.

    Args:
        task_config: Task configuration
        seed: Random seed for environments

    Returns:
        Vectorized gym environment
    """
    vnc_env_class = gym.vector.SyncVectorEnv
    envs = vnc_env_class(
        [
            make_pusht_env(task_config, idx, False, seed=seed)
            for idx in range(task_config.num_envs)
        ],
    )
    return envs


def make_pusht_env(task_config: TaskConfig, idx, render=False, seed=None):
    """Create a single PushT environment.

    Args:
        task_config: Task configuration
        idx: Environment index for seed offset
        render: Whether to render
        seed: Random seed

    Returns:
        Callable that creates the environment
    """
    from mip.envs.pusht.pusht_env import PushTEnv
    from mip.envs.pusht.pusht_image_env import PushTImageEnv

    def thunk():
        # Create base environment
        if task_config.obs_type == "state":
            env = PushTEnv(
                legacy=False,
                render_size=96,
                render_action=True,
            )
        elif task_config.obs_type == "keypoint":
            # For keypoint, we use the state env but will extract keypoints
            # Note: This requires custom keypoint extraction logic
            env = PushTEnv(
                legacy=False,
                render_size=96,
                render_action=True,
            )
            env = PushTKeypointWrapper(env)
        elif task_config.obs_type == "image":
            env = PushTImageEnv(
                legacy=False,
                render_size=96,
            )
        else:
            raise ValueError(f"Invalid obs_type: {task_config.obs_type}")

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
            env, video_recoder, file_path=file_path, steps_per_render=2
        )

        # Add multi-step wrapper
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


class PushTKeypointWrapper(gym.Wrapper):
    """Wrapper to extract keypoint observations from PushT environment.

    This wrapper extracts keypoints from the T-shaped block and provides them
    as observations along with the agent position.
    """

    def __init__(self, env):
        super().__init__(env)
        # Keypoint observation: 9 keypoints (18 values) + agent_pos (2 values) = 20 values
        # But we'll return a dict for consistency with image observations
        self.observation_space = gym.spaces.Box(
            low=0,
            high=512,
            shape=(20,),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._extract_keypoints(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._extract_keypoints(obs), reward, terminated, truncated, info

    def _extract_keypoints(self, obs):
        """Extract keypoints from the block pose.

        The block is a T-shape, so we extract corners and key points.
        For simplicity, we'll compute 9 keypoints from the block geometry.
        """
        # obs = [agent_x, agent_y, block_x, block_y, block_angle]
        agent_pos = obs[:2]
        block_pos = obs[2:4]
        block_angle = obs[4]

        # T-shape dimensions (from pusht_env.py)
        scale = 30
        length = 4

        # Define keypoints in local coordinates (relative to block center)
        # These are the corners and centers of the T-shape
        local_keypoints = [
            # Horizontal bar corners
            (-length * scale / 2, 0),  # left bottom
            (-length * scale / 2, scale),  # left top
            (length * scale / 2, 0),  # right bottom
            (length * scale / 2, scale),  # right top
            # Vertical bar corners
            (-scale / 2, scale),  # left middle
            (scale / 2, scale),  # right middle
            (-scale / 2, length * scale),  # left top
            (scale / 2, length * scale),  # right top
            # Center point
            (0, scale / 2),  # center
        ]

        # Transform keypoints to world coordinates
        cos_a = np.cos(block_angle)
        sin_a = np.sin(block_angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        world_keypoints = []
        for kp in local_keypoints:
            kp_rotated = rotation_matrix @ np.array(kp)
            kp_world = kp_rotated + block_pos
            world_keypoints.extend(kp_world)

        # Return concatenated observation
        keypoint_obs = np.array(world_keypoints + agent_pos.tolist(), dtype=np.float32)
        return keypoint_obs
