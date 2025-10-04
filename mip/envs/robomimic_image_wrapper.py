"""Image wrapper for Robomimic environments. Ported from https://github.com/CleanDiffuserTeam/CleanDiffuser"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger
from omegaconf import OmegaConf
from robomimic.envs.env_robosuite import EnvRobosuite


class RobomimicImageWrapper(gym.Env):
    def __init__(
        self,
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: np.ndarray | None = None,
        render_obs_key="agentview_image",
    ):
        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False

        # setup spaces
        action_shape = shape_meta["action"]["shape"]
        action_space = spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            min_value, max_value = -1, 1
            if key.endswith("image"):
                min_value, max_value = 0, 1
            elif key.endswith("quat") or key.endswith("qpos"):
                min_value, max_value = -1, 1
            elif key.endswith("pos"):
                # better range?
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")

            this_space = spaces.Box(
                low=min_value, high=max_value, shape=shape, dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()

        # Debug: Check what keys are available
        # print(f"DEBUG: Available keys in raw_obs: {list(raw_obs.keys()) if raw_obs else 'None'}")
        # print(f"DEBUG: Looking for render_obs_key: {self.render_obs_key}")

        # Handle render cache key mapping
        render_key = self.render_obs_key
        if render_key not in raw_obs:
            # Try without _image suffix
            if render_key.endswith('_image'):
                base_key = render_key.replace('_image', '')
                if base_key in raw_obs:
                    render_key = base_key
                else:
                    logger.error(f"ERROR: Neither '{render_key}' nor '{base_key}' found in raw_obs keys: {list(raw_obs.keys())}")
                    # Use first image key as fallback
                    for key in raw_obs.keys():
                        if 'image' in key.lower() or key in ['agentview', 'robot0_eye_in_hand']:
                            render_key = key
                            logger.warning(f"Using fallback render key: {render_key}")
                            break
        self.render_cache = raw_obs[render_key]

        obs = dict()
        # Debug: print available keys
        # print(f"Available keys in raw_obs: {raw_obs.keys()}")
        # print(f"Expected keys in observation_space: {self.observation_space.keys()}")

        for key in self.observation_space.keys():
            # Map dataset keys to environment keys
            # Dataset has keys like 'robot0_eye_in_hand_image' but env provides 'robot0_eye_in_hand'
            # Also 'agentview_image' -> 'agentview'
            env_key = key
            if key.endswith('_image'):
                # Try removing '_image' suffix for camera observations
                base_key = key.replace('_image', '')
                if base_key in raw_obs:
                    env_key = base_key

            # Check if this key exists in raw_obs
            if env_key not in raw_obs:
                logger.warning(f"Warning: Observation key '{key}' (mapped to '{env_key}') not found in raw observations. Available keys: {list(raw_obs.keys())}")
                # Skip this key if not found
                continue
            else:
                obs[key] = raw_obs[env_key]
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self, seed=None, options=None):
        # Handle seed parameter from gymnasium API
        if seed is not None:
            self.seed(seed)

        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            # to be compatible with gym
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                raw_obs = self.env.reset_to({"states": self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()["states"]
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()

        # return obs
        obs = self.get_observation(raw_obs)
        return obs

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        img = np.moveaxis(self.render_cache, 0, -1)
        img = (img * 255).astype(np.uint8)
        return img
