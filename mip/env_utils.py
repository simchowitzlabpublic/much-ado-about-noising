"""Environment utilities.

Port from https://github.com/CleanDiffuserTeam/CleanDiffuser

Author: Chaoyi Pan
Date: 2025-10-03
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

import av
import dill
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ------------------ MultiStepWrapper ------------------------


def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x, axis=0), n, axis=0)


def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype,
    )


def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f"Unsupported space type {type(space)}")


def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])


def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def aggregate(data, method="max"):
    if method == "max":
        # equivalent to any
        return np.max(data)
    elif method == "min":
        # equivalent to all
        return np.min(data)
    elif method == "mean":
        return np.mean(data)
    elif method == "sum":
        return np.sum(data)
    else:
        raise NotImplementedError()


def stack_last_n_obs(all_obs, n_steps):
    assert len(all_obs) > 0
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class MultiStepWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        n_obs_steps,
        n_action_steps,
        max_episode_steps=None,
        reward_agg_method="max",
    ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method

        self.obs = deque(maxlen=n_obs_steps + 1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=n_obs_steps + 1))

    def seed(self, seed=None):
        """Set the seed for the environment's random number generator."""
        # Try to seed the underlying environment
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        # For Gymnasium environments, we store the seed and use it in reset
        self._seed = seed
        return [seed]

    def reset(self, **kwargs):
        """Resets the environment using kwargs."""
        # Use stored seed if available (Gymnasium style), otherwise use provided seed
        if hasattr(self, "_seed") and self._seed is not None:
            kwargs["seed"] = self._seed
            self._seed = None  # Use seed only once

        result = super().reset(**kwargs)

        # Handle both old gym (returns obs) and new gymnasium (returns obs, info) APIs
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        self.obs = deque([obs], maxlen=self.n_obs_steps + 1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=self.n_obs_steps + 1))

        obs = self._get_obs(self.n_obs_steps)
        return obs, info

    def step(self, action):
        """actions: (n_action_steps,) + action_shape"""
        for act in action:
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            observation, reward, done, info = super().step(act)

            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) and (
                len(self.reward) >= self.max_episode_steps
            ):
                # truncation
                done = True
            self.done.append(done)
            self._add_info(info)

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, "max")
        info = dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, done, info

    def _get_obs(self, n_steps=1):
        """Output (n_steps,) + obs_shape"""
        assert len(self.obs) > 0
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs([obs[key] for obs in self.obs], n_steps)
            return result
        else:
            raise RuntimeError("Unsupported space type")

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)

    def get_rewards(self):
        return self.reward

    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)

    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result


# ------------------ VideoWrapper ------------------------


class VideoWrapper(gym.Wrapper):
    def __init__(
        self, env, mode="rgb_array", enabled=True, steps_per_render=1, **kwargs
    ):
        super().__init__(env)

        self.mode = mode
        self.enabled = enabled
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render

        self.frames = list()
        self.step_count = 0

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()
        self.step_count = 1
        if self.enabled:
            frame = self.env.render(mode=self.mode, **self.render_kwargs)
            assert frame.dtype == np.uint8
            self.frames.append(frame)
        return obs

    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        if self.enabled and ((self.step_count % self.steps_per_render) == 0):
            frame = self.env.render(mode=self.mode, **self.render_kwargs)
            assert frame.dtype == np.uint8
            self.frames.append(frame)
        return result

    def render(self, mode="rgb_array", **kwargs):
        return self.frames


# ------------------ VideoRecordingWrapper ------------------------


class VideoRecordingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        video_recoder: VideoRecorder,
        mode="rgb_array",
        file_path=None,
        steps_per_render=1,
        **kwargs,
    ):
        """When file_path is None, don't record."""
        super().__init__(env)

        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.file_path = file_path
        self.video_recoder = video_recoder

        self.step_count = 0

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        # Handle both old gym (returns obs) and new gymnasium (returns obs, info) APIs
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        self.frames = list()
        self.step_count = 1
        self.video_recoder.stop()
        return obs, info

    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        if self.file_path is not None and (
            (self.step_count % self.steps_per_render) == 0
        ):
            if not self.video_recoder.is_ready():
                self.video_recoder.start(self.file_path)

            frame = self.env.render(mode=self.mode, **self.render_kwargs)
            assert frame.dtype == np.uint8
            self.video_recoder.write_frame(frame)
        return result

    def render(self, mode="rgb_array", **kwargs):
        if self.video_recoder.is_ready():
            self.video_recoder.stop()
        return self.file_path


def get_accumulate_timestamp_idxs(
    timestamps: list[float],
    start_time: float,
    dt: float,
    eps: float = 1e-5,
    next_global_idx: int | None = 0,
    allow_negative=False,
) -> tuple[list[int], list[int], int]:
    """For each dt window, choose the first timestamp in the window.
    Assumes timestamps sorted. One timestamp might be chosen multiple times due to dropped frames.
    next_global_idx should start at 0 normally, and then use the returned next_global_idx.
    However, when overwiting previous values are desired, set last_global_idx to None.

    Returns:
    local_idxs: which index in the given timestamps array to chose from
    global_idxs: the global index of each chosen timestamp
    next_global_idx: used for next call.
    """
    local_idxs = list()
    global_idxs = list()
    for local_idx, ts in enumerate(timestamps):
        # add eps * dt to timestamps so that when ts == start_time + k * dt
        # is always recorded as kth element (avoiding floating point errors)
        global_idx = math.floor((ts - start_time) / dt + eps)
        if (not allow_negative) and (global_idx < 0):
            continue
        if next_global_idx is None:
            next_global_idx = global_idx

        n_repeats = max(0, global_idx - next_global_idx + 1)
        for i in range(n_repeats):
            local_idxs.append(local_idx)
            global_idxs.append(next_global_idx + i)
        next_global_idx += n_repeats
    return local_idxs, global_idxs, next_global_idx


class VideoRecorder:
    def __init__(
        self,
        fps,
        codec,
        input_pix_fmt,
        # options for codec
        **kwargs,
    ):
        """input_pix_fmt: rgb24, bgr24 see https://github.com/PyAV-Org/PyAV/blob/bc4eedd5fc474e0f25b22102b2771fe5a42bb1c7/av/video/frame.pyx#L352"""
        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.kwargs = kwargs
        # runtime set
        self._reset_state()

    def _reset_state(self):
        self.container = None
        self.stream = None
        self.shape = None
        self.dtype = None
        self.start_time = None
        self.next_global_idx = 0

    @classmethod
    def create_h264(
        cls,
        fps,
        codec="h264",
        input_pix_fmt="rgb24",
        output_pix_fmt="yuv420p",
        crf=18,
        profile="high",
        **kwargs,
    ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={
                "crf": str(crf),
                # 'profile': profile
            },
            **kwargs,
        )
        return obj

    def __del__(self):
        self.stop()

    def is_ready(self):
        return self.stream is not None

    def start(self, file_path, start_time=None):
        if self.is_ready():
            # if still recording, stop first and start anew.
            self.stop()

        self.container = av.open(file_path, mode="w")
        self.stream = self.container.add_stream(self.codec, rate=self.fps)
        codec_context = self.stream.codec_context
        for k, v in self.kwargs.items():
            setattr(codec_context, k, v)
        self.start_time = start_time

    def write_frame(self, img: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError("Must run start() before writing!")

        n_repeats = 1
        if self.start_time is not None:
            local_idxs, global_idxs, self.next_global_idx = (
                get_accumulate_timestamp_idxs(
                    # only one timestamp
                    timestamps=[frame_time],
                    start_time=self.start_time,
                    dt=1 / self.fps,
                    next_global_idx=self.next_global_idx,
                )
            )
            # number of appearance means repeats
            n_repeats = len(local_idxs)

        if self.shape is None:
            self.shape = img.shape
            self.dtype = img.dtype
            h, w, c = img.shape
            self.stream.width = w
            self.stream.height = h
        assert img.shape == self.shape
        assert img.dtype == self.dtype

        frame = av.VideoFrame.from_ndarray(img, format=self.input_pix_fmt)
        for i in range(n_repeats):
            for packet in self.stream.encode(frame):
                self.container.mux(packet)

    def stop(self):
        if not self.is_ready():
            return

        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)

        # Close the file
        self.container.close()

        # reset runtime parameters
        self._reset_state()
