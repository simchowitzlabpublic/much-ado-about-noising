"""Simple tests for env_utils.py.

Author: Chaoyi Pan
Date: 2025-10-03
"""

from unittest.mock import Mock

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mip.env_utils import (
    MultiStepWrapper,
    VideoRecorder,
    VideoRecordingWrapper,
    VideoWrapper,
    aggregate,
    dict_take_last_n,
    get_accumulate_timestamp_idxs,
    repeated_box,
    repeated_space,
    stack_last_n_obs,
    stack_repeated,
    take_last_n,
)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_stack_repeated(self):
        """Test stack_repeated function."""
        x = np.array([1, 2, 3])
        result = stack_repeated(x, 3)
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result[0], x)
        np.testing.assert_array_equal(result[1], x)
        np.testing.assert_array_equal(result[2], x)

    def test_repeated_box(self):
        """Test repeated_box function."""
        box = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        result = repeated_box(box, 3)
        assert result.shape == (3, 2)
        assert result.dtype == np.float32

    def test_repeated_space_box(self):
        """Test repeated_space with Box space."""
        box = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        result = repeated_space(box, 2)
        assert isinstance(result, spaces.Box)
        assert result.shape == (2, 2)

    def test_repeated_space_dict(self):
        """Test repeated_space with Dict space."""
        dict_space = spaces.Dict(
            {"obs": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)}
        )
        result = repeated_space(dict_space, 2)
        assert isinstance(result, spaces.Dict)
        assert result["obs"].shape == (2, 2)

    def test_take_last_n(self):
        """Test take_last_n function."""
        x = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        result = take_last_n(x, 2)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result[0], np.array([3, 4]))
        np.testing.assert_array_equal(result[1], np.array([5, 6]))

    def test_take_last_n_more_than_available(self):
        """Test take_last_n with n larger than list length."""
        x = [np.array([1, 2])]
        result = take_last_n(x, 3)
        assert result.shape == (1, 2)

    def test_dict_take_last_n(self):
        """Test dict_take_last_n function."""
        x = {
            "a": [np.array([1]), np.array([2]), np.array([3])],
            "b": [np.array([4]), np.array([5]), np.array([6])],
        }
        result = dict_take_last_n(x, 2)
        assert len(result) == 2
        np.testing.assert_array_equal(result["a"], np.array([[2], [3]]))
        np.testing.assert_array_equal(result["b"], np.array([[5], [6]]))

    def test_aggregate_max(self):
        """Test aggregate with max method."""
        data = np.array([1, 2, 3, 4])
        result = aggregate(data, method="max")
        assert result == 4

    def test_aggregate_min(self):
        """Test aggregate with min method."""
        data = np.array([1, 2, 3, 4])
        result = aggregate(data, method="min")
        assert result == 1

    def test_aggregate_mean(self):
        """Test aggregate with mean method."""
        data = np.array([1, 2, 3, 4])
        result = aggregate(data, method="mean")
        assert result == 2.5

    def test_aggregate_sum(self):
        """Test aggregate with sum method."""
        data = np.array([1, 2, 3, 4])
        result = aggregate(data, method="sum")
        assert result == 10

    def test_stack_last_n_obs(self):
        """Test stack_last_n_obs function."""
        obs_list = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        result = stack_last_n_obs(obs_list, 2)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result[0], np.array([3, 4]))
        np.testing.assert_array_equal(result[1], np.array([5, 6]))

    def test_stack_last_n_obs_with_padding(self):
        """Test stack_last_n_obs with padding."""
        obs_list = [np.array([1, 2])]
        result = stack_last_n_obs(obs_list, 3)
        assert result.shape == (3, 2)
        # Should pad with first observation
        np.testing.assert_array_equal(result[0], np.array([1, 2]))
        np.testing.assert_array_equal(result[1], np.array([1, 2]))
        np.testing.assert_array_equal(result[2], np.array([1, 2]))

    def test_get_accumulate_timestamp_idxs(self):
        """Test get_accumulate_timestamp_idxs function."""
        timestamps = [0.1, 0.2, 0.3]
        local_idxs, global_idxs, next_idx = get_accumulate_timestamp_idxs(
            timestamps, start_time=0.0, dt=0.1, next_global_idx=0
        )
        assert len(local_idxs) == len(global_idxs)
        assert next_idx > 0


class TestMultiStepWrapper:
    """Test MultiStepWrapper class."""

    def test_initialization(self):
        """Test MultiStepWrapper initialization."""
        # Create a simple mock environment
        mock_env = Mock(spec=gym.Env)
        mock_env.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        mock_env.observation_space = spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )

        wrapper = MultiStepWrapper(mock_env, n_obs_steps=2, n_action_steps=3)

        assert wrapper.n_obs_steps == 2
        assert wrapper.n_action_steps == 3
        assert wrapper._action_space.shape == (3, 2)
        assert wrapper._observation_space.shape == (2, 4)

    def test_reset(self):
        """Test MultiStepWrapper reset."""
        mock_env = Mock(spec=gym.Env)
        mock_env.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        mock_env.observation_space = spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )
        mock_env.reset.return_value = np.array([1, 2, 3, 4], dtype=np.float32)

        wrapper = MultiStepWrapper(mock_env, n_obs_steps=2, n_action_steps=3)
        obs = wrapper.reset()

        assert obs.shape == (2, 4)
        mock_env.reset.assert_called_once()


class TestVideoWrapper:
    """Test VideoWrapper class."""

    def test_initialization(self):
        """Test VideoWrapper initialization."""
        mock_env = Mock(spec=gym.Env)
        wrapper = VideoWrapper(mock_env, mode="rgb_array", enabled=True)

        assert wrapper.mode == "rgb_array"
        assert wrapper.enabled is True
        assert len(wrapper.frames) == 0

    def test_initialization_disabled(self):
        """Test VideoWrapper initialization with disabled recording."""
        mock_env = Mock(spec=gym.Env)
        wrapper = VideoWrapper(mock_env, mode="rgb_array", enabled=False)

        assert wrapper.enabled is False


class TestVideoRecorder:
    """Test VideoRecorder class."""

    def test_initialization(self):
        """Test VideoRecorder initialization."""
        recorder = VideoRecorder(fps=30, codec="h264", input_pix_fmt="rgb24")

        assert recorder.fps == 30
        assert recorder.codec == "h264"
        assert recorder.input_pix_fmt == "rgb24"
        assert not recorder.is_ready()

    def test_create_h264(self):
        """Test VideoRecorder.create_h264 class method."""
        recorder = VideoRecorder.create_h264(fps=30)

        assert recorder.fps == 30
        assert recorder.codec == "h264"
        assert recorder.input_pix_fmt == "rgb24"
        assert not recorder.is_ready()

    def test_is_ready_before_start(self):
        """Test is_ready returns False before start."""
        recorder = VideoRecorder.create_h264(fps=30)
        assert not recorder.is_ready()


class TestVideoRecordingWrapper:
    """Test VideoRecordingWrapper class."""

    def test_initialization(self):
        """Test VideoRecordingWrapper initialization."""
        mock_env = Mock(spec=gym.Env)
        mock_recorder = Mock(spec=VideoRecorder)

        wrapper = VideoRecordingWrapper(
            mock_env,
            mock_recorder,
            mode="rgb_array",
            file_path=None,
        )

        assert wrapper.mode == "rgb_array"
        assert wrapper.file_path is None
        assert wrapper.video_recoder is mock_recorder

    def test_initialization_with_file_path(self):
        """Test VideoRecordingWrapper initialization with file path."""
        mock_env = Mock(spec=gym.Env)
        mock_recorder = Mock(spec=VideoRecorder)

        wrapper = VideoRecordingWrapper(
            mock_env,
            mock_recorder,
            mode="rgb_array",
            file_path="/tmp/test.mp4",
        )

        assert wrapper.file_path == "/tmp/test.mp4"


if __name__ == "__main__":
    import pytest

    # Run tests
    pytest.main([__file__, "-v"])
