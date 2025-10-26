"""Simple tests for env_utils.py.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import numpy as np
from gymnasium import spaces

from mip.env_utils import (
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
