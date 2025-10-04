"""Simple tests for robomimic_env.py

Author: Chaoyi Pan
Date: 2025-10-03
"""

from unittest.mock import Mock, patch

from mip.config import TaskConfig
from mip.envs.robomimic_env import make_env, make_robomimic_env, make_vec_env


class TestMakeRobomimicEnv:
    """Test make_robomimic_env function."""

    def test_make_robomimic_env_returns_callable(self):
        """Test that make_robomimic_env returns a callable thunk."""
        task_config = TaskConfig(
            env_name="lift",
            obs_type="state",
            dataset_path="~/data/lift/ph/low_dim.hdf5",
            obs_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
            seed=42,
        )

        thunk = make_robomimic_env(task_config, idx=0, render=False)

        # Check that it returns a callable
        assert callable(thunk)

    def test_make_robomimic_env_different_indices(self):
        """Test that make_robomimic_env works with different indices."""
        task_config = TaskConfig(
            env_name="lift",
            obs_type="state",
            dataset_path="~/data/lift/ph/low_dim.hdf5",
            seed=42,
        )

        thunk1 = make_robomimic_env(task_config, idx=0, render=False)
        thunk2 = make_robomimic_env(task_config, idx=1, render=False)

        # Check that both return callables
        assert callable(thunk1)
        assert callable(thunk2)

    def test_make_robomimic_env_with_render(self):
        """Test that make_robomimic_env works with render=True."""
        task_config = TaskConfig(
            env_name="lift",
            obs_type="state",
            dataset_path="~/data/lift/ph/low_dim.hdf5",
            seed=42,
        )

        thunk = make_robomimic_env(task_config, idx=0, render=True)

        # Check that it returns a callable
        assert callable(thunk)


class TestMakeEnv:
    """Test make_env function."""

    def test_make_env_supported_environments(self):
        """Test make_env with supported environment names."""
        supported_envs = ["can", "lift", "square", "tool_hang", "transport"]

        for env_name in supported_envs:
            task_config = TaskConfig(
                env_name=env_name,
                obs_type="state",
                dataset_path=f"~/data/{env_name}/ph/low_dim.hdf5",
                seed=42,
            )

            thunk = make_env(task_config, idx=0, render=False)

            # Check that it returns a callable
            assert callable(thunk)

    def test_make_env_unsupported_environment(self):
        """Test make_env raises error for unsupported environments."""
        task_config = TaskConfig(
            env_name="unsupported_env",
            obs_type="state",
            dataset_path="~/data/unsupported/ph/low_dim.hdf5",
            seed=42,
        )

        try:
            make_env(task_config, idx=0, render=False)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not supported" in str(e)


class TestMakeVecEnv:
    """Test make_vec_env function."""

    @patch("mip.envs.robomimic_env.gym.vector.SyncVectorEnv")
    @patch("mip.envs.robomimic_env.make_robomimic_env")
    def test_make_vec_env_single_env(self, mock_make_robomimic_env, mock_sync_vec_env):
        """Test make_vec_env with single environment."""
        task_config = TaskConfig(
            env_name="lift",
            obs_type="state",
            dataset_path="~/data/lift/ph/low_dim.hdf5",
            num_envs=1,
            seed=42,
        )

        # Mock the thunk
        mock_thunk = Mock()
        mock_make_robomimic_env.return_value = mock_thunk

        # Mock SyncVectorEnv
        mock_vec_env_instance = Mock()
        mock_sync_vec_env.return_value = mock_vec_env_instance

        result = make_vec_env(task_config)

        # Check that SyncVectorEnv was used (not AsyncVectorEnv)
        mock_sync_vec_env.assert_called_once()
        assert result == mock_vec_env_instance

    @patch("mip.envs.robomimic_env.gym.vector.SyncVectorEnv")
    @patch("mip.envs.robomimic_env.make_robomimic_env")
    def test_make_vec_env_with_video_saving(
        self, mock_make_robomimic_env, mock_sync_vec_env
    ):
        """Test make_vec_env with video saving uses SyncVectorEnv."""
        task_config = TaskConfig(
            env_name="lift",
            obs_type="state",
            dataset_path="~/data/lift/ph/low_dim.hdf5",
            num_envs=4,
            save_video=True,
            seed=42,
        )

        # Mock the thunk
        mock_thunk = Mock()
        mock_make_robomimic_env.return_value = mock_thunk

        # Mock SyncVectorEnv
        mock_vec_env_instance = Mock()
        mock_sync_vec_env.return_value = mock_vec_env_instance

        result = make_vec_env(task_config)

        # Check that SyncVectorEnv was used even with multiple envs
        mock_sync_vec_env.assert_called_once()
        assert result == mock_vec_env_instance

    @patch("mip.envs.robomimic_env.gym.vector.AsyncVectorEnv")
    @patch("mip.envs.robomimic_env.make_robomimic_env")
    def test_make_vec_env_multiple_envs(
        self, mock_make_robomimic_env, mock_async_vec_env
    ):
        """Test make_vec_env with multiple environments uses AsyncVectorEnv."""
        task_config = TaskConfig(
            env_name="lift",
            obs_type="state",
            dataset_path="~/data/lift/ph/low_dim.hdf5",
            num_envs=4,
            save_video=False,
            seed=42,
        )

        # Mock the thunk
        mock_thunk = Mock()
        mock_make_robomimic_env.return_value = mock_thunk

        # Mock AsyncVectorEnv
        mock_vec_env_instance = Mock()
        mock_async_vec_env.return_value = mock_vec_env_instance

        result = make_vec_env(task_config)

        # Check that AsyncVectorEnv was used
        mock_async_vec_env.assert_called_once()
        assert result == mock_vec_env_instance

    def test_make_vec_env_unsupported_environment(self):
        """Test make_vec_env raises error for unsupported environments."""
        task_config = TaskConfig(
            env_name="unsupported_env",
            obs_type="state",
            dataset_path="~/data/unsupported/ph/low_dim.hdf5",
            num_envs=1,
            seed=42,
        )

        try:
            make_vec_env(task_config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not supported" in str(e)


if __name__ == "__main__":
    import pytest

    # Run tests
    pytest.main([__file__, "-v"])
