"""Tests for action_utils: roundtrip abs->rel->abs conversion."""

import numpy as np
from scipy.spatial.transform import Rotation

from mip.action_utils import (
    action_abs_to_rel,
    action_rel_to_abs,
    get_eef_state_from_obs,
    rot6d_to_rotmat,
    rotmat_to_rot6d,
)


class TestRot6dConversion:
    def test_roundtrip_identity(self):
        """Identity matrix should roundtrip through rot6d."""
        R = np.eye(3)
        r6d = rotmat_to_rot6d(R)
        R_back = rot6d_to_rotmat(r6d)
        np.testing.assert_allclose(R, R_back, atol=1e-7)

    def test_roundtrip_random(self):
        """Random rotation matrices should roundtrip through rot6d."""
        for _ in range(10):
            R = Rotation.random().as_matrix()
            r6d = rotmat_to_rot6d(R)
            R_back = rot6d_to_rotmat(r6d)
            np.testing.assert_allclose(R, R_back, atol=1e-6)

    def test_batched(self):
        """Batch of rotations should work."""
        R_batch = Rotation.random(20).as_matrix()  # (20, 3, 3)
        r6d = rotmat_to_rot6d(R_batch)  # (20, 6)
        assert r6d.shape == (20, 6)
        R_back = rot6d_to_rotmat(r6d)
        np.testing.assert_allclose(R_batch, R_back, atol=1e-6)


class TestAbsRelRoundtrip:
    def _make_random_eef(self):
        """Create a random EEF pose."""
        eef_pos = np.random.randn(3)
        eef_rotmat = Rotation.random().as_matrix()
        return eef_pos, eef_rotmat

    def _make_random_actions(self, T=8):
        """Create random absolute 10D actions."""
        actions = np.zeros((T, 10), dtype=np.float64)
        for t in range(T):
            actions[t, :3] = np.random.randn(3) * 0.1  # pos
            R = Rotation.random().as_matrix()
            actions[t, 3:9] = rotmat_to_rot6d(R)  # rot6d
            actions[t, 9] = np.random.uniform(-1, 1)  # grip
        return actions.astype(np.float32)

    def test_roundtrip_single(self):
        """Single action roundtrip: abs->rel->abs."""
        eef_pos, eef_rotmat = self._make_random_eef()
        action_abs = self._make_random_actions(1)[0]  # (10,)

        action_rel = action_abs_to_rel(eef_pos, eef_rotmat, action_abs)
        action_abs_recovered = action_rel_to_abs(eef_pos, eef_rotmat, action_rel)

        np.testing.assert_allclose(
            action_abs,
            action_abs_recovered,
            atol=1e-5,
            err_msg="Roundtrip abs->rel->abs failed for single action",
        )

    def test_roundtrip_chunk(self):
        """Action chunk roundtrip: abs->rel->abs."""
        eef_pos, eef_rotmat = self._make_random_eef()
        action_abs = self._make_random_actions(8)  # (8, 10)

        action_rel = action_abs_to_rel(eef_pos, eef_rotmat, action_abs)
        action_abs_recovered = action_rel_to_abs(eef_pos, eef_rotmat, action_rel)

        np.testing.assert_allclose(
            action_abs,
            action_abs_recovered,
            atol=1e-5,
            err_msg="Roundtrip abs->rel->abs failed for action chunk",
        )

    def test_gripper_unchanged(self):
        """Gripper values should be unchanged by conversion."""
        eef_pos, eef_rotmat = self._make_random_eef()
        action_abs = self._make_random_actions(8)

        action_rel = action_abs_to_rel(eef_pos, eef_rotmat, action_abs)

        np.testing.assert_allclose(
            action_abs[:, 9],
            action_rel[:, 9],
            atol=1e-7,
            err_msg="Gripper values changed during abs->rel",
        )

    def test_identity_reference(self):
        """When EEF is at origin with identity rotation, rel == abs."""
        eef_pos = np.zeros(3)
        eef_rotmat = np.eye(3)
        action_abs = self._make_random_actions(4)

        action_rel = action_abs_to_rel(eef_pos, eef_rotmat, action_abs)

        np.testing.assert_allclose(
            action_abs,
            action_rel,
            atol=1e-5,
            err_msg="With identity reference, rel should equal abs",
        )

    def test_rel_rotation_near_identity_for_close_actions(self):
        """When target rotation is close to current, relative rot6d should be near identity columns."""
        eef_pos = np.array([0.5, 0.3, 0.2])
        eef_rotmat = Rotation.random().as_matrix()

        # Actions very close to current EEF pose
        T = 4
        action_abs = np.zeros((T, 10), dtype=np.float64)
        for t in range(T):
            action_abs[t, :3] = eef_pos + np.random.randn(3) * 0.001
            # Small perturbation from current rotation
            small_angle = Rotation.from_rotvec(np.random.randn(3) * 0.01)
            R_target = (small_angle * Rotation.from_matrix(eef_rotmat)).as_matrix()
            action_abs[t, 3:9] = rotmat_to_rot6d(R_target)
            action_abs[t, 9] = 1.0

        action_rel = action_abs_to_rel(
            eef_pos, eef_rotmat, action_abs.astype(np.float32)
        )

        # Relative positions should be near zero
        assert np.all(np.abs(action_rel[:, :3]) < 0.01), "Relative pos should be near 0"

        # Relative rot6d should be near identity columns [1,0,0, 0,1,0]
        identity_rot6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        for t in range(T):
            np.testing.assert_allclose(
                action_rel[t, 3:9],
                identity_rot6d,
                atol=0.03,
                err_msg=f"Relative rot6d at step {t} should be near identity",
            )


class TestGetEefStateFromObs:
    def test_extraction(self):
        """Test EEF state extraction from concatenated obs."""
        obs_keys = [
            "object",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
        obs_key_dims = {
            "object": 14,
            "robot0_eef_pos": 3,
            "robot0_eef_quat": 4,
            "robot0_gripper_qpos": 2,
        }

        # Build a fake obs vector
        obj = np.random.randn(14).astype(np.float32)
        pos = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        quat = Rotation.random().as_quat().astype(np.float32)  # xyzw
        grip = np.array([0.1, 0.2], dtype=np.float32)
        obs_raw = np.concatenate([obj, pos, quat, grip])

        eef_pos, eef_rotmat = get_eef_state_from_obs(obs_raw, obs_keys, obs_key_dims)

        np.testing.assert_allclose(eef_pos, pos, atol=1e-6)
        expected_rotmat = Rotation.from_quat(quat).as_matrix()
        np.testing.assert_allclose(eef_rotmat, expected_rotmat, atol=1e-5)

    def test_batched_extraction(self):
        """Test batched EEF state extraction."""
        obs_keys = [
            "object",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
        obs_key_dims = {
            "object": 14,
            "robot0_eef_pos": 3,
            "robot0_eef_quat": 4,
            "robot0_gripper_qpos": 2,
        }

        batch_size = 5
        obs_raw = np.random.randn(batch_size, 23).astype(np.float32)
        # Set valid quaternions
        for i in range(batch_size):
            quat = Rotation.random().as_quat().astype(np.float32)
            obs_raw[i, 17:21] = quat  # eef_quat at offset 14+3=17

        eef_pos, eef_rotmat = get_eef_state_from_obs(obs_raw, obs_keys, obs_key_dims)

        assert eef_pos.shape == (batch_size, 3)
        assert eef_rotmat.shape == (batch_size, 3, 3)
