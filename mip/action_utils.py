"""Action-space conversion utilities for UMI-style relative actions.

Ported from behavior-ik/flyinghand_robocasa/math.py with simplified signatures.

Action layout per arm: pos(3) + rot6d(6) + grip(1) = 10D.
Dual-arm actions are 20D (two stacked 10D arms).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation  # noqa: TC002

# ---------------------------------------------------------------------------
# Rotation helpers (NumPy)
# ---------------------------------------------------------------------------


def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """Recover 3x3 rotation matrix from 6D representation via Gram-Schmidt.

    Args:
        rot6d: (..., 6) array -- first two columns of rotation matrix.

    Returns:
        (..., 3, 3) proper rotation matrix.
    """
    x_raw = rot6d[..., 0:3]
    y_raw = rot6d[..., 3:6]
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


def rotmat_to_rot6d(rotmat: np.ndarray) -> np.ndarray:
    """Extract 6D rotation representation (first two columns) from 3x3 matrix.

    Args:
        rotmat: (..., 3, 3) rotation matrix.

    Returns:
        (..., 6) array -- first two columns flattened.
    """
    return np.concatenate([rotmat[..., :, 0], rotmat[..., :, 1]], axis=-1)


# ---------------------------------------------------------------------------
# Abs <-> Rel conversion (single-arm 10D actions)
# ---------------------------------------------------------------------------


def action_abs_to_rel(
    eef_pos: np.ndarray,
    eef_rotmat: np.ndarray,
    action_abs: np.ndarray,
) -> np.ndarray:
    """Convert absolute actions to relative (delta in current body frame).

    Uses the given EEF pose as the reference for ALL steps in the action chunk
    (UMI-style: single reference per chunk).

    Args:
        eef_pos: (3,) current end-effector position (world frame).
        eef_rotmat: (3, 3) current end-effector rotation matrix (world frame).
        action_abs: (T, 10) or (10,) absolute actions [pos(3) + rot6d(6) + grip(1)].
            For dual-arm (T, 20), reshape to (T, 2, 10) before calling per-arm.

    Returns:
        action_rel: same shape as input. Position and rotation expressed as
            deltas in the body frame. Gripper values copied unchanged.
    """
    action_abs = np.asarray(action_abs, dtype=np.float64)
    squeezed = action_abs.ndim == 1
    if squeezed:
        action_abs = action_abs[np.newaxis]

    out = action_abs.copy()
    eef_pos = np.asarray(eef_pos, dtype=np.float64).reshape(3)
    eef_R = np.asarray(eef_rotmat, dtype=np.float64).reshape(3, 3)

    # delta_pos[t] = R_cur^T @ (target_pos[t] - cur_pos)
    target_pos = action_abs[:, :3]  # (T, 3)
    delta_pos = np.einsum("ji,tj->ti", eef_R, target_pos - eef_pos[None, :])

    # R_rel[t] = R_cur^T @ R_target[t]
    target_R = rot6d_to_rotmat(action_abs[:, 3:9])  # (T, 3, 3)
    R_rel = np.einsum("ji,tjk->tik", eef_R, target_R)
    delta_6d = rotmat_to_rot6d(R_rel)  # (T, 6)

    out[:, :3] = delta_pos
    out[:, 3:9] = delta_6d
    # grip (index 9) copied unchanged

    result = out.astype(np.float32)
    return result[0] if squeezed else result


def action_rel_to_abs(
    eef_pos: np.ndarray,
    eef_rotmat: np.ndarray,
    action_rel: np.ndarray,
) -> np.ndarray:
    """Convert relative actions back to absolute (world frame).

    Inverse of action_abs_to_rel.

    Args:
        eef_pos: (3,) current end-effector position (world frame).
        eef_rotmat: (3, 3) current end-effector rotation matrix (world frame).
        action_rel: (T, 10) or (10,) relative actions [pos(3) + rot6d(6) + grip(1)].

    Returns:
        action_abs: same shape as input. Absolute actions in world frame.
    """
    action_rel = np.asarray(action_rel, dtype=np.float64)
    squeezed = action_rel.ndim == 1
    if squeezed:
        action_rel = action_rel[np.newaxis]

    out = action_rel.copy()
    eef_pos = np.asarray(eef_pos, dtype=np.float64).reshape(3)
    eef_R = np.asarray(eef_rotmat, dtype=np.float64).reshape(3, 3)

    # abs_pos[t] = cur_pos + R_cur @ delta_pos[t]
    delta_pos = action_rel[:, :3]  # (T, 3)
    abs_pos = eef_pos[None, :] + np.einsum("ij,tj->ti", eef_R, delta_pos)

    # R_abs[t] = R_cur @ R_rel[t]
    R_rel = rot6d_to_rotmat(action_rel[:, 3:9])  # (T, 3, 3)
    R_abs = np.einsum("ij,tjk->tik", eef_R, R_rel)
    abs_6d = rotmat_to_rot6d(R_abs)  # (T, 6)

    out[:, :3] = abs_pos
    out[:, 3:9] = abs_6d
    # grip (index 9) copied unchanged

    result = out.astype(np.float32)
    return result[0] if squeezed else result


# ---------------------------------------------------------------------------
# EEF state extraction from concatenated observation vector
# ---------------------------------------------------------------------------


def get_eef_state_from_obs(
    obs_raw: np.ndarray,
    obs_keys: list[str],
    obs_key_dims: dict[str, int],
    robot_prefix: str = "robot0",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract EEF position and rotation matrix from a raw observation vector.

    Args:
        obs_raw: (..., obs_dim) raw (unnormalized) observation.
        obs_keys: ordered list of observation keys used to build the obs vector.
        obs_key_dims: mapping from key name to its dimension.
        robot_prefix: which robot's EEF to extract ("robot0" or "robot1").

    Returns:
        eef_pos: (..., 3) end-effector position.
        eef_rotmat: (..., 3, 3) end-effector rotation matrix.
    """
    pos_key = f"{robot_prefix}_eef_pos"
    quat_key = f"{robot_prefix}_eef_quat"

    # Compute cumulative offsets
    offset = 0
    eef_pos_start = None
    eef_quat_start = None
    for key in obs_keys:
        dim = obs_key_dims[key]
        if key == pos_key:
            eef_pos_start = offset
        elif key == quat_key:
            eef_quat_start = offset
        offset += dim

    if eef_pos_start is None or eef_quat_start is None:
        raise ValueError(f"obs_keys must contain '{pos_key}' and '{quat_key}'")

    eef_pos = obs_raw[..., eef_pos_start : eef_pos_start + 3]
    eef_quat = obs_raw[..., eef_quat_start : eef_quat_start + 4]  # xyzw (scipy)

    # Convert quaternion to rotation matrix
    original_shape = eef_quat.shape[:-1]
    eef_quat_flat = eef_quat.reshape(-1, 4)
    rotmat_flat = Rotation.from_quat(eef_quat_flat).as_matrix()
    eef_rotmat = rotmat_flat.reshape(*original_shape, 3, 3)

    return eef_pos, eef_rotmat
