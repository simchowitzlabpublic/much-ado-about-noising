"""Environments using kitchen and Franka robot.

Uses Gymnasium-Robotics implementation of Franka Kitchen environment.
Reference: https://robotics.farama.org/envs/franka_kitchen/
"""

from mip.envs.kitchen.kitchen_env_wrapper import make_vec_env

__all__ = ["make_vec_env"]
