"""Test Kitchen environment by rollout trajectory in the dataset.

Author: Chaoyi Pan
Date: 2025-10-25
"""

import pathlib

import numpy as np

from mip.config import TaskConfig
from mip.envs.kitchen import make_vec_env


def main():
    # create env
    task_config = TaskConfig()
    task_config.env_name = "kitchen-all-v0"
    task_config.save_video = True
    task_config.act_steps = 1
    envs = make_vec_env(task_config)

    # get data
    data_directory = pathlib.Path(
        "/home/pcy/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/snapshots/71de6c7b6d83e4edad3cfbd10bfe9f8942f9d792/kitchen/kitchen"
    )
    actions = np.load(data_directory / "actions_seq.npy")
    masks = np.load(data_directory / "existence_mask.npy")
    mask = masks[0]
    action_list = actions[0, : int(mask.sum())]

    _ = envs.reset()
    for action in action_list:
        obs, reward, terminated, truncated, info = envs.step([action])
        if terminated or truncated:
            break
    envs.close()


if __name__ == "__main__":
    main()
