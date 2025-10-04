"""This file contains the functions to create the environment."""

import collections
import io
import os
import sys

import gymnasium as gym

from mip.config import TaskConfig
from mip.env_utils import MultiStepWrapper, VideoRecorder, VideoRecordingWrapper


def make_env(task_config: TaskConfig, idx, render=False, seed=None):
    if task_config.env_name in ["can", "lift", "square", "tool_hang", "transport"]:
        return make_robomimic_env(task_config, idx, render, seed=seed)
    else:
        raise ValueError(f"Environment {task_config.env_name} not supported")


def make_vec_env(task_config: TaskConfig, seed=None):
    # Suppress output by redirecting stdout temporarily
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()  # Redirect stdout to a string buffer
    if task_config.num_envs == 1 or task_config.save_video:
        vnc_env_class = gym.vector.SyncVectorEnv
    else:
        vnc_env_class = gym.vector.AsyncVectorEnv
    if task_config.env_name in ["can", "lift", "square", "tool_hang", "transport"]:
        try:
            envs = vnc_env_class(
                [
                    make_robomimic_env(task_config, idx, False, seed=seed)
                    for idx in range(task_config.num_envs)
                ],
            )
        finally:
            sys.stdout = original_stdout  # Restore stdout
        return envs
    else:
        raise ValueError(f"Environment {task_config.env_name} not supported")


def make_robomimic_env(task_config: TaskConfig, idx, render=False, seed=None):
    from mip.envs.robomimic_image_wrapper import (
        RobomimicImageWrapper,
    )
    from mip.envs.robomimic_lowdim_wrapper import (
        RobomimicLowdimWrapper,
    )

    def thunk():
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.obs_utils as ObsUtils

        def create_robomimic_env(
            env_meta, obs_keys=None, shape_meta=None, enable_render=True
        ):
            if task_config.obs_type == "state":
                ObsUtils.initialize_obs_modality_mapping_from_dict(
                    {"low_dim": obs_keys}
                )
            else:  # image observation
                modality_mapping = collections.defaultdict(list)
                for key, attr in shape_meta["obs"].items():
                    modality_mapping[attr.get("type", "low_dim")].append(key)
                ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                render=False,
                render_offscreen=enable_render
                if task_config.obs_type == "image"
                else False,
                use_image_obs=enable_render
                if task_config.obs_type == "image"
                else False,
            )
            return env

        # Get dataset path (either from explicit path or HuggingFace download)
        if hasattr(task_config, 'dataset_repo') and hasattr(task_config, 'dataset_filename'):
            from huggingface_hub import hf_hub_download
            dataset_path = hf_hub_download(
                repo_id=task_config.dataset_repo,
                filename=task_config.dataset_filename,
                repo_type="dataset"
            )
        elif hasattr(task_config, 'dataset_path'):
            dataset_path = os.path.expanduser(task_config.dataset_path)
        else:
            raise ValueError("Either dataset_repo/dataset_filename or dataset_path must be provided")

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        if task_config.obs_type == "image":
            # disable object state observation for image mode
            env_meta["env_kwargs"]["use_object_obs"] = False
        abs_action = task_config.abs_action
        if abs_action:
            env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False

        if task_config.obs_type == "state":
            env = create_robomimic_env(env_meta=env_meta, obs_keys=task_config.obs_keys)
            env = RobomimicLowdimWrapper(
                env=env,
                obs_keys=task_config.obs_keys,
                init_state=None,
                render_hw=(256, 256),
                render_camera_name="agentview",
            )
        else:  # image observation
            env = create_robomimic_env(
                env_meta=env_meta, shape_meta=task_config.shape_meta
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            env.env.hard_reset = False
            env = RobomimicImageWrapper(
                env=env,
                shape_meta=task_config.shape_meta,
                init_state=None,
                render_obs_key=task_config.render_obs_key,
            )

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
        env = MultiStepWrapper(
            env,
            n_obs_steps=task_config.obs_steps,
            n_action_steps=task_config.act_steps,
            max_episode_steps=task_config.max_episode_steps,
        )
        if seed is not None:
            env.seed(seed + idx)
            print("Env seed: ", seed + idx)
        return env

    return thunk
