"""Training pipeline for PushT dataset.

Author: Chaoyi Pan
Date: 2025-10-15
"""

import time

import hydra
import loguru
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from mip.agent import TrainingAgent
from mip.config import Config
from mip.dataset_utils import loop_dataloader
from mip.datasets.pusht_dataset import make_dataset
from mip.envs.pusht import make_vec_env
from mip.logger import Logger, compute_average_metrics, update_best_metrics
from mip.samplers import get_default_step_list
from mip.scheduler import WarmupAnnealingScheduler
from mip.torch_utils import set_seed


def train(config: Config, envs, dataset, agent, logger, resume_state=None):
    """Standalone training function.

    Args:
        config: Configuration for training
        envs: Environment
        dataset: Training dataset
        agent: Agent to train
        logger: Logger for metrics
        resume_state: Optional dict with training state to resume from
    """
    # dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        num_workers=4 if config.task.obs_type in ["state", "keypoint"] else 8,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True,
    )
    loop_loader = loop_dataloader(dataloader)

    # lr scheduler
    lr_scheduler = CosineAnnealingLR(
        agent.optimizer, T_max=config.optimization.gradient_steps
    )

    # warmup scheduler (mainly for flow map learning)
    warmup_scheduler = WarmupAnnealingScheduler(
        max_steps=config.optimization.gradient_steps,
        warmup_ratio=config.optimization.warmup_ratio,
        rampup_ratio=config.optimization.rampup_ratio,
        min_value=config.optimization.min_value,
        max_value=config.optimization.max_value,
    )

    # Resume from checkpoint if available
    start_step = 0
    best_metrics = {}
    eval_history = []
    if resume_state is not None:
        start_step = resume_state.get("n_gradient_step", 0) + 1
        best_metrics = resume_state.get("best_metrics", {})
        eval_history = resume_state.get("eval_history", [])
        loguru.logger.info(f"Resuming training from step {start_step}")
        loguru.logger.info(f"Restored best metrics: {best_metrics}")

        # Fast-forward the lr_scheduler to the correct step
        for _ in range(start_step):
            lr_scheduler.step()

    info_list = []
    start_time = time.time()
    for n_gradient_step in range(start_step, config.optimization.gradient_steps):
        # get batch from dataloader
        batch = next(loop_loader)

        # preprocess data
        if config.task.obs_type == "image":
            obs_batch = batch["obs"]
            obs = {}
            for k in obs_batch:
                obs[k] = obs_batch[k][:, : config.task.obs_steps, :].to(
                    config.optimization.device
                )
        elif config.task.obs_type == "state":
            obs = batch["obs"]["state"].to(config.optimization.device)
            obs = obs[:, : config.task.obs_steps, :]  # (B, obs_horizon, obs_dim)
        elif config.task.obs_type == "keypoint":
            obs_batch = batch["obs"]
            obs = {}
            for k in obs_batch:
                obs_data = obs_batch[k].to(config.optimization.device)
                obs[k] = obs_data[
                    :, : config.task.obs_steps, :
                ]  # (B, obs_horizon, obs_dim)
        else:
            raise ValueError(f"Invalid obs_type: {config.task.obs_type}")

        act = batch["action"].to(config.optimization.device)
        act = act[:, : config.task.horizon, :]  # (B, horizon, act_dim)

        # update diffusion
        delta_t_scalar = warmup_scheduler(n_gradient_step)
        batch_size = act.shape[0]
        delta_t = torch.full(
            (batch_size,), delta_t_scalar, device=config.optimization.device
        )
        info = agent.update(act, obs, delta_t)
        lr_scheduler.step()
        info_list.append(info)

        # log metrics
        if (n_gradient_step + 1) % config.log.log_freq == 0:
            metrics = {
                "step": n_gradient_step,
                "total_time": time.time() - start_time,
                "lr": lr_scheduler.get_last_lr()[0],
                "delta_t": delta_t_scalar,
            }
            for key in info:
                try:
                    metrics[key] = np.nanmean([info[key] for info in info_list])
                except (KeyError, TypeError, ValueError):
                    metrics[key] = np.nan
            logger.log(metrics, category="train")
            info_list = []

        if (n_gradient_step + 1) % config.log.save_freq == 0:
            loguru.logger.info("Save model...")
            logger.save_agent(agent=agent, identifier="latest")

        if (n_gradient_step + 1) % config.log.eval_freq == 0:
            loguru.logger.info("Evaluate model...")
            agent.eval()
            metrics = {"step": n_gradient_step}
            num_steps_list = get_default_step_list(config.optimization.loss_type)
            for num_steps in num_steps_list:
                metrics.update(
                    evaluate(config, envs, dataset, agent, logger, num_steps)
                )

            # Update best metrics and average metrics
            old_best_metrics = best_metrics.copy()
            best_metrics = update_best_metrics(best_metrics, metrics)
            eval_history.append(metrics.copy())
            avg_metrics = compute_average_metrics(eval_history)

            # Check if this is a new best model based on success rate
            # Use the first num_steps in the list as the primary metric
            primary_metric_key = f"mean_success_{num_steps_list[0]}"
            if primary_metric_key in metrics:
                is_new_best = (
                    primary_metric_key not in old_best_metrics
                    or metrics[primary_metric_key]
                    > old_best_metrics[primary_metric_key]
                )
                if is_new_best:
                    success_rate = metrics[primary_metric_key]
                    loguru.logger.info(
                        f"New best model! {primary_metric_key} = {success_rate:.4f}"
                    )
                    # Save to local models directory
                    logger.save_agent(agent=agent, identifier="best")

                    # Save to global checkpoints directory with success rate comparison
                    # Include training state for resuming
                    checkpoint_base_name = (
                        f"{config.task.env_name}_{config.task.env_type}_{config.task.obs_type}_"
                        f"{config.optimization.loss_type}_{config.network.network_type}_"
                        f"{config.network.emb_dim}_seed{config.optimization.seed}"
                    )
                    training_state = {
                        "n_gradient_step": n_gradient_step,
                        "best_metrics": best_metrics,
                        "eval_history": eval_history,
                    }
                    logger.save_global_checkpoint(
                        agent,
                        checkpoint_base_name,
                        success_rate,
                        training_state=training_state,
                    )

            # Add best and average metrics to current metrics for logging
            for key, value in best_metrics.items():
                metrics[f"best_{key}"] = value
            for key, value in avg_metrics.items():
                metrics[key] = value

            # Print best and average metrics
            loguru.logger.info("Best metrics so far:")
            for key, value in best_metrics.items():
                loguru.logger.info(f"  {key}: {value:.4f}")
            if avg_metrics:
                loguru.logger.info("Average metrics (last 5 evals):")
                for key, value in avg_metrics.items():
                    loguru.logger.info(f"  {key}: {value:.4f}")

            logger.log(metrics, category="eval")
            agent.train()


def evaluate(config: Config, envs, dataset, agent, logger, num_steps=1):
    """Standalone inference function to evaluate a trained agent and optionally save a video.

    Args:
        config: Configuration object containing evaluation parameters
        envs: Environment
        dataset: Dataset
        agent: Trained agent
        logger: Logger for metrics
        num_steps: Number of steps for sampling

    Returns:
        dict: Metrics including mean step, reward, and success rate
    """
    # ---------------- Start Rollout ----------------
    episode_rewards = []
    episode_steps = []
    episode_success = []

    for i in range(config.log.eval_episodes // config.task.num_envs):
        step_reward = []
        ep_reward = [0.0] * config.task.num_envs
        # NOTE: update env seed, the original envs is update seed so reset is broken
        for j in range(len(envs.envs)):
            envs.envs[j].seed(config.optimization.seed + i * config.task.num_envs + j)
        obs, _ = envs.reset()
        t = 0

        # initialize video stream
        if config.log.save_video:
            logger.video_init(envs.envs[0], enable=True, video_id=str(i))  # save videos

        while t < config.task.max_episode_steps:
            if config.task.obs_type == "state":
                obs = obs.astype(np.float32)  # (num_envs, obs_steps, obs_dim)
                # normalize obs
                obs = dataset.normalizer["obs"]["state"].normalize(obs)
                obs = torch.tensor(
                    obs, device=config.optimization.device, dtype=torch.float32
                )  # (num_envs, obs_steps, obs_dim)
                obs = {"state": obs}
            elif config.task.obs_type == "keypoint":
                obs_raw = obs.astype(np.float32)  # (num_envs, obs_steps, 20)
                # Split into keypoint and agent_pos
                keypoint_obs = obs_raw[:, :, :18]  # (num_envs, obs_steps, 18)
                agent_pos_obs = obs_raw[:, :, 18:20]  # (num_envs, obs_steps, 2)

                # Normalize
                nkeypoint = (
                    dataset.normalizer["obs"]["keypoint"]
                    .normalize(keypoint_obs.reshape(-1, 2))
                    .reshape(config.task.num_envs, config.task.obs_steps, 18)
                )
                nagent_pos = dataset.normalizer["obs"]["agent_pos"].normalize(
                    agent_pos_obs
                )

                obs = {
                    "keypoint": torch.tensor(
                        nkeypoint,
                        device=config.optimization.device,
                        dtype=torch.float32,
                    ),
                    "agent_pos": torch.tensor(
                        nagent_pos,
                        device=config.optimization.device,
                        dtype=torch.float32,
                    ),
                }
            elif config.task.obs_type == "image":
                obs_raw = obs
                obs = {}
                for k in obs_raw:
                    obs[k] = obs_raw[k].astype(
                        np.float32
                    )  # (num_envs, obs_steps, obs_dim)
                    obs[k] = dataset.normalizer["obs"][k].normalize(obs[k])
                    obs[k] = torch.tensor(
                        obs[k], device=config.optimization.device, dtype=torch.float32
                    )  # (num_envs, obs_steps, obs_dim)
            else:
                raise ValueError(f"Invalid obs_type: {config.task.obs_type}")

            act_0 = torch.randn(
                (config.task.num_envs, config.task.horizon, config.task.act_dim),
                device=config.optimization.device,
            )
            # run sampling (num_envs, horizon, action_dim)
            act_normed = agent.sample(
                act_0=act_0,
                obs=obs,
                num_steps=num_steps,
                use_ema=True,
            )

            # unnormalize prediction
            act_normed = (
                act_normed.detach().to("cpu").numpy()
            )  # (num_envs, horizon, action_dim)
            act = dataset.normalizer["action"].unnormalize(act_normed)

            # get action by slicing from start to end
            start = config.task.obs_steps - 1
            end = start + config.task.act_steps
            act = act[:, start:end, :]

            obs, reward, terminated, truncated, _ = envs.step(act)
            _ = terminated | truncated  # Track done status
            ep_reward += reward
            step_reward.append(reward)
            t += config.task.act_steps

        success = np.around(np.max(np.array(step_reward), axis=0), 2)
        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)

    loguru.logger.info(
        f"Nstep: {num_steps} Mean step: {np.nanmean(episode_steps)} Mean reward: {np.nanmean(episode_rewards)} Mean success: {np.nanmean(episode_success)}"
    )

    metrics = {
        f"mean_step_{num_steps}": np.nanmean(episode_steps),
        f"mean_reward_{num_steps}": np.nanmean(episode_rewards),
        f"mean_success_{num_steps}": np.nanmean(episode_success),
    }

    return metrics


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
    """Main pipeline function that calls the appropriate standalone function based on mode."""
    # general config setup
    set_seed(config.optimization.seed)
    logger = Logger(config)
    loguru.logger.info("Finished setting up logger")

    # env setup
    envs = make_vec_env(config.task, seed=config.optimization.seed)
    obs, _ = envs.reset()
    loguru.logger.info("Finished setting up env")

    # dataset setup
    dataset = make_dataset(config.task)
    loguru.logger.info("Finished setting up dataset")

    agent = TrainingAgent(config)
    resume_state = None

    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(f"Loading model from {config.optimization.model_path}")
        resume_state = agent.load(config.optimization.model_path, load_optimizer=True)
    elif config.mode == "train" and config.optimization.auto_resume:
        # Automatically look for checkpoint to resume from
        checkpoint_base_name = (
            f"{config.task.env_name}_{config.task.env_type}_{config.task.obs_type}_"
            f"{config.optimization.loss_type}_{config.network.network_type}_"
            f"{config.network.emb_dim}_seed{config.optimization.seed}"
        )
        checkpoint_path = logger.find_latest_checkpoint(checkpoint_base_name)
        if checkpoint_path:
            loguru.logger.info(f"Found checkpoint to resume from: {checkpoint_path}")
            loguru.logger.info("Loading checkpoint with optimizer state...")
            resume_state = agent.load(str(checkpoint_path), load_optimizer=True)
        else:
            loguru.logger.info("No checkpoint found, starting training from scratch")
    elif config.mode == "train" and not config.optimization.auto_resume:
        loguru.logger.info("Auto-resume disabled, starting training from scratch")

    if config.mode == "train":
        train(config, envs, dataset, agent, logger, resume_state=resume_state)
    elif config.mode == "eval":
        agent.eval()

        num_steps_list = get_default_step_list(config.optimization.loss_type)
        for num_steps in num_steps_list:
            metrics = {"step": num_steps}
            metrics.update(evaluate(config, envs, dataset, agent, logger, num_steps))
            logger.log(metrics, category="eval")

        # print result in easy to read format
        for key, val in metrics.items():
            if "mean_success" in key:
                loguru.logger.info(f"{key} - {val}")
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    main()
