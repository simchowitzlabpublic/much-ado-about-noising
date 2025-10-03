"""Training pipeline for robomimic dataset.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import os
import pathlib
import time
from copy import deepcopy

import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

from mip.dataset_utils import loop_dataloader
from mip.logger import Logger, compute_average_metrics, update_best_metrics
from mip.scheduler import WarmupAnnealingScheduler
from mip.torch_utils import set_seed


def train(args, envs, dataset, agent, logger, val_dataset=None):
    """Standalone training function.

    Args:
        args: Arguments for training
        envs: Environment
        dataset: Training dataset
        agent: Agent to train
        logger: Logger for metrics
        val_dataset: Validation dataset (optional)
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4 if args.obs_type == "state" else 8,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True,
    )
    loop_loader = loop_dataloader(dataloader)
    lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps)

    # training loop
    n_gradient_step = 0
    start_time = time.time()
    warmup_scheduler = WarmupAnnealingScheduler(
        max_steps=args.gradient_steps,
        warmup_ratio=args.warmup_ratio,
        rampup_ratio=args.rampup_ratio,
        min_value=args.min_value,
        max_value=args.max_value,
    )
    info_list = []

    # track best evaluation metrics
    best_metrics = {}
    eval_history = []
    for batch in loop_loader:
        # preprocess
        naction = batch["action"].to(args.device)

        # get condition
        if args.obs_type == "image":
            nobs = batch["obs"]
            condition = {}
            for k in nobs.keys():
                condition[k] = nobs[k][:, : args.obs_steps, :].to(args.device)
            naction = batch["action"].to(args.device)
        elif args.obs_type == "state":
            nobs = batch["obs"]["state"].to(args.device)
            condition = nobs[:, : args.obs_steps, :]  # (B, obs_horizon, obs_dim)
            if (
                args.nn == "dit"
                or args.nn == "chi_unet"
                or args.nn == "dit_origin"
                or args.nn == "cnn"
            ):
                condition = condition.flatten(start_dim=1)  # (B, obs_horizon*obs_dim)
            else:
                pass  # (B, obs_horizon, obs_dim)

        # update diffusion
        delta = warmup_scheduler(n_gradient_step)

        info = agent.update(naction, delta, condition)
        lr_scheduler.step()
        info_list.append(info)

        if (n_gradient_step + 1) % args.log_freq == 0:
            metrics = {
                "step": n_gradient_step,
                "total_time": time.time() - start_time,
                "lr": lr_scheduler.get_last_lr()[0],
                "delta": delta,
            }
            for key in info.keys():
                try:
                    metrics[key] = np.nanmean([info[key] for info in info_list])
                except:
                    metrics[key] = np.nan
            logger.log(metrics, category="train")
            info_list = []

        if (n_gradient_step + 1) % args.save_freq == 0:
            logger.save_agent(agent=agent, identifier="latest")

        if (n_gradient_step + 1) % args.eval_freq == 0:
            print("Evaluate model...")
            agent.model.eval()
            agent.model_ema.eval()
            metrics = {"step": n_gradient_step}
            if (
                agent.regression_mode
                or "two_step" in args.loss_type
                or "straight_flow" in args.loss_type
            ):
                Nsteps = [1]
            elif args.loss_type == "mip":
                Nsteps = [1, 2, 3]
            else:
                Nsteps = 3 ** np.arange(0, 3)
                # if "block-push" in args.env_name:
                #     Nsteps = 3 ** np.arange(4, 0, -1)

            # Compute evaluation metrics
            for Nstep in Nsteps:
                metrics.update(eval(args, envs, dataset, agent, logger, Nstep))

            # Update best metrics
            best_metrics = update_best_metrics(best_metrics, metrics)

            # Add to evaluation history
            eval_history.append(metrics.copy())

            # Compute average metrics from last 5 evaluations
            avg_metrics = compute_average_metrics(eval_history)

            # Add best and average metrics to current metrics for logging
            for key, value in best_metrics.items():
                metrics[f"best_{key}"] = value
            for key, value in avg_metrics.items():
                metrics[key] = value

            # Print best and average metrics
            print("Best metrics so far:")
            for key, value in best_metrics.items():
                print(f"  {key}: {value:.4f}")
            if avg_metrics:
                print("Average metrics (last 5 evals):")
                for key, value in avg_metrics.items():
                    print(f"  {key}: {value:.4f}")

            logger.log(metrics, category="inference")
            agent.model.train()
            agent.model_ema.train()

        # Compute validation loss on a separate frequency
        if (n_gradient_step + 1) % args.val_freq == 0:
            if val_dataset is not None:
                print("Computing validation loss...")
                agent.model.eval()
                agent.model_ema.eval()
                val_metrics = {"step": n_gradient_step}
                if (
                    agent.regression_mode
                    or "two_step" in args.loss_type
                    or "regression" in args.loss_type
                ):
                    Nsteps = [1]
                else:
                    Nsteps = 3 ** np.arange(0, 3)

                for Nstep in Nsteps:
                    val_loss_metrics = compute_validation_loss(
                        args, val_dataset, agent, Nstep
                    )
                    val_metrics.update(val_loss_metrics)

                logger.log(val_metrics, category="inference")
                agent.model.train()
                agent.model_ema.train()

        n_gradient_step += 1
        if n_gradient_step > args.gradient_steps:
            # Print final summary
            print("\n" + "=" * 50)
            print("TRAINING SUMMARY")
            print("=" * 50)

            if best_metrics:
                print("Best evaluation metrics achieved:")
                for key, value in best_metrics.items():
                    print(f"  {key}: {value:.4f}")

            if eval_history:
                final_avg_metrics = compute_average_metrics(eval_history)
                if final_avg_metrics:
                    print("\nAverage metrics (last 5 evaluations):")
                    for key, value in final_avg_metrics.items():
                        print(f"  {key}: {value:.4f}")

            print("=" * 50)

            # finish
            logger.finish(agent)
            break


def eval(args, envs, dataset, agent, logger, Nstep=1):
    """Standalone inference function to evaluate a trained agent and optionally save a video.

    Args:
        args: Arguments for inference
        envs: Environment
        dataset: Dataset
        agent: Trained agent
        logger: Logger for metrics
        Nstep: Number of steps for sampling

    Returns:
        dict: Metrics including mean step, reward, and success rate
    """
    # ---------------- Start Rollout ----------------
    episode_rewards = []
    episode_steps = []
    episode_success = []
    episode_kit_success = []
    episode_block_push_success = []

    for i in range(args.eval_episodes // args.num_envs):
        ep_reward = [0.0] * args.num_envs
        obs, t = envs.reset(), 0

        # initialize video stream
        if args.save_video:
            logger.video_init(envs.envs[0], enable=True, video_id=str(i))  # save videos

        while t < args.max_episode_steps:
            if args.obs_type == "state":
                obs_seq = obs.astype(np.float32)  # (num_envs, obs_steps, obs_dim)
                # normalize obs
                nobs = dataset.normalizer["obs"]["state"].normalize(obs_seq)
                nobs = torch.tensor(
                    nobs, device=args.device, dtype=torch.float32
                )  # (num_envs, obs_steps, obs_dim)
                if (
                    args.nn == "chi_unet"
                    or args.nn == "dit"
                    or args.nn == "dit_origin"
                    or args.nn == "cnn"
                ):
                    # reshape observation to (num_envs, obs_horizon*obs_dim)
                    condition = nobs.flatten(start_dim=1)
                else:
                    # reshape observation to (num_envs, obs_horizon, obs_dim)
                    condition = nobs
            else:  # image-based observation
                obs_dict = {}
                for k in obs.keys():
                    obs_seq = obs[k].astype(
                        np.float32
                    )  # (num_envs, obs_steps, obs_dim)
                    nobs = dataset.normalizer["obs"][k].normalize(obs_seq)
                    obs_dict[k] = torch.tensor(
                        nobs, device=args.device, dtype=torch.float32
                    )  # (num_envs, obs_steps, obs_dim)
                condition = obs_dict
            # if agent.deterministic_mode:
            #     prior = torch.zeros(
            #         (args.num_envs, args.horizon, args.action_dim), device=args.device
            #     )
            # else:
            #     prior = torch.randn(
            #         (args.num_envs, args.horizon, args.action_dim), device=args.device
            #     )
            if args.sample_mode == "zero":
                prior = torch.zeros(
                    (args.num_envs, args.horizon, args.action_dim), device=args.device
                )
            elif args.sample_mode == "random":
                prior = torch.randn(
                    (args.num_envs, args.horizon, args.action_dim), device=args.device
                )
            elif args.sample_mode == "average":
                Naverage = 16
                prior = torch.randn(
                    (Naverage, args.num_envs, args.horizon, args.action_dim),
                    device=args.device,
                )
                # flatten condition and prior to merge dim 0 and 1
                if args.obs_type == "state":
                    if args.nn == "chi_unet" or args.nn == "dit_origin":
                        condition = condition.repeat(Naverage, 1)
                        condition_flat = condition.view(Naverage * args.num_envs, -1)
                    else:
                        condition = condition.repeat(Naverage, 1, 1)
                        condition_flat = condition.view(
                            Naverage * args.num_envs, *condition.shape[1:]
                        )
                elif args.obs_type == "image":
                    condition_flat = {}
                    for k in condition.keys():
                        condition_repeated = condition[k].repeat(Naverage, 1, 1)
                        condition_flat[k] = condition_repeated.view(
                            Naverage * args.num_envs, *condition[k].shape[1:]
                        )
                prior_flat = prior.view(
                    Naverage * args.num_envs, args.horizon, args.action_dim
                )

                with torch.no_grad():
                    # run sampling (Naverage * num_envs, horizon, action_dim)
                    naction_flat, log = agent.sample(
                        x0=prior_flat,
                        Nsteps=Nstep,
                        label=condition_flat,
                        use_ema=True,
                    )

                # reshape back and compute mean
                naction_reshaped = naction_flat.view(
                    Naverage, args.num_envs, args.horizon, args.action_dim
                )
                naction = naction_reshaped.mean(
                    dim=0
                )  # (num_envs, horizon, action_dim)

            else:
                raise ValueError(f"Invalid sample mode: {args.sample_mode}")

            if args.sample_mode != "average":
                with torch.no_grad():
                    # run sampling (num_envs, horizon, action_dim)
                    naction, log = agent.sample(
                        x0=prior,
                        Nsteps=Nstep,
                        label=condition,
                        use_ema=True,
                    )

            # unnormalize prediction
            naction = (
                naction.detach().to("cpu").numpy()
            )  # (num_envs, horizon, action_dim)
            action_pred = dataset.normalizer["action"].unnormalize(naction)

            # get action
            start = args.obs_steps - 1
            end = start + args.action_steps
            action = action_pred[:, start:end, :]

            if args.abs_action and args.env_name in [
                "can",
                "lift",
                "square",
                "tool_hang",
                "transport",
            ]:
                action = dataset.undo_transform_action(action)
            obs, reward, done, info = envs.step(action)
            ep_reward += reward
            t += args.action_steps
        success = [1.0 if s > 0 else 0.0 for s in ep_reward]

        # evaluate kitchen
        kit_success = []
        if "kitchen" in args.env_name:
            task_completion_counts = [
                len(info[i]["completed_tasks"][0]) for i in range(args.num_envs)
            ]
            for num in task_completion_counts:
                sublist = [1 if i < num else 0 for i in range(7)]
                kit_success.append(sublist)
            # Use p4 success rate as the main success metric for kitchen environments
            success = [1 if num >= 4 else 0 for num in task_completion_counts]

        # evalute block push
        block_push_success = []
        if "block-push" in args.env_name:
            p1_success = reward > 0.4  # (num_envs,)
            p2_success = reward > 0.9  # (num_envs,)
            block_push_success.append(p1_success)
            block_push_success.append(p2_success)
            success = p2_success

        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)
        episode_kit_success.append(kit_success)
        episode_block_push_success.append(block_push_success)
    print(
        f"Nstep: {Nstep} Mean step: {np.nanmean(episode_steps)} Mean reward: {np.nanmean(episode_rewards)} Mean success: {np.nanmean(episode_success)}"
    )

    metrics = {
        f"mean_step_{Nstep}": np.nanmean(episode_steps),
        f"mean_reward_{Nstep}": np.nanmean(episode_rewards),
        f"mean_success_{Nstep}": np.nanmean(episode_success),
    }

    if "kitchen" in args.env_name:
        mean_kit_success = np.mean(np.array(episode_kit_success), axis=(0, 1))
        kit_metrics = {}
        for i in range(7):
            kit_metrics[f"p{i + 1}_NFE{Nstep}"] = mean_kit_success[i]
        metrics.update(kit_metrics)
        print(f"Kit metrics: {kit_metrics}")

    if "block-push" in args.env_name:
        mean_block_push_success = np.mean(
            np.array(episode_block_push_success), axis=(0, -1)
        )  # average over episode and environment axis
        block_push_metrics = {}
        for i in range(2):
            block_push_metrics[f"p{i + 1}_NFE{Nstep}"] = mean_block_push_success[i]
        metrics.update(block_push_metrics)
        print(f"Block push metrics: {block_push_metrics}")

    # print("p1,p2,p3,p4,mean")
    # print(
    #     f"{mean_kit_success[0]:.2f}, {mean_kit_success[1]:.2f}, {mean_kit_success[2]:.2f}, {mean_kit_success[3]:.2f}, {mean_kit_success[:4].mean():.2f}"
    # )

    return metrics


def debug(args, envs, dataset, agent, logger):
    relabel_dataset(args, envs, dataset, agent, logger)
    # expand_dataset(args, envs, dataset, agent, logger)
    # args.loss_type = "lmd"
    # if "kitchen" in args.env_name:
    #     env_name = "kitchen"
    # else:
    #     env_name = args.env_name
    # args.model_path = f"checkpoints/{env_name}/{env_name}_{args.loss_type}_unet.pt"
    # agent = setup_agent(args)
    # for seed in range(1, 7):
    #     args.seed = seed
    #     log_rollout(args, envs, dataset, agent, logger)


def relabel_dataset(args, envs, dataset, agent, logger):
    # get all observation in dataset
    # get condition variable and sampled action
    relabel_mode = "smart"  # "deterministic" , "stochastic", "smart"
    if relabel_mode == "smart":
        assert args.n_hutchinson_samples > 0, (
            "n_hutchinson_samples must be greater than 0 for smart relabeling"
        )
    Ndata = len(dataset)
    Nsteps = 9
    c = []
    x1_gt = []
    for i in trange(Ndata):
        # create condition as normalized observation
        data = dataset[i]
        c.append(data["obs"]["state"][: args.obs_steps, :])
        x1_gt.append(data["action"])
    c = torch.stack(c, dim=0)
    x1_gt = torch.stack(x1_gt, dim=0)
    print(f"c.shape: {c.shape}, x1_gt.shape: {x1_gt.shape}")
    # sample x1 from diffusion model
    c_in = c.flatten(start_dim=1).to(args.device)
    if relabel_mode == "deterministic":
        x0 = torch.zeros_like(x1_gt, device=args.device)
    elif relabel_mode == "stochastic":
        x0 = torch.randn_like(x1_gt, device=args.device)
    elif relabel_mode == "smart":
        Ncandidates = 4
        Nsample, Hrollout, Na = x1_gt.shape
        x0 = torch.zeros_like(x1_gt, device=args.device)

        batch_size = 12
        Nprocess_step = Nsample // batch_size
        if Nsample % batch_size != 0:
            Nprocess_step += 1

        for i in trange(Nprocess_step):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, Nsample)
            batch_size_i = end_idx - start_idx

            # Generate candidates for this batch
            x0_batch = torch.randn(
                Ncandidates, batch_size_i, Hrollout, Na, device=args.device
            )
            x0_flat = x0_batch.reshape(Ncandidates * batch_size_i, Hrollout, Na)

            # Setup time vectors
            ss = torch.zeros(Ncandidates * batch_size_i, device=args.device)
            tt = torch.ones(Ncandidates * batch_size_i, device=args.device)

            # Get and repeat conditions
            c_in_batch = c_in[start_idx:end_idx]
            c_in_repeat = c_in_batch.repeat(Ncandidates, 1)

            # Compute nabla_x0 of flow
            nabla_x0 = agent.flow_map.spec_norm_multi_step(
                ss, tt, x0_flat, c_in_repeat, Nsteps=Nsteps
            )
            nabla_x0 = nabla_x0.reshape(Ncandidates, batch_size_i)

            # Select best candidate for each sample in batch
            _, idx = torch.min(nabla_x0, dim=0)
            x0[start_idx:end_idx] = x0_batch[idx, torch.arange(batch_size_i)]
    else:
        raise ValueError(f"Invalid relabel mode: {relabel_mode}")

    batch_size = 8192
    x1_pred = np.zeros(x1_gt.shape)
    Nprocess_step = Ndata // batch_size
    if Ndata % batch_size != 0:
        Nprocess_step += 1
    for i in trange(Nprocess_step):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, Ndata)
        c_in_batch = c_in[start_idx:end_idx]
        x0_batch = x0[start_idx:end_idx]
        with torch.no_grad():
            naction, _ = agent.sample(
                x0=x0_batch,
                Nsteps=Nsteps,
                label=c_in_batch,
            )
            x1_pred[start_idx:end_idx] = naction.detach().to("cpu").numpy()
    x1_gt = x1_gt.detach().to("cpu").numpy()
    c = c.detach().to("cpu").numpy()
    np.savez(
        f"./results/robomimic/{args.env_name}/{args.loss_type}/{args.seed}/relabel_dataset.npz",
        c=c,
        x1_gt=x1_gt,
        x1_pred=x1_pred,
    )


def expand_dataset(args, envs, dataset, agent, logger):
    # config
    Nstep = 50 if args.loss_type == "flow" else 1
    Nrollout = 1
    Nenv = args.num_envs
    Nrepeat = Nrollout // Nenv
    sample_mode = "deterministic"  # "deterministic" or "stochastic"

    obs = envs.reset()

    # rollout
    info_list = []
    success_info_list = []
    for i in range(Nrepeat):
        traj_info_list = []
        obs = envs.reset()
        for time_step in trange(0, args.max_episode_steps, args.action_steps):
            obs_seq = obs.astype(np.float32)  # (Nenv, obs_steps, obs_dim)
            nobs = dataset.normalizer["obs"]["state"].normalize(obs_seq)
            nobs = torch.tensor(nobs, device=args.device, dtype=torch.float32)
            with torch.no_grad():
                condition_transformer = nobs
                condition = nobs.flatten(start_dim=1)
                if sample_mode == "deterministic":
                    prior = torch.zeros(
                        (Nenv, args.horizon, args.action_dim), device=args.device
                    )
                else:
                    prior = torch.randn(
                        (Nenv, args.horizon, args.action_dim), device=args.device
                    )
                naction, log = agent.sample(
                    x0=prior,
                    Nsteps=Nstep,
                    label=(
                        condition
                        if args.nn == "dit"
                        or args.nn == "chi_unet"
                        or args.nn == "dit_origin"
                        else condition_transformer
                    ),
                )
            action = naction.detach().to("cpu").numpy()
            action_pred = dataset.normalizer["action"].unnormalize(action)
            start = args.obs_steps - 1
            end = start + args.action_steps
            action = action_pred[:, start:end, :]
            if args.abs_action and args.env_name in [
                "can",
                "lift",
                "square",
                "tool_hang",
                "transport",
            ]:
                action = dataset.undo_transform_action(action)
            obs, reward, done, env_info = envs.step(action)
            info = {
                "x0": prior.detach().to("cpu").numpy(),
                "x1": naction.detach().to("cpu").numpy(),
                "c": condition.detach().to("cpu").numpy(),
                "c_transformer": condition_transformer.detach().to("cpu").numpy(),
            }
            info_list.append(info)
            traj_info_list.append(info)

            if "kitchen" in args.env_name:
                print(env_info["completed_tasks"][0])
            else:
                if np.all(reward > 0):
                    print(f"success rate: {np.mean(reward)}")
                    success_info_list.extend(traj_info_list)
                    break
    info_aggregated = {}
    success_info_aggregated = {}
    for k in info_list[0].keys():
        info_aggregated[k] = np.stack([info[k] for info in info_list], axis=0)
        success_info_aggregated[k] = np.stack(
            [info[k] for info in success_info_list], axis=0
        )

    path = f"./results/robomimic/{args.env_name}/{args.loss_type}/{args.seed}"
    os.makedirs(path, exist_ok=True)
    np.savez(
        f"{path}/expand_dataset_{args.nn}_{Nrollout}_{sample_mode}.npz",
        **info_aggregated,
    )
    np.savez(
        f"{path}/expand_dataset_success_{args.nn}_{Nrollout}_{sample_mode}.npz",
        **success_info_aggregated,
    )


@hydra.main(config_path="configs/", config_name="main")
def main(config):
    """Main pipeline function that calls the appropriate standalone function based on mode."""
    set_seed(config.seed)
    logger = Logger(pathlib.Path(config.work_dir), config)
    print(f"making env: {config.env_name}")
    envs = make_vec_env(config)
    obs = envs.reset()
    if config.obs_type == "state":
        config.obs_dim = obs.shape[-1]
    dataset = make_dataset(config, mode="train")

    # Create validation dataset if validation percentage > 0
    val_dataset = None
    if config.val_dataset_percentage > 0.0:
        val_dataset = make_dataset(config, mode="val")

    agent = setup_agent(config)
    if config.model_path and config.model_path != "None":
        print(f"Loading model from {config.model_path}")
        agent.load(config.model_path)

    if config.mode == "train":
        train(config, envs, dataset, agent, logger, val_dataset)
    elif config.mode == "eval":
        if not config.model_path:
            raise ValueError("Empty model for inference")
        agent.model.eval()
        agent.model_ema.eval()

        if config.loss_type in ["flow", "ctm", "ctm_discrete"]:
            Nstep_list = 3 ** np.arange(2, -1, -1)
            # if "block-push" in args.env_name:
            #     Nstep_list = 3 ** np.arange(4, 0, -1)
        elif (
            config.loss_type in ["regression", "three_step", "straight_flow"]
            or "two_step" in config.loss_type
        ):
            Nstep_list = [1]
        elif config.loss_type == "mip":
            Nstep_list = [1, 2, 3]
        else:
            Nstep_list = 3 ** np.arange(2, -1, -1)

        # TODO: first compute validation loss
        if val_dataset is not None:
            print("Computing validation loss...")
            agent.model.eval()
            agent.model_ema.eval()
            val_metrics = {"step": 0}
            for Nstep in Nstep_list:
                val_metrics.update(
                    compute_validation_loss(config, val_dataset, agent, Nstep)
                )
            logger.log(val_metrics, category="inference")
            agent.model.train()
            agent.model_ema.train()

        for i in range(5):
            for Nstep in Nstep_list:
                metrics = {"step": i}
                metrics.update(eval(config, envs, dataset, agent, logger, Nstep))
                logger.log(metrics, category="inference")

        # print result in easy to read format
        for key, val in metrics.items():
            if "mean_success" in key:
                print(f"{key} - {val}")
    elif config.mode == "debug":
        debug(config, envs, dataset, agent, logger)
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    main()
