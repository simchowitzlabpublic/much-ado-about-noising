# Much Ado About Nosing: Dispelling the Myths of Generative Robotic Control

This repository contains the code for the paper "Much Ado About Noising: Dispelling the Myths of Generative Robotic Control".
We include common robot behavior cloning algorithms and easy to use dataset loading and training scripts.
This repository contains all the best practices discovered in paper and is designed to be a reference for future research on flow/consistency model/flow map model/shortcut model/regression model for robot learning.


## Features

- Clean: modular design, easy to use functions.
- Fast: support torch compile and cudagraph
- Performance: port best practice from diffusion training to flow / flow map policy training, auto resume support
- Diverse algorithm: to assist future research, support from regression to shortcut models.

## Installation

```
uv sync
# install robomimic dependencies
uv sync --extras robomimic
# install kitchen dependencies
uv sync --extras kitchen
```

## Training

```
# train robomimic (you can also run launch with hydra multirun by adding --multirun)
uv run examples/train_robomimic.py \
    task=lift_ph_state \ # specify the task
    network=chiunet \ # specify the network
    optimization.loss_type=flow \ # specify the loss type, support flow, regression, mip, tsd, psd, lsd, esd.
    log.wandb_mode=online \ # specify the wandb mode, support online, offline, disabled.

# train kitchen
uv run examples/train_kitchen.py task=kitchen_state

# train pusht
uv run examples/train_pusht.py task=pusht_state

# evaluate
uv run examples/train_robomimic.py optimization.model_path="" # specify the model path

# debug
uv run examples/train_robomimic.py -cn exps/debug.yaml
```

## Know Issues

CudaGraph is not supported for image-based tasks.

## Citation

```

```
