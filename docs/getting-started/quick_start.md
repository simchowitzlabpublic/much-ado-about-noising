# Quick Start

## Installation

```
uv sync
```

## Training

We provide a few quick launch scripts for different tasks.

```
source .venv/bin/activate

# train robomimic
python examples/train_robomimic.py task=lift_ph_state network=chiunet optimization.loss_type=flow log.wandb_mode=online

# train kitchen
# NOTE: kitchen requires a older version of mujoco
uv pip install "mujoco==3.1.6"
python examples/train_kitchen.py task=kitchen_state

# train pusht
python examples/train_pusht.py task=pusht_state
```

Launch scripts with hydra multirun by adding `--multirun`:

```
uv run examples/train_robomimic.py task=lift_ph_state,lift_mh_state launcher=cluster -m
```
