# Much Ado About Noising: Do Flow Models Actually Make Better Control Policies?

This repository contains the code for the paper "Much Ado About Noising: Do Flow Models Actually Make Better Control Policies?".
We include common robot behavior cloning algorithms and easy to use dataset loading and training scripts.
This repository contains all the best practices discovered in paper and is designed to be a reference for future research on flow/consistency model/flow map model/shortcut model/regression model for robot learning.


## Features

- Minimum
- Performance: port best practice from diffusion training to flow / flow map policy training
- Diverse algorithm: to assist future research, support from regression to shortcut models.

## Installation

```
uv sync
```

## Training

```
uv run examples/train_robomimic.py
```
