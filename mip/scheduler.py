"""
This file contains the scheduler for the training process.

Author: Chaoyi Pan
Date: 2025-04-17
"""

import numpy as np
from matplotlib import pyplot as plt


class WarmupAnnealingScheduler:
    def __init__(
        self,
        max_steps,
        warmup_ratio=0.1,
        rampup_ratio=0.8,
        min_value=0.0,
        max_value=1.0,
    ):
        """
        Args:
            max_steps: Total number of gradient steps in training
            warmup_ratio: Ratio of steps to stay at 0 (e.g., 0.1 means first 10% of steps)
            rampup_ratio: Ratio of steps to ramp up from 0 to max_value (e.g., 0.2 means 20% of steps)
            max_value: The maximum value to reach
        """
        self.max_steps = max_steps
        self.warmup_steps = int(max_steps * warmup_ratio)
        self.rampup_steps = int(max_steps * rampup_ratio)
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.min_value
        elif step < self.warmup_steps + self.rampup_steps:
            # Linear ramp from 0 to max_value
            progress = (step - self.warmup_steps) / self.rampup_steps
            return progress * (self.max_value - self.min_value) + self.min_value
        else:
            return self.max_value
