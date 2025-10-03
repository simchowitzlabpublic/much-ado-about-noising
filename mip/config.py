from dataclasses import dataclass


@dataclass
class OptimizationConfig:
    loss_type: str = "flow"
    loss_scale: float = 100.0
    t_two_step: float = 0.9
    discrete_dt: float = 0.01
    norm_type: str = "l2"
