from dataclasses import dataclass, field


@dataclass
class LogConfig:
    log_dir: str
    wandb_mode: str
    project: str
    group: str
    exp_name: str
    eval_freq: int = 20000
    log_freq: int = 1000
    save_freq: int = 10000
    eval_episodes: int = 10
    save_video: bool = False


@dataclass
class OptimizationConfig:
    seed: int = 0
    loss_type: str = "flow"
    loss_scale: float = 100.0
    norm_type: str = "l2"
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_steps: int = 1
    sample_mode: str = "stochastic"  # "zero", "mean"
    t_two_step: float = 0.9
    discrete_dt: float = 0.01
    grad_clip_norm: float = 10.0
    ema_rate: float = 0.995
    batch_size: int = 1024
    gradient_steps: int = 300000
    warmup_ratio: float = 0.0
    rampup_ratio: float = 0.5
    min_value: float = 0.0
    max_value: float = 1.0
    model_path: str | None = None
    interp_type: str = "linear"  # "linear" or "trig"
    device: str = "cuda"


@dataclass
class NetworkConfig:
    network_type: str = "mlp"  # "mlp" or "cnn"
    encoder_type: str = "identity"  # "identity" or "mlp" or "image"
    num_layers: int = 4
    emb_dim: int = 512
    dropout: float = 0.1
    encoder_dropout: float = 0.0
    expansion_factor: int = 4


@dataclass
class TaskConfig:
    env_name: str = "lift"
    obs_type: str = "state"
    env_type: str = "ph"
    abs_action: bool = True
    # Dataset configuration - either HuggingFace or local path
    dataset_repo: str | None = (
        None  # HuggingFace repository ID (e.g., "ChaoyiPan/mip-dataset")
    )
    dataset_filename: str | None = (
        None  # Path within the repository (e.g., "robomimic/lift/ph/image.hdf5")
    )
    dataset_path: str | None = (
        None  # Local path (deprecated, use dataset_repo/dataset_filename)
    )
    max_episode_steps: int = 400
    obs_keys: list[str] = field(
        default_factory=lambda: [
            "object",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
    )
    obs_dim: int = -1
    act_dim: int = 10
    obs_steps: int = 2
    act_steps: int = 8
    horizon: int = 10  # Prediction horizon (typically obs_steps + act_steps)
    num_envs: int = 1
    save_video: bool = False
    shape_meta: dict = field(default_factory=dict)
    render_obs_key: str = "agentview_image"
    val_dataset_percentage: float = 0.0
    # Image observation settings
    rgb_model: str = "resnet18"
    resize_shape: list[int] | None = None
    crop_shape: list[int] | None = None
    random_crop: bool = True
    use_group_norm: bool = True
    use_seq: bool = True


@dataclass
class Config:
    optimization: OptimizationConfig
    network: NetworkConfig
    task: TaskConfig
    log: LogConfig
    mode: str = "train"  # "train" or "eval"
