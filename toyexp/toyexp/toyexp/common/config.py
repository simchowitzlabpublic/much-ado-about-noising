"""
Configuration loading and management for toy experiments.

Simple YAML-based configuration without Hydra dependency.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class Config:
    """
    Simple configuration class that supports dict-like and attribute access.

    Example:
        >>> config = Config({'a': 1, 'b': {'c': 2}})
        >>> config.a  # Returns 1
        >>> config['a']  # Returns 1
        >>> config.b.c  # Returns 2
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary."""
        self._config = config_dict

        # Convert nested dicts to Config objects
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        """Dict-style access."""
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-style assignment."""
        self._config[key] = value
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to plain dictionary."""
        result = {}
        for key, value in self._config.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self._config})"


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object with configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def save_config(config: Union[Config, Dict], save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Config object or dictionary to save
        save_path: Path where to save the configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert Config to dict if necessary
    if isinstance(config, Config):
        config_dict = config.to_dict()
    else:
        config_dict = config

    with open(save_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    logger.info(f"Saved configuration to {save_path}")


def merge_configs(base_config: Config, override_config: Dict[str, Any]) -> Config:
    """
    Merge two configurations, with override taking precedence.

    Args:
        base_config: Base configuration
        override_config: Dictionary with values to override

    Returns:
        New Config with merged values
    """
    merged = base_config.to_dict()

    def deep_update(base_dict: Dict, update_dict: Dict) -> None:
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    deep_update(merged, override_config)
    return Config(merged)


def parse_override_args(args: list[str]) -> Dict[str, Any]:
    """
    Parse command-line override arguments in the format key=value.

    Supports nested keys using dot notation:
        experiment.mode=flow
        training.learning_rate=0.01
        dataset.num_train=100

    Args:
        args: List of override arguments

    Returns:
        Dict with nested structure for config overrides

    Raises:
        ValueError: If any override format is invalid

    Example:
        >>> parse_override_args(['experiment.mode=flow', 'training.lr=0.01'])
        {'experiment': {'mode': 'flow'}, 'training': {'lr': 0.01}}
    """
    overrides = {}

    for arg in args:
        if "=" not in arg:
            raise ValueError(f"Invalid override format (missing '='): {arg}")

        key_path, value = arg.split("=", 1)
        keys = key_path.split(".")

        # Try to convert value to appropriate type
        value = _parse_value(value)

        # Build nested dict
        current = overrides
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    return overrides


def _parse_value(value_str: str) -> Any:
    """
    Parse a string value to appropriate Python type.

    Args:
        value_str: String value to parse

    Returns:
        Parsed value (int, float, bool, or str)
    """
    # Try boolean
    if value_str.lower() in ("true", "yes", "on", "1"):
        return True
    if value_str.lower() in ("false", "no", "off", "0"):
        return False

    # Try int
    try:
        return int(value_str)
    except ValueError:
        pass

    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass

    # Return as string
    return value_str


def validate_overrides(
    base_config: Dict[str, Any], overrides: Dict[str, Any], path: str = ""
) -> None:
    """
    Validate that all override keys exist in the base configuration.

    Args:
        base_config: Base configuration dictionary
        overrides: Override dictionary to validate
        path: Current path (for error messages)

    Raises:
        ValueError: If any override key doesn't exist in base config
    """
    for key, value in overrides.items():
        current_path = f"{path}.{key}" if path else key

        if key not in base_config:
            raise ValueError(
                f"Invalid override key: '{current_path}' does not exist in config. "
                f"Available keys at this level: {list(base_config.keys())}"
            )

        # If the value is a dict, recursively validate
        if isinstance(value, dict):
            if not isinstance(base_config[key], dict):
                raise ValueError(
                    f"Invalid override: '{current_path}' is not a nested config section"
                )
            validate_overrides(base_config[key], value, current_path)


def apply_overrides(config: Config, overrides: Dict[str, Any]) -> Config:
    """
    Apply command-line overrides to a config, with validation.

    Args:
        config: Base configuration
        overrides: Dictionary of overrides (from parse_override_args)

    Returns:
        New Config with overrides applied

    Raises:
        ValueError: If any override key is invalid
    """
    # Validate that all override keys exist
    validate_overrides(config.to_dict(), overrides)

    # Apply overrides
    return merge_configs(config, overrides)


def validate_config(config: Config) -> None:
    """
    Validate configuration has required fields.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Check required top-level sections
    required_sections = ["experiment", "dataset", "network", "training", "evaluation"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate experiment section
    if "mode" not in config.experiment:
        raise ValueError("Missing experiment.mode")
    if config.experiment.mode not in [
        "regression",
        "flow",
        "mip",
        "mip_one_step_integrate",
        "straight_flow",
    ]:
        raise ValueError(f"Invalid mode: {config.experiment.mode}")

    # Validate dataset section
    if config.dataset.type not in [
        "TargetFunctionDataset",
        "ProjectedTargetFunctionDataset",
        "LieAlgebraRotationDataset",
    ]:
        raise ValueError(f"Invalid dataset type: {config.dataset.type}")

    # Validate network section
    if config.network.architecture not in ["concat", "film"]:
        raise ValueError(f"Invalid network architecture: {config.network.architecture}")

    # Validate training section
    if config.training.loss_type not in ["l1", "l2", "flow_matching"]:
        raise ValueError(f"Invalid loss type: {config.training.loss_type}")

    # Ensure consistency between experiment mode and loss type
    if config.experiment.mode == "regression" and config.training.loss_type not in [
        "l1",
        "l2",
    ]:
        logger.warning(
            "Experiment mode is 'regression' but loss type is not 'l1 or l2'"
        )
    elif (
        config.experiment.mode == "flow"
        and config.training.loss_type != "flow_matching"
    ):
        logger.warning("Experiment mode is 'flow' but loss type is not 'flow_matching'")
    elif (
        config.experiment.mode == "mip" and config.training.loss_type != "flow_matching"
    ):
        logger.warning("Experiment mode is 'mip' but loss type is not 'flow_matching'")

    logger.info("Configuration validation passed")


# =============================================================================
# Testing Code
# =============================================================================

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test Config class
    logger.info("Testing Config class...")
    test_dict = {
        "experiment": {"name": "test", "seed": 42},
        "training": {"batch_size": 32},
    }

    config = Config(test_dict)
    assert config.experiment.name == "test"
    assert config["experiment"]["seed"] == 42
    assert config.training.batch_size == 32

    # Test to_dict
    assert config.to_dict() == test_dict

    # Test merge
    override = {"training": {"batch_size": 64, "lr": 0.001}}
    merged = merge_configs(config, override)
    assert merged.training.batch_size == 64
    assert merged.training.lr == 0.001
    assert merged.experiment.name == "test"  # Original value preserved

    logger.info("All tests passed!")

    # Example of loading a config (if file exists)
    logger.info("\nTo use in practice:")
    logger.info("  config = load_config('config_recon.yaml')")
    logger.info("  validate_config(config)")
    logger.info("  # Access values:")
    logger.info("  print(config.experiment.name)")
    logger.info("  print(config.training.batch_size)")
