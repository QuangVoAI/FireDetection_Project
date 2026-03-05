"""
src/config.py
-------------
Configuration loader for the Early Fire Detection System.

Reads configs/default.yaml and returns a nested object with dot-notation access.
"""

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigNode:
    """Recursively wraps a dictionary to support dot-notation attribute access."""

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        return f"ConfigNode({self.__dict__})"

    def to_dict(self) -> dict:
        """Convert the config node back to a plain dictionary."""
        result: dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key with an optional default."""
        return getattr(self, key, default)


def load_config(path: str = "configs/default.yaml") -> ConfigNode:
    """Load a YAML configuration file and return a dot-notation accessible object.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        ConfigNode: Nested object with dot-notation access for all config values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.

    Example:
        >>> config = load_config("configs/default.yaml")
        >>> print(config.model.architecture)   # "rtdetr-l"
        >>> print(config.training.epochs)      # 100
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path.resolve()}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f)

    if raw is None:
        raw = {}

    return ConfigNode(raw)
