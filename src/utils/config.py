from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    def __getattr__(self, item):
        value = self._config.get(item)

        if isinstance(value, dict):
            return Config(value)

        return value

    def to_dict(self):
        return self._config


def load_config(config_path: str) -> Config:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return Config(config)