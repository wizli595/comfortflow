"""Load project configuration from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def load_config(env: str = "base") -> dict:
    path = CONFIG_DIR / env / "parameters.yml"
    with open(path) as f:
        return yaml.safe_load(f)


_config = None


def get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config()
    return _config
