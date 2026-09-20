"""Utility to load YAML config."""
import yaml
from pathlib import Path


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
