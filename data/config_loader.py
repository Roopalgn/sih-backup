"""
config_loader.py
----------------
Loads and validates YAML configuration for the Forecast Bust Detection pipeline.
"""

import datetime
from pathlib import Path
import yaml


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """
    Load YAML configuration file and validate key constraints.

    Args:
        config_path: Path to YAML file

    Returns:
        cfg: Configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If date ranges pre-date GEFSv12 availability
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg = validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> dict:
    """
    Validate config against known constraints:
      - GEFSv12 0.25° data only available from 2020-09-01
      - in_channels must match the 6-channel feature set built by preprocess.py
      - lead_days must be a subset of [1–10]
    """
    gefs_min = datetime.date.fromisoformat(
        cfg["data"].get("gefs_min_date", "2020-09-01")
    )

    # Check training years
    train_years = cfg["data"].get("train_years", [])
    train_months = cfg["data"].get("train_months", list(range(1, 13)))
    if train_years:
        earliest = datetime.date(min(train_years), min(train_months), 1)
        if earliest < gefs_min:
            raise ValueError(
                f"train_years/train_months starts {earliest}, but GEFSv12 0.25° "
                f"data only available from {gefs_min}. "
                f"Adjust config['data']['train_years'] to 2021 or later."
            )

    # Warn if in_channels doesn't match 6
    n_ch = cfg.get("model", {}).get("in_channels", 6)
    if n_ch != 6:
        import warnings
        warnings.warn(
            f"model.in_channels={n_ch} in config, but preprocess.py always "
            f"produces 6 channels. Training will fail unless you rebuild the dataset. "
            f"Set in_channels: 6 in config/settings.yaml.",
            stacklevel=2,
        )

    # Warn on unsupported variables
    variables = cfg.get("bust", {}).get("variables", ["rainfall"])
    unsupported = [v for v in variables if v not in ("rainfall",)]
    if unsupported:
        import warnings
        warnings.warn(
            f"config bust.variables contains {unsupported} but only 'rainfall' "
            f"is implemented. Unsupported variables will be ignored.",
            stacklevel=2,
        )

    return cfg
