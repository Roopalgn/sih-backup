"""
dataset.py
----------
PyTorch Dataset for the forecast bust detection model.

Loads the preprocessed Zarr dataset produced by data/preprocess.py.

Each sample:
  features:  [C, H, W]  - multi-channel input (forecast + static + temporal encoding)
  error_map: [1, H, W]  - regression target (absolute forecast error)
  bust_map:  [1, H, W]  - classification target (binary bust label)

Splits are year-based:
  train: 2018-2021
  val:   2022
  test:  2023
"""

from __future__ import annotations

import json
import numpy as np
import xarray as xr
import pandas as pd
import torch
from torch.utils.data import Dataset


class BustDataset(Dataset):
    """
    PyTorch Dataset wrapping the preprocessed Zarr dataset.

    Args:
        zarr_path: Path to dataset.zarr
        split: 'train' | 'val' | 'test'
        train_years: Years to include in train split
        val_years: Years to include in val split
        test_years: Years to include in test split
        transform: Optional callable applied to (features, error_map, bust_map) tuple
        max_samples: Optional cap on number of samples (for quick testing)
    """

    SPLIT_YEARS: dict[str, list[int]] = {
        "train": [2018, 2019, 2020, 2021],
        "val": [2022],
        "test": [2023],
    }

    def __init__(
        self,
        zarr_path: str,
        split: str = "train",
        train_years: list[int] | None = None,
        val_years: list[int] | None = None,
        test_years: list[int] | None = None,
        transform=None,
        max_samples: int | None = None,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), f"split must be train/val/test, got '{split}'"

        # Override default year splits if provided
        if train_years:
            self.SPLIT_YEARS["train"] = train_years
        if val_years:
            self.SPLIT_YEARS["val"] = val_years
        if test_years:
            self.SPLIT_YEARS["test"] = test_years

        self.split = split
        self.transform = transform

        print(f"[Dataset] Loading {split} data from {zarr_path}")
        ds = xr.open_zarr(zarr_path)

        # Filter samples by year
        target_years = self.SPLIT_YEARS[split]
        if "init_date" in ds:
            dates = pd.DatetimeIndex(ds["init_date"].values)
            mask = dates.year.isin(target_years)
            indices = np.where(mask)[0]
        else:
            # Fall back to all samples
            indices = np.arange(ds.dims.get("sample", len(ds["features"])))

        if max_samples is not None:
            indices = indices[:max_samples]

        # Load all data into memory (for small datasets) or keep as Zarr references
        self.features = ds["features"].values[indices]   # [N, C, H, W]
        self.error_maps = ds["error_map"].values[indices] # [N, 1, H, W]
        self.bust_maps = ds["bust_map"].values[indices]   # [N, 1, H, W]
        self.n_samples = len(indices)

        # Store metadata
        if "init_date" in ds:
            self.dates = pd.DatetimeIndex(ds["init_date"].values[indices])
        else:
            self.dates = None
        if "lead_day" in ds:
            self.lead_days = ds["lead_day"].values[indices]
        else:
            self.lead_days = None

        bust_rate = self.bust_maps.mean()
        print(
            f"[Dataset] {split}: {self.n_samples} samples | "
            f"shape={self.features.shape[1:]} | bust_rate={bust_rate:.3f}"
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        features = torch.from_numpy(self.features[idx].astype(np.float32))
        error_map = torch.from_numpy(self.error_maps[idx].astype(np.float32))
        bust_map = torch.from_numpy(self.bust_maps[idx].astype(np.float32))

        sample = {
            "features": features,
            "error_map": error_map,
            "bust_map": bust_map,
        }

        if self.dates is not None:
            sample["date"] = str(self.dates[idx].date())
        if self.lead_days is not None:
            sample["lead_day"] = int(self.lead_days[idx])

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    @property
    def n_channels(self) -> int:
        return self.features.shape[1]

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.features.shape[2], self.features.shape[3]
