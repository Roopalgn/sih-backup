"""
dataset.py
----------
PyTorch Dataset for the forecast bust detection model.

Loads the preprocessed Zarr dataset(s) produced by data/preprocess.py.

Each sample:
  features:  [C, H, W]  - multi-channel input (forecast + static + temporal encoding)
  error_map: [1, H, W]  - regression target (absolute forecast error, mm/day)
  bust_map:  [1, H, W]  - classification target (binary bust label, 0 or 1)

Split files (new per-split zarr, produced by current preprocess.py):
  dataset_train.zarr, dataset_val.zarr, dataset_test.zarr

Legacy fallback (single zarr, year-filtered):
  dataset.zarr  — filtered by year using init_date coordinate
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

    Supports two file layouts produced by preprocess.py:
      1. Per-split zarr files: dataset_train.zarr, dataset_val.zarr, dataset_test.zarr
         (produced by current preprocess.py — preferred)
      2. Single zarr with year-based filtering: dataset.zarr
         (legacy fallback)

    Args:
        zarr_path: Path to a zarr file, or directory containing split zarr files.
                   Can be 'data/processed/' (will auto-find split file)
                   or 'data/processed/dataset_train.zarr' (explicit).
        split: 'train' | 'val' | 'test'
        transform: Optional callable applied to each sample dict
        max_samples: Optional cap (for quick testing)
    """

    SPLIT_YEARS: dict = {
        "train": [2021],
        "val":   [2022],
        "test":  [2022],
    }

    def __init__(
        self,
        zarr_path: str,
        split: str = "train",
        transform=None,
        max_samples: int | None = None,
        # Legacy year-override params (kept for backward compat)
        train_years: list | None = None,
        val_years:   list | None = None,
        test_years:  list | None = None,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), f"split must be train/val/test, got '{split}'"

        self.split     = split
        self.transform = transform

        import os
        zarr_path = str(zarr_path)

        # Auto-detect per-split vs. single zarr
        split_zarr = os.path.join(os.path.dirname(zarr_path), f"dataset_{split}.zarr")
        if os.path.isdir(split_zarr):
            load_path = split_zarr
            year_filter = False
            print(f"[Dataset] Loading {split} split from {split_zarr}")
        elif os.path.isdir(zarr_path):
            load_path = zarr_path
            year_filter = True
            print(f"[Dataset] Loading from single zarr {zarr_path}, filtering by year for '{split}'")
        else:
            raise FileNotFoundError(
                f"No Zarr data found at {zarr_path} or {split_zarr}. "
                f"Run: python data/preprocess.py"
            )

        ds = xr.open_zarr(load_path)

        if year_filter:
            # Legacy: filter by year from init_date coordinate
            if train_years:
                self.SPLIT_YEARS["train"] = train_years
            if val_years:
                self.SPLIT_YEARS["val"] = val_years
            if test_years:
                self.SPLIT_YEARS["test"] = test_years
            target_years = self.SPLIT_YEARS[split]

            if "init_date" in ds:
                dates = pd.DatetimeIndex(ds["init_date"].values)
                mask = dates.year.isin(target_years)
                indices = np.where(mask)[0]
            else:
                indices = np.arange(ds.dims.get("sample", len(ds["features"])))
        else:
            indices = np.arange(ds.dims.get("sample", len(ds["features"])))

        if max_samples is not None:
            indices = indices[:max_samples]

        self.features   = ds["features"].values[indices]    # [N, C, H, W]
        self.error_maps = ds["error_map"].values[indices]   # [N, 1, H, W]
        self.bust_maps  = ds["bust_map"].values[indices]    # [N, 1, H, W]
        self.n_samples  = len(indices)

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

    def __getitem__(self, idx: int) -> dict:
        features  = torch.from_numpy(self.features[idx].astype(np.float32))
        error_map = torch.from_numpy(self.error_maps[idx].astype(np.float32))
        bust_map  = torch.from_numpy(self.bust_maps[idx].astype(np.float32))

        sample = {
            "features":  features,
            "error_map": error_map,
            "bust_map":  bust_map,
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
    def spatial_shape(self) -> tuple:
        return self.features.shape[2], self.features.shape[3]

