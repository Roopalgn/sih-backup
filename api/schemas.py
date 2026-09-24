"""
schemas.py
----------
Pydantic request/response schemas for the Forecast Bust Detection API.
"""

from __future__ import annotations
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, validator


class PredictRequest(BaseModel):
    """Request body for POST /api/v1/predict."""

    date: str = Field(
        ...,
        description="Forecast initialisation date (YYYY-MM-DD)",
        example="2022-06-15",
    )
    lead_day: int = Field(
        ...,
        description="Forecast lead time in days (1-10)",
        ge=1,
        le=10,
        example=3,
    )
    variable: str = Field(
        default="rainfall",
        description="Target variable",
        example="rainfall",
    )
    use_cache: bool = Field(
        default=True,
        description="Return precomputed prediction if available (faster for demo)",
    )

    @validator("variable")
    def variable_must_be_valid(cls, v):
        allowed = {"rainfall", "temperature", "z500", "wind850"}
        if v not in allowed:
            raise ValueError(f"variable must be one of {allowed}")
        return v


class GridCellPrediction(BaseModel):
    """Prediction for a single grid cell."""
    lat: float
    lon: float
    bust_probability: float = Field(..., ge=0.0, le=1.0)
    error_magnitude: float = Field(..., ge=0.0)
    confidence: float = Field(..., ge=0.0, le=100.0)


class MeteoDriver(BaseModel):
    """A single meteorological driver from the explainability module."""
    rank: int
    channel_name: str
    attribution_pct: float
    description: str


class BustRegion(BaseModel):
    """A region with high bust probability."""
    name: str
    lat_center: float
    lon_center: float
    mean_bust_prob: float
    area_fraction: float  # Fraction of the region's cells with bust_prob > 0.5


class PredictResponse(BaseModel):
    """Response for POST /api/v1/predict."""

    request_date: str
    lead_day: int
    variable: str

    # Grid-level output
    grid_latitudes: list[float] = Field(..., description="Latitude values of output grid")
    grid_longitudes: list[float] = Field(..., description="Longitude values of output grid")
    bust_probability_map: list[list[float]] = Field(..., description="[H x W] bust probability [0,1]")
    error_magnitude_map: list[list[float]] = Field(..., description="[H x W] predicted error (mm/day)")
    confidence_map: list[list[float]] = Field(..., description="[H x W] confidence [0-100]")

    # Summary statistics
    mean_bust_probability: float
    mean_confidence: float
    high_bust_regions: list[BustRegion]

    # Explainability
    top_drivers: list[MeteoDriver]
    event_type: Optional[str] = None

    # Data provenance — always present, never "live_model" unless model actually ran
    data_source: str = Field(
        default="unknown",
        description=(
            "'live_model': ForecastBustUNet ran on real forecast data. "
            "'precomputed_cache': loaded from pre-saved JSON. "
            "'illustrative_only': hand-authored numbers, NOT model output — label visibly in UI."
        ),
    )

    # Legacy field — kept for backward compat with old cached JSONs
    from_cache: bool = False



class HistoricalEvent(BaseModel):
    """Historical bust event from the catalog."""
    name: str
    event_type: str
    start_date: str
    end_date: str
    lat_center: float
    lon_center: float
    severity: int
    basin: str
    description: str
    imd_name: Optional[str] = None


class HistoricalResponse(BaseModel):
    """Response for GET /api/v1/historical."""
    total_events: int
    events: list[HistoricalEvent]
    event_types: list[str]
    data_source: str = Field(
        default="reference_catalog",
        description="Always 'reference_catalog' — this endpoint returns metadata, not model predictions.",
    )
    catalog_note: Optional[str] = Field(
        default=None,
        description="Human-readable note clarifying this is curated metadata, not model output.",
    )
