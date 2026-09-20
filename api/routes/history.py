"""
history.py
----------
GET /api/v1/historical route: returns historical bust events from the catalog.
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query

from api.schemas import HistoricalEvent, HistoricalResponse

router = APIRouter()

CATALOG_PATH = Path("dashboard/precomputed/event_catalog.json")


@router.get("/historical", response_model=HistoricalResponse)
async def get_historical(
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    year: Optional[int] = Query(None, description="Filter by year"),
    severity_min: int = Query(1, description="Minimum severity (1-5)"),
):
    """
    Return historical Indian weather bust events.

    Query parameters:
      - event_type: cyclone | monsoon_depression | heat_wave | western_disturbance | active_break
      - year: e.g. 2020
      - severity_min: Minimum severity level (1-5)
    """
    # Load catalog
    if CATALOG_PATH.exists():
        with open(CATALOG_PATH) as f:
            raw_events = json.load(f)
    else:
        # Inline fallback catalog
        raw_events = [
            {
                "name": "Cyclone Amphan", "event_type": "cyclone",
                "start_date": "2020-05-16", "end_date": "2020-05-21",
                "lat_center": 20.7, "lon_center": 87.5, "severity": 5,
                "basin": "Bay of Bengal",
                "description": "Super cyclonic storm Amphan — strongest Bay of Bengal cyclone in 21 years.",
                "imd_name": "SCS AMPHAN",
            },
            {
                "name": "Kerala Floods 2018", "event_type": "monsoon_depression",
                "start_date": "2018-08-14", "end_date": "2018-08-18",
                "lat_center": 16.0, "lon_center": 77.5, "severity": 5,
                "basin": "Central India / Kerala",
                "description": "Catastrophic Kerala floods triggered by multiple monsoon lows.",
                "imd_name": None,
            },
        ]

    # Filter
    events = raw_events
    if event_type:
        events = [e for e in events if e.get("event_type") == event_type]
    if year:
        events = [e for e in events if str(year) in e.get("start_date", "")]
    events = [e for e in events if e.get("severity", 1) >= severity_min]

    # Sort by date descending
    events = sorted(events, key=lambda e: e.get("start_date", ""), reverse=True)

    event_types = list(set(e.get("event_type", "unknown") for e in raw_events))

    return HistoricalResponse(
        total_events=len(events),
        events=[HistoricalEvent(**e) for e in events],
        event_types=sorted(event_types),
    )
