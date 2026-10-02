"""API-backed showcase-event catalog for the React dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()

PRECOMPUTED_DIR = Path("dashboard/precomputed")
EVENT_FIELDS = (
    "event_name",
    "request_date",
    "lead_day",
    "variable",
    "event_type",
    "description",
    "data_source",
)


@router.get("/events")
async def events():
    """Return event-selector metadata without sending large gridded maps."""
    events_path = PRECOMPUTED_DIR / "events.json"
    if not events_path.exists():
        return {"events": [], "data_source": "unavailable"}

    with events_path.open(encoding="utf-8") as handle:
        source_events = json.load(handle)

    return {
        "events": [
            {field: event.get(field) for field in EVENT_FIELDS}
            for event in source_events
            if event.get("event_name") and event.get("request_date") and event.get("lead_day")
        ],
        "data_source": "precomputed_catalog",
    }
