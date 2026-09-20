"""
event_catalog.py
----------------
Curated catalog of significant Indian weather events for:
  1. Stratified train/val/test splitting
  2. Dashboard showcase events
  3. Explainability reference labelling

Events are stored as a list of dicts with:
  - name: Human-readable event name
  - type: Event category (cyclone, monsoon_depression, heat_wave, western_disturbance, active_break)
  - start_date / end_date: Date range
  - basin: Geographic focus area
  - lat_center / lon_center: Approximate centre
  - severity: 1 (moderate) to 5 (catastrophic)
  - description: Brief meteorological description
"""

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Optional
import json
from pathlib import Path


@dataclass
class WeatherEvent:
    name: str
    event_type: str  # cyclone | monsoon_depression | heat_wave | western_disturbance | active_break
    start_date: date
    end_date: date
    lat_center: float
    lon_center: float
    severity: int  # 1–5
    basin: str
    description: str
    imd_name: Optional[str] = None  # Official IMD designation

    def to_dict(self):
        d = asdict(self)
        d["start_date"] = self.start_date.isoformat()
        d["end_date"] = self.end_date.isoformat()
        return d


# ─────────────────────────────────────────────────────
# Event Catalog
# ─────────────────────────────────────────────────────
EVENT_CATALOG: list[WeatherEvent] = [
    # ── Cyclones ────────────────────────────────────
    WeatherEvent(
        name="Cyclone Amphan",
        event_type="cyclone",
        start_date=date(2020, 5, 16),
        end_date=date(2020, 5, 21),
        lat_center=20.7,
        lon_center=87.5,
        severity=5,
        basin="Bay of Bengal",
        description="Super cyclonic storm Amphan — strongest Bay of Bengal cyclone in 21 years. "
                    "Landfall near Sundarbans delta. Category 5 equivalent.",
        imd_name="SCS AMPHAN",
    ),
    WeatherEvent(
        name="Cyclone Fani",
        event_type="cyclone",
        start_date=date(2019, 4, 26),
        end_date=date(2019, 5, 4),
        lat_center=19.5,
        lon_center=86.7,
        severity=5,
        basin="Bay of Bengal",
        description="Extremely Severe Cyclonic Storm Fani. Landfall Puri, Odisha. "
                    "Rapid intensification event; NWP models initially underestimated intensification.",
        imd_name="ESCS FANI",
    ),
    WeatherEvent(
        name="Cyclone Tauktae",
        event_type="cyclone",
        start_date=date(2021, 5, 14),
        end_date=date(2021, 5, 19),
        lat_center=21.0,
        lon_center=72.0,
        severity=5,
        basin="Arabian Sea",
        description="Extremely Severe Cyclone over Arabian Sea. Landfall Gujarat. "
                    "Notable RI event over warm Arabian Sea eddies.",
        imd_name="ESCS TAUKTAE",
    ),
    # ── Monsoon Depressions ─────────────────────────
    WeatherEvent(
        name="Kerala Floods Monsoon Depression",
        event_type="monsoon_depression",
        start_date=date(2018, 8, 14),
        end_date=date(2018, 8, 18),
        lat_center=16.0,
        lon_center=77.5,
        severity=5,
        basin="Central India / Kerala",
        description="Catastrophic floods in Kerala triggered by multiple monsoon lows over Bay of Bengal. "
                    "D5–D7 forecasts significantly underestimated rainfall over Western Ghats.",
    ),
    WeatherEvent(
        name="Odisha Monsoon Depression 2022",
        event_type="monsoon_depression",
        start_date=date(2022, 9, 22),
        end_date=date(2022, 9, 26),
        lat_center=20.5,
        lon_center=85.0,
        severity=3,
        basin="Bay of Bengal → Odisha",
        description="Depression causing heavy rainfall over Odisha, Jharkhand and Gangetic West Bengal. "
                    "Track misplacement by ~80 km caused localized flood bust.",
    ),
    # ── Heat Waves ──────────────────────────────────
    WeatherEvent(
        name="North India Heat Wave May 2022",
        event_type="heat_wave",
        start_date=date(2022, 5, 12),
        end_date=date(2022, 5, 21),
        lat_center=30.0,
        lon_center=76.0,
        severity=4,
        basin="Northwest India / Delhi NCR",
        description="Unprecedented April–May 2022 heat wave. Temperatures exceeding 47°C in Rajasthan. "
                    "NWP models underestimated Tmax by 2–4°C in D4–D7 forecasts due to soil moisture error.",
    ),
    WeatherEvent(
        name="Rajasthan–Maharashtra Heat Wave 2019",
        event_type="heat_wave",
        start_date=date(2019, 5, 28),
        end_date=date(2019, 6, 4),
        lat_center=25.0,
        lon_center=74.0,
        severity=4,
        basin="Central & Northwest India",
        description="Severe heat wave over Rajasthan and north Maharashtra with Tmax > 48°C. "
                    "Land-atmosphere coupling deficiency in GFS/NCUM caused systematic warm bias underestimation.",
    ),
    # ── Western Disturbances ─────────────────────────
    WeatherEvent(
        name="Delhi Heavy Snowfall WD January 2023",
        event_type="western_disturbance",
        start_date=date(2023, 1, 6),
        end_date=date(2023, 1, 10),
        lat_center=33.0,
        lon_center=76.0,
        severity=3,
        basin="Northwest Himalaya / J&K / HP",
        description="Intense Western Disturbance causing heavy snowfall J&K, HP and disrupting Delhi flights. "
                    "Models underestimated precipitation intensity due to Himalayan orographic interaction.",
    ),
    # ── Active/Break Monsoon ─────────────────────────
    WeatherEvent(
        name="Monsoon Break Spell August 2021",
        event_type="active_break",
        start_date=date(2021, 8, 3),
        end_date=date(2021, 8, 10),
        lat_center=25.0,
        lon_center=80.0,
        severity=3,
        basin="Central India / Northeast India",
        description="Prolonged monsoon break over central India with surplus rainfall in Assam & Bihar. "
                    "Intraseasonal oscillation (BSISO) transition poorly captured by NCUM D5+.",
    ),
]


def get_events_by_type(event_type: str) -> list[WeatherEvent]:
    """Filter events by type."""
    return [e for e in EVENT_CATALOG if e.event_type == event_type]


def get_events_in_daterange(start: date, end: date) -> list[WeatherEvent]:
    """Return events overlapping with the given date range."""
    return [
        e
        for e in EVENT_CATALOG
        if e.start_date <= end and e.end_date >= start
    ]


def get_showcase_events() -> list[WeatherEvent]:
    """Return the 5 events used for the SIH demo dashboard."""
    showcase_names = [
        "Cyclone Amphan",
        "Kerala Floods Monsoon Depression",
        "North India Heat Wave May 2022",
        "Cyclone Fani",
        "Delhi Heavy Snowfall WD January 2023",
    ]
    return [e for e in EVENT_CATALOG if e.name in showcase_names]


def export_catalog_json(output_path: str) -> None:
    """Export full catalog to JSON for dashboard use."""
    data = [e.to_dict() for e in EVENT_CATALOG]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Catalog] Exported {len(data)} events to {output_path}")


if __name__ == "__main__":
    export_catalog_json("dashboard/precomputed/event_catalog.json")
    print("Showcase events:")
    for e in get_showcase_events():
        print(f"  {e.name} ({e.event_type}) {e.start_date} → {e.end_date} | Severity: {e.severity}")
