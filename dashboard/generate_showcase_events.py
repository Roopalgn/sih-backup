"""
generate_showcase_events.py
---------------------------
Generate real showcase events by calling the live API.

Run this AFTER:
  1. python data/preprocess.py  (builds the dataset)
  2. python model/train.py      (trains the model, saves checkpoints/best_model.pt)
  3. uvicorn api.main:app --port 8000 &  (starts the API)

This will replace the 'illustrative_only' placeholder events in events.json
with real model predictions (data_source='live_model').

Usage:
    python dashboard/generate_showcase_events.py
    python dashboard/generate_showcase_events.py --api-url http://localhost:8000
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import date

SHOWCASE_EVENTS = [
    # Events within training/val/test window (Jun–Sep 2021 + Jun–Sep 2022)
    {
        "event_name": "Monsoon Onset 2021",
        "request_date": "2021-06-15",
        "lead_day": 5,
        "variable": "rainfall",
        "description": "Monsoon onset phase over Konkan & Goa / Odisha — first rains of 2021 season.",
        "event_type": "active_break",
    },
    {
        "event_name": "Peak Monsoon 2021",
        "request_date": "2021-08-10",
        "lead_day": 5,
        "variable": "rainfall",
        "description": "Active monsoon phase — peak precipitation over Gangetic WB and Odisha.",
        "event_type": "monsoon_depression",
    },
    {
        "event_name": "Monsoon Break 2021",
        "request_date": "2021-09-15",
        "lead_day": 7,
        "variable": "rainfall",
        "description": "Break phase — suppressed rainfall over peninsula, active over Himalayas.",
        "event_type": "active_break",
    },
    {
        "event_name": "Monsoon Onset 2022",
        "request_date": "2022-06-25",
        "lead_day": 3,
        "variable": "rainfall",
        "description": "Monsoon 2022 onset over Konkan coast — validation-window event.",
        "event_type": "active_break",
    },
    {
        "event_name": "Peak Monsoon 2022",
        "request_date": "2022-08-15",
        "lead_day": 5,
        "variable": "rainfall",
        "description": "Peak monsoon 2022 — validation period.",
        "event_type": "monsoon_depression",
    },
]

# Permanently illustrative events (pre-GEFSv12 — cannot be run through real pipeline)
ILLUSTRATIVE_EVENTS = [
    {
        "event_name": "Cyclone Amphan OOD Test",
        "request_date": "2020-05-16",
        "lead_day": 4,
        "variable": "rainfall",
        "description": (
            "Cyclone Amphan (May 2020). Pre-GEFSv12 era — no 0.25° GEFS data available. "
            "Permanently illustrative_only. Shown as out-of-distribution context."
        ),
        "event_type": "cyclone",
        "data_source": "illustrative_only",
        "data_source_reason": "Pre-GEFSv12 (before 2020-09-01). Cannot be run through real pipeline.",
        "mean_bust_probability": None,
        "mean_confidence": None,
        "top_drivers": [],
        "high_bust_regions": [],
        "grid_latitudes": [],
        "grid_longitudes": [],
        "bust_probability_map": [],
        "error_magnitude_map": [],
        "confidence_map": [],
        "from_cache": True,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Generate showcase events via live API")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", default="dashboard/precomputed/events.json")
    args = parser.parse_args()

    try:
        import requests
    except ImportError:
        print("ERROR: pip install requests")
        sys.exit(1)

    api_url    = args.api_url.rstrip("/")
    output     = Path(args.output)
    precomp    = output.parent

    # Check API is up
    try:
        r = requests.get(f"{api_url}/", timeout=5)
        r.raise_for_status()
        print(f"[OK] API at {api_url} is reachable")
    except Exception as e:
        print(f"ERROR: Cannot reach API at {api_url}: {e}")
        print("  Start it with: uvicorn api.main:app --port 8000")
        sys.exit(1)

    events = list(ILLUSTRATIVE_EVENTS)  # Start with permanently illustrative ones
    success, skipped = 0, 0

    for ev in SHOWCASE_EVENTS:
        date_str = ev["request_date"]
        lead_day = ev["lead_day"]
        name     = ev["event_name"]
        print(f"\n[→] {name} ({date_str}, Day {lead_day})...")

        try:
            r = requests.post(
                f"{api_url}/api/v1/predict",
                json={"date": date_str, "lead_day": lead_day, "variable": ev["variable"]},
                timeout=120,
            )

            if r.status_code == 503:
                print(f"  [503] No data for {date_str} Day {lead_day}. "
                      f"Marking illustrative_only.")
                ev_out = {**ev,
                          "data_source": "illustrative_only",
                          "data_source_reason": "No preprocessed forecast data for this date — run python data/preprocess.py",
                          "mean_bust_probability": None,
                          "mean_confidence": None,
                          "top_drivers": [], "high_bust_regions": [],
                          "grid_latitudes": [], "grid_longitudes": [],
                          "bust_probability_map": [], "error_magnitude_map": [],
                          "confidence_map": [], "from_cache": False}
                events.append(ev_out)
                skipped += 1
                continue

            r.raise_for_status()
            data = r.json()
            data["event_name"]  = ev["event_name"]
            data["description"] = ev["description"]
            data["event_type"]  = ev["event_type"]

            # Save individual prediction file
            pred_file = precomp / f"prediction_{date_str}_day{lead_day:02d}.json"
            with open(pred_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  [OK] data_source={data.get('data_source')} | "
                  f"bust={data.get('mean_bust_probability'):.3f} | "
                  f"conf={data.get('mean_confidence'):.1f}%")
            events.append(data)
            success += 1

        except Exception as e:
            print(f"  [ERROR] {e}")
            skipped += 1

    # Write master events.json
    with open(output, "w") as f:
        json.dump(events, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Done. {success} real events + {skipped} illustrative + {len(ILLUSTRATIVE_EVENTS)} permanent illustrative")
    print(f"Saved to {output}")
    if success == 0:
        print("\nWARNING: No real model events generated.")
        print("  Run the full pipeline first:")
        print("    python data/preprocess.py")
        print("    python model/train.py")


if __name__ == "__main__":
    main()
