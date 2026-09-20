"""
download_gefs_stub.py
---------------------
Minimal 5-date GEFS download for pipeline validation.

Use this before committing to the full 2021–2022 season download.
If this works end-to-end (download → preprocess → train 1 epoch), you know
your pipeline is structurally correct.

Dates selected to cover different monsoon phases:
  2021-06-15 — Monsoon onset phase
  2021-08-10 — Peak monsoon / active phase
  2021-09-15 — Monsoon break/withdrawal
  2022-06-25 — Next year onset
  2022-08-15 — Peak monsoon 2022 (will be used for validation)

Run:
    python data/download_gefs_stub.py
    python data/preprocess.py --stub

This should take ~5–15 minutes and produce ~20–40 MB of data.
"""

import sys
import datetime
from pathlib import Path

sys.path.insert(0, ".")
from data.download_gefs import fetch_one_day
from data.config_loader import load_config

# 5 real dates spanning two monsoon seasons
STUB_DATES = [
    datetime.date(2021, 6, 15),
    datetime.date(2021, 8, 10),
    datetime.date(2021, 9, 15),
    datetime.date(2022, 6, 25),
    datetime.date(2022, 8, 15),
]


def main():
    cfg = load_config("config/settings.yaml")
    domain    = cfg["domain"]
    lead_days = cfg["data"]["lead_days"]
    lead_hours = [d * 24 for d in lead_days]
    output_dir = Path(cfg["paths"]["raw_data"]) / "gefs"

    print(f"[STUB] Downloading {len(STUB_DATES)} dates for pipeline validation")
    print(f"[STUB] Lead hours: {lead_hours}")
    print(f"[STUB] Output: {output_dir}\n")

    total = 0
    for date in STUB_DATES:
        print(f"\n[STUB] === {date.isoformat()} ===")
        n = fetch_one_day(date, lead_hours, output_dir, domain)
        print(f"[STUB]   → {n}/{len(lead_hours)} files saved")
        total += n

    print(f"\n[STUB] Done. {total} files saved to {output_dir}/")
    print("[STUB] Next step: python data/preprocess.py --stub")


if __name__ == "__main__":
    main()
