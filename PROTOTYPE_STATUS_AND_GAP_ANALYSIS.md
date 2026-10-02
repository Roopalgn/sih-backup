# SIH #26079 — Prototype Status & Gap Analysis

Deep dive as of 2026-10-02

## 🟢 What Is Fully Working Right Now

### Verification

| Check | State |
| --- | --- |
| Frontend TypeScript production build | ✅ Passed |
| Frontend lint | ✅ No errors (2 pre-existing legacy `Header` warnings) |
| Review-region regression test | ✅ Passed |
| Event catalog API regression test | ✅ Passed |
| Vite → FastAPI proxy smoke test | ✅ Passed — 6 events and 3 review areas returned |
| Unsupported-date behaviour | ✅ HTTP 503 — no synthetic fallback |

> The local Python runtime used for this verification does not include `xarray`, so the full scientific pipeline suite cannot run here. This is an environment dependency gap, not a passing result to claim.

### Backend Pipeline

| Component | State |
| --- | --- |
| GEFS preprocessing, IMD observations and P90 error definition | ✅ Existing pipeline retained |
| Cached/live prediction endpoint | ✅ Working with honest provenance |
| Forecast-bust review-area detection | ✅ Fixed — connected regions detected at a 0.30 review threshold |
| Cached prediction region backfill | ✅ Fixed — older real grids no longer return an empty panel |
| `GET /api/v1/events` | ✅ Added — lightweight selector metadata, no large grids transferred |
| `/predict`, `/events`, `/info`, `/historical` | ✅ Available |
| Out-of-scope dates | ✅ HTTP 503, never fabricated |

### Frontend (React)

| Feature | State |
| --- | --- |
| Backend-driven event selector | ✅ Fixed — refreshes from `/api/v1/events` on mount |
| Forecast lead explorer | ✅ Days 3, 5, 7 and 10 remain clearly scoped |
| Lead-time chart | ✅ Exact API outputs only; unavailable leads are not interpolated |
| Spatial map and driver-attribution display | ✅ Existing model-output views retained |
| Flagged review-area panel | ✅ Now shows ranked areas from real cached grids |
| Model-skill disclosure | ✅ Added — reported validation AUC 0.46 labelled as prototype-scale |
| “How DRISHTI works” workflow | ✅ Added — forecast inputs → model score → forecaster review |
| Presentation mode, briefing export and provenance badges | ✅ Retained |

## 🟡 Real Gaps — Ranked by Demo Impact

### GAP 1 — Training data remains prototype-scale

**Impact: HIGH (model quality).** The available implementation is limited to stub-era coverage. More GEFS dates and retraining are still required before making a skill or accuracy claim.

**Demo guidance:** describe the system as a research prototype trained on two monsoon seasons; do not claim high accuracy or production readiness.

**Next fix:** download broader GEFS coverage with the existing download scripts, rebuild splits and retrain/evaluate.

### GAP 2 — Empty high-bust-region panel

**Impact: MEDIUM (visible completeness).** **Fixed.** The region algorithm now uses a documented 0.30 review threshold, four-neighbour connected components, a four-cell minimum and top-three ranking. The 2022-06-25 Day-3 cached grid returns three review areas.

### GAP 3 — Event selector could drift from backend data

**Impact: LOW–MEDIUM.** **Fixed.** The frontend fetches `/api/v1/events`; the hardcoded list remains metadata-only fallback for an offline selector and is never promoted into prediction output.

### GAP 4 — No live GEFS download for future dates

**Impact: LOW for hackathon, HIGH for production.** Still open. Future dates outside available data return HTTP 503 rather than false output.

**Next fix:** add an explicitly monitored asynchronous acquisition pipeline around `data/download_gefs.py`, then preprocess and infer only after data validation succeeds.

### GAP 5 — Reported validation AUC is 0.464

**Impact: HIGH for deployment, LOW for a transparent demo.** Still open. The UI now surfaces this as a limitation rather than hiding it.

### GAP 6 — No full skill-score dashboard

**Impact: LOW.** Partially addressed with an honest model-skill note. A calibration view and RMSE/MAE/bias breakdown remain future work because they require a larger validated evaluation set.

## Verdict: Demo Readiness

The demo is materially stronger now: the previously empty Signal Review panel contains genuine, grid-derived regions; the selector is API-backed; unavailable values are not invented; and the workflow is easier to explain visually.

## What You Can Truthfully Claim

- ✅ “We built an end-to-end AI forecast-bust detection prototype.”
- ✅ “The system exposes gridded confidence, forecast-bust and error signals with provenance.”
- ✅ “Integrated Gradients provides model-sensitivity drivers for available predictions.”
- ✅ “Unsupported dates fail with HTTP 503 instead of fabricated output.”
- ✅ “Review areas are derived from connected high-risk cells in the model-output grid.”
- ✅ “The current model is prototype-scale and its reported validation AUC is 0.46.”

Do not claim high accuracy, production readiness or validated operational skill.
