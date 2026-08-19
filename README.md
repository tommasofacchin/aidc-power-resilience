# AIDC Power Supply Resilience — Huawei Tech Arena 2026, Topic 2 (Phase 1)

County-level forecasting of the power outage ratio `x = customers_out / total_customers`
across the United States, driven by archived ECMWF IFS HRES forecasts.

- **Task A** — day-ahead: 48-hour horizon, 1-hour resolution
- **Task B** — hours-ahead: 6-hour horizon, 15-minute resolution

Test window: **1 September – 30 November 2025 (UTC)**.

> **Status: work in progress.** This README is the reproducibility guide required by the
> submission guidelines. Sections marked _TODO_ are filled in as the pipeline is built.
> The internal working plan lives in [PLAN.md](PLAN.md).

---

## 1. Environment configuration

_TODO — pin versions with `pip freeze` before submission._

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r code/requirements.txt
```

Python version: _TODO_
Hardware requirements: _TODO_ (expected: CPU-only, no GPU needed)

## 2. Data acquisition

No raw data is versioned in this repository — everything is fetched by the scripts in
[code/data_acquisition/](code/data_acquisition/) and cached under `data/` (git-ignored).

| Source | Purpose | Script |
|---|---|---|
| EAGLE-I Power Outage Data (ORNL), 2014–2025 | Ground truth | _TODO_ |
| Open-Meteo Single Runs API — ECMWF IFS HRES | Forecast weather features | _TODO_ |

See [PLAN.md §7](PLAN.md) for the data source and licensing declaration.

## 3. Execution order

_TODO_

```
1. code/data_acquisition/...     # download EAGLE-I + IFS runs
2. code/preprocessing/...        # 15-min target grid, zero-fill, x computation
3. code/features/...             # feature construction
4. code/train.py                 # model training
5. code/predict.py               # generate submission/predictions.csv
```

## 4. Expected runtime

_TODO — per-step estimates so reviewers can plan reproduction time._

## 5. Repository layout

```
├── code/
│   ├── data_acquisition/   # EAGLE-I + Open-Meteo download, with on-disk caching
│   ├── preprocessing/      # target grid construction, timestamp alignment
│   ├── features/           # feature engineering
│   ├── train.py
│   ├── predict.py
│   └── requirements.txt
├── data/                   # git-ignored: raw downloads and caches
├── docs/                   # competition rules and reference material
├── submission/             # deliverables: predictions.csv, report.pdf
└── PLAN.md                 # internal working plan
```

## 6. Reproducibility notes

**Causality.** Predictions for a given `issue_time` use only information available at that
time. Weather input comes from the archived ECMWF IFS HRES run that would genuinely have
been disseminated by `issue_time` — never observed weather at `target_time`. See
[PLAN.md §2.2](PLAN.md) for the exact `issue_time → run` mapping and its rationale.
