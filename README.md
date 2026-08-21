# AIDC Power Supply Resilience — Huawei Tech Arena 2026, Topic 2 (Phase 1)

County-level forecasting of the power outage ratio `x = customers_out / total_customers`
across the United States, driven by archived ECMWF IFS HRES forecasts.

- **Task A** — day-ahead: 48-hour horizon, 1-hour resolution
- **Task B** — hours-ahead: 6-hour horizon, 15-minute resolution

Test window: **1 September – 30 November 2025 (UTC)**.

Deliverables: [submission/predictions.csv](submission/predictions.csv) and
[submission/report.pdf](submission/report.pdf). This README is the reproducibility guide
the submission guidelines ask for; the methodology and results live in the report, and the
internal working plan in [PLAN.md](PLAN.md).

---

## 1. Environment configuration

Reference environment: **Python 3.14.6**, Windows 11, **CPU only** — no GPU is used
anywhere, and no step needs more than ~8 GB of RAM. Peak memory is in EAGLE-I parsing
(step 3 below), not in training.

```bash
python -m venv .venv
.venv/Scripts/activate           # Linux/macOS: source .venv/bin/activate
pip install -r code/requirements.txt
```

`code/requirements.txt` pins every direct dependency to the exact version the submitted
results were produced with, and lists the resolved transitive versions in a comment.

**Disk:** ~3.5 GB under `data/` — 2.9 GB of EAGLE-I annual CSVs, ~130 MB of HTTP cache,
~250 MB of weather parquet. Nothing under `data/` is versioned.

**Windows note.** If the checkout path contains non-ASCII characters, set `PYTHONUTF8=1`
before running anything; without it the default cp1252 console encoding raises
`UnicodeEncodeError` on the first non-ASCII county name.

```bash
export PYTHONUTF8=1              # PowerShell: $env:PYTHONUTF8 = "1"
```

## 2. Data acquisition

No raw data is versioned here — every script below downloads what it needs and caches it
under `data/` (git-ignored), so re-running a step is cheap and safe.

| Source | Purpose | Script |
|---|---|---|
| EAGLE-I Recorded Electricity Outages 2014–2025 (ORNL, figshare article 24237376 v4) | Ground truth, plus `MCC.csv` for the denominator | [code/data_acquisition/eagle_i.py](code/data_acquisition/eagle_i.py) |
| Open-Meteo Single Runs API (archived ECMWF IFS HRES initializations) | Forecast weather features | [code/data_acquisition/open_meteo.py](code/data_acquisition/open_meteo.py), driven by [bulk_download_training_weather.py](code/data_acquisition/bulk_download_training_weather.py) |
| US Census Bureau 2025 County Gazetteer | County centroids used as forecast coordinates | [code/data_acquisition/county_coordinates.py](code/data_acquisition/county_coordinates.py) |

Full licence declaration: **section 9 of the report**, verified at each source rather than
assumed.

**The one thing that will bite a reproducer: Open-Meteo's rate limits.** The Single Runs
API enforces three separate, undocumented quotas — per minute, per hour, and **per day**
(observed at roughly 110–130 calls/day). The daily one cannot be waited out inside a
single sitting, so a full weather download spans **several calendar days**. Every call is
cached permanently in `data/raw/open_meteo_cache.sqlite` and every completed run is
written to its own parquet, so the downloader is resumable: kill it and re-run it, and it
picks up exactly where it stopped. It also absorbs all three quota tiers by itself
(minute and hour by sleeping, day by polling every 4 h), so it is safe to start once and
leave running unattended.

## 3. Execution order

Steps 1–6 are the data pipeline, 7–9 the model, 10–11 the deliverables. Every step writes
to `data/processed/` and is idempotent — re-running one does not invalidate the others.

```bash
export PYTHONUTF8=1

# 1. Ground truth: EAGLE-I annual files (2024 + 2025) and MCC.csv, from figshare
python code/data_acquisition/eagle_i.py

# 2. County centroids, from the Census Gazetteer
python code/data_acquisition/county_coordinates.py

# 3. The five reporting counties  -> data/processed/selected_counties.csv
python code/preprocessing/select_counties.py

# 4. The 102-county stratified training sample -> data/processed/training_counties.csv
python code/preprocessing/select_training_sample.py

# 5. Reconciled per-county denominator -> data/processed/total_customers_reconciled.csv
python code/preprocessing/reconcile_denominators.py

# 6. Weather. The long pole - see the rate-limit note in section 2.
#    --budget caps how many NEW runs one pass may fetch, so training downloads cannot
#    eat the quota that step 10 needs.
python code/data_acquisition/bulk_download_training_weather.py --start 2025-01-01 --end 2025-08-31 --budget 80
python code/data_acquisition/bulk_download_training_weather.py --start 2024-09-01 --end 2024-11-30 --budget 80

# 7. Feature table + climatology -> data/processed/training_table_partial.parquet
python code/features/build_training_table.py --years 2024 2025

# 8. Train -> data/processed/model_bundle/
python code/train.py

# 9. Fit the lead-dependent persistence blend -> model_bundle/blend_weights.json
#    Defaults to --season autumn: the weights are fitted on a seasonally matched
#    out-of-sample window, which is worth ~9 points of RMSE skill over fitting them on
#    the forward split. See the module docstring and report section 5.1.
python code/blend.py --season autumn

# 10. Generate the submission -> submission/predictions.csv
python code/predict.py

# 11. Format check against the guidelines checklist
python code/validate_submission.py submission/predictions.csv
```

Report artefacts, if the numbers in the report need regenerating:

```bash
python code/ablation.py                          # -> data/processed/ablation_results.{csv,md}
python code/seasonal_holdout.py                  # -> data/processed/seasonal_holdout.{csv,md}
python code/report_figures.py --season autumn    # -> report/figures/skill_by_lead_autumn.png
python code/report_figures.py --season reference # -> report/figures/skill_by_lead.png
python code/report_tables.py                     # -> data/processed/report_county_profile.csv
```

The report embeds the autumn figure; the reference-season one is kept because the two
together are what justify fitting the blend where we fit it.

`report/report.pdf` is `report/report.html` printed to PDF by a headless Chromium — no
manual step, and the output is byte-stable across runs:

```bash
chrome --headless=new --disable-gpu --no-pdf-header-footer \
       --print-to-pdf=report/report.pdf \
       "file:///ABSOLUTE/PATH/TO/report/report.html"
# Windows: "C:\Program Files\Google\Chrome\Application\chrome.exe" (msedge.exe also works)
```

The HTML is hand-maintained prose, but every figure and every quoted number in it comes
from the three scripts above, so a figure cannot silently disagree with the table beside
it. The result is 8 pages, against the guidelines' 3–8 limit.

## 4. Expected runtime

Wall clock on the reference machine (16 logical cores, 15 GB RAM), single run, warm cache where the
step has one. Steps 7–11 were timed on 21 August 2026; steps 1–5 are from the runs that
originally produced their outputs, so treat them as approximate.

| Step | Cold | Warm (cached) | Note |
|---|---|---|---|
| 1. EAGLE-I download | ~25 min | instant | 2.9 GB over HTTPS, bandwidth-bound |
| 2. Gazetteer | ~5 s | instant | 140 KB |
| 3. Select 5 counties | ~4 min | ~4 min | full scan of the 2025 annual file |
| 4. Training sample | ~4 min | ~4 min | same scan |
| 5. Denominator reconciliation | ~5 min | ~5 min | reads the 2024 file for its in-file `total_customers` |
| 6. Weather download | **days** | instant | quota-bound, not bandwidth-bound — see section 2 |
| 7. Build training table | ~1.5 min | — | 184 runs × 102 counties → 1.33 M rows |
| 8. Train | ~30 s | — | LightGBM, CPU, 1.2 M training rows |
| 9. Blend fit | ~5 s | — | |
| 10. Generate submission | ~4 min | ~4 min | 93 IFS runs, all cache hits after the first pass |
| 11. Validate | ~3 s | — | |

**End to end from a clean checkout: about an hour of compute, plus however many days the
weather quota takes.** A reproducer who only wants to re-derive the model from an existing
`data/` directory needs steps 7–11, about 6 minutes.

## 5. Repository layout

```
├── code/
│   ├── data_acquisition/   # EAGLE-I + Open-Meteo + Gazetteer download, on-disk caching
│   ├── preprocessing/      # county selection, target grid, denominator reconciliation
│   ├── features/           # feature engineering, causality assertions
│   ├── train.py            # LightGBM Tweedie training -> model bundle
│   ├── blend.py            # lead-dependent model/persistence blend
│   ├── predict.py          # submission generation
│   ├── ablation.py         # feature-group and baseline comparison
│   ├── seasonal_holdout.py # seasonal-transfer probe: autumn vs the forward split
│   ├── validate_submission.py
│   └── requirements.txt
├── data/                   # git-ignored: raw downloads, caches, processed tables
├── docs/                   # competition rules and reference material
├── report/                 # report source (HTML), figures, built PDF
├── submission/             # deliverables: predictions.csv, report.pdf
└── PLAN.md                 # internal working plan
```

## 6. Reproducibility notes

**Causality.** Predictions for a given `issue_time` use only information available at that
time. Weather input comes from the archived ECMWF IFS HRES run that would genuinely have
been disseminated by `issue_time` — never observed weather at `target_time`. The mapping
is asserted at runtime by [code/features/causality.py](code/features/causality.py), not
just respected by construction. See [PLAN.md section 2.2](PLAN.md) for the mapping and its
rationale.

**Determinism.** The train/validation split is temporal (a fixed date, not a random
shuffle) and LightGBM runs with a fixed seed, so repeated runs on the same table reproduce
the same model. The pipeline has no other stochastic step.

**The model bundle is a bundle on purpose.** `data/processed/model_bundle/` stores the
booster *together with* the `fips_code` category ordering and the fitted climatology
thresholds. Prediction-time code that re-derives those from its own data silently produces
different answers — see [code/model_bundle.py](code/model_bundle.py) for the two concrete
bugs that motivated this.

**Known environment quirk.** LightGBM's own `Booster.save_model()` fails against a
OneDrive-synced path (`LightGBMError: ... is not available for writes`) even where plain
Python file I/O succeeds; the bundle therefore serializes via `model_to_string()`. Nothing
to do on a normal filesystem.
