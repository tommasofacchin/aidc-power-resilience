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

No large raw data is versioned here — the EAGLE-I and Open-Meteo scripts below download
what they need and cache it under `data/` (git-ignored), so re-running a step is cheap and
safe. **One exception, and it is not automated:** `county_coordinates.py` only *reads* the
Census gazetteer, it does not fetch it. The submission package therefore ships that file
(see section 2.1); working from a bare clone instead, download it by hand:

```bash
curl -o data/raw/gaz_counties.zip   https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2025_Gazetteer/2025_Gaz_counties_national.zip
unzip -o data/raw/gaz_counties.zip -d data/raw/    # -> 2025_Gaz_counties_national.txt
```

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

## 2.1 Reproducing `predictions.csv` from the submitted package

**Inference entry point:** [code/predict.py](code/predict.py). **Exact command:**

```bash
python code/predict.py            # -> submission/predictions.csv
```

It takes no arguments: the issue-time schedule, the horizons and the five reporting
counties are all fixed by the submitted inputs, so there is nothing to configure and
nothing to get wrong. Do not retrain — `predict.py` loads the supplied model as-is.

**Shipped in the package** (everything inference needs that the submitted code cannot
honestly recreate):

| File | Role at inference |
|---|---|
| `data/processed/model_bundle/model.txt` | The exact trained LightGBM booster behind the submitted `predictions.csv` |
| `data/processed/model_bundle/fips_categories.json` | Fitted county encoder — category order is part of the model |
| `data/processed/model_bundle/climatology.parquet` | Offline-derived per-county climatology feature table |
| `data/processed/model_bundle/blend_weights.json` | Fitted persistence/model blend weights, per lead-time bucket |
| `data/processed/model_bundle/target_kind.json` | Records that the target is a residual, not a level — `predict.py` reconstructs accordingly |
| `data/processed/selected_counties.csv` | The five reporting counties. Pinned because `select_counties.py` re-derives a *decision* from data, and a reviewer re-running it could legitimately land on a different five |
| `data/processed/total_customers_reconciled.csv` | The reconciled training denominator. Regenerating it needs the 1.4 GB 2024 EAGLE-I file for one 64 KB output |
| `data/raw/MCC.csv` | The **grading** denominator, as published. Pinned so a future figshare version cannot silently change what the ratios are divided by |
| `data/raw/2025_Gaz_counties_national.txt` | County centroids used as forecast coordinates. Pinned because no submitted script downloads it |
| `data/processed/training_counties.csv` | The 102 counties the archived forecast set was downloaded for (training only, see section 6) |

`make_submission_package.py` enforces this list against the code rather than against
memory: it reads the data-path literals back out of the inference modules and refuses to
build if one is neither shipped nor recorded as recreatable.

**Not shipped, and how to obtain it** — the two large public datasets, with the exact
retrieval parameters used:

| Input | Retrieval |
|---|---|
| EAGLE-I outages, `eaglei_outages_2025.csv` (1.4 GB) | `python code/data_acquisition/eagle_i.py`. figshare article **24237376 v4**, file id **62164877**, anonymous HTTPS GET, no login. `predict.py` reads 2025 only, for the observed state at each issue time |
| Archived ECMWF IFS HRES forecasts (262 MB cache) | `code/data_acquisition/open_meteo.py`. Endpoint **`https://single-runs-api.open-meteo.com/v1/forecast`**, **`models=ecmwf_ifs`** (IFS HRES, 9 km). Variables: `temperature_2m`, `dew_point_2m`, `surface_pressure`, `wind_speed_10m`, `wind_direction_10m`, `wind_gusts_10m`, `precipitation`, `rain`, `snowfall`, `cape`, `cloud_cover`, `freezing_level_height`. Runs: 00Z and 12Z initializations, archive from 2024-03-14. **Issue-time restriction:** both tasks use the **(D-1) 12:00Z** run for every issue time on day D — Task A at +12h..+60h from run time, Task B at +12h15m..+36h. That is a run genuinely disseminated before `issue_time` (IFS lag is ~5-7 h), never a later one and never observed weather. Task B could legitimately use a fresher run and deliberately does not: one run per day costs 93 API calls for the window against 367, and the daily rate limit is what decides whether the submission can be generated at all. See the `predict.py` docstring |

`predict.py` fetches the forecast runs it needs on its own; with a cold cache expect the
Open-Meteo rate limits described above, so allow several calendar days rather than one
sitting.

## 3. Execution order

**Note: The trained model is included in the submission package at `data/processed/model_bundle/`. Do not retrain. Use the supplied model as-is to reproduce `predictions.csv`.**

Steps 1–6 are the data pipeline, 7–9 the model, 10–11 the deliverables. Every step writes
to `data/processed/` and is idempotent — re-running one does not invalidate the others. To
reproduce predictions without retraining, skip steps 1–9 and run steps 10–11 directly.

```bash
export PYTHONUTF8=1

# 1. Ground truth: EAGLE-I annual files (2024 + 2025) and MCC.csv, from figshare
python code/data_acquisition/eagle_i.py

# 2. County centroids, from the Census Gazetteer
python code/data_acquisition/county_coordinates.py

# 3. Reconciled per-county denominator -> data/processed/total_customers_reconciled.csv
#    Must run BEFORE step 4: county selection divides by this denominator, and raw
#    MCC.csv is wrong by up to 21x for some counties. Step 4 refuses to fall back to it.
python code/preprocessing/reconcile_denominators.py

# 4. The five reporting counties  -> data/processed/selected_counties.csv
#    Also writes county_climatology_jan_aug_2025.parquet, which step 5 consumes.
python code/preprocessing/select_counties.py

# 5. The 102-county stratified training sample -> data/processed/training_counties.csv
#    This file is VERSIONED and ships inside the submission package at that same path,
#    and this step keeps it rather than re-deriving it: the archived forecast set under
#    data/raw was downloaded for exactly these counties.
#    --refresh re-selects, and then step 6 has to fetch runs for whatever it adds.
python code/preprocessing/select_training_sample.py

# 6. Weather. The long pole - see the rate-limit note in section 2.
#    --budget caps how many NEW runs one pass may fetch, so training downloads cannot
#    eat the quota that step 10 needs.
python code/data_acquisition/bulk_download_training_weather.py --start 2025-01-01 --end 2025-08-31 --budget 80
python code/data_acquisition/bulk_download_training_weather.py --start 2024-09-01 --end 2024-11-30 --budget 80

# 7. Feature table + climatology -> data/processed/training_table_partial.parquet
#    Fails if a selected training county has no forecast runs on disk, rather than
#    dropping it silently. --allow-missing-counties accepts a partial table on purpose,
#    which is what you want while a weather download is still filling in.
python code/features/build_training_table.py --years 2024 2025

# 8. Train -> data/processed/model_bundle/
#    The model predicts the RESIDUAL FROM PERSISTENCE (target_x - x_at_issue), not the
#    level; the bundle records that in target_kind.json and predict.py reconstructs
#    through model_bundle.to_level(). See train.py's docstring for the measurement that
#    motivated it (the Tweedie head on the level beat always-zero by under 4%).
python code/train.py

# 9. Fit the lead-dependent persistence blend -> model_bundle/blend_weights.json
#    Defaults to --season autumn: the weights are fitted on a seasonally matched
#    out-of-sample window, which is worth ~9 points of RMSE skill over fitting them on
#    the forward split. See the module docstring and report section 5.1.
python code/blend.py --season autumn

# 10. Generate the submission -> submission/predictions.csv
#     predicted_x leaves this step in the GRADING denominator (MCC.csv as published),
#     not the reconciled one everything upstream divides by. The organisers confirmed on
#     24 Aug 2026 that they score against MCC and drop the timestamps it makes
#     impossible; to_grading_units() converts, one constant per county, in one place.
python code/predict.py

# 11. Format check against the guidelines checklist
python code/validate_submission.py submission/predictions.csv

# 12. Assemble what actually gets handed in -> dist/submission/ + dist/submission.zip
#     Re-runs the validation above, checks the report's page count, and refuses to build
#     if submission/report.pdf has drifted from report/report.pdf or if predictions.csv
#     is older than the model bundle. dist/ is git-ignored and rebuilt from scratch.
python code/make_submission_package.py
```

Report artefacts, if the numbers in the report need regenerating:

```bash
python code/ablation.py                          # -> data/processed/ablation_results.{csv,md}
python code/seasonal_holdout.py                  # -> data/processed/seasonal_holdout.{csv,md}
python code/report_figures.py --season autumn    # -> report/figures/skill_by_lead_autumn.png
python code/report_figures.py --season reference # -> report/figures/skill_by_lead.png
python code/report_tables.py                     # -> data/processed/report_county_profile.csv
python code/report_numbers.py                    # -> data/processed/report_numbers.md (run last)
```

The report embeds the autumn figure; the reference-season one is kept because the two
together are what justify fitting the blend where we fit it.

`report/report.pdf` is `report/report.html` printed to PDF by a headless Chromium — no
manual step. The rendering is deterministic, but the file is not reproducible byte for
byte: Skia stamps `/CreationDate` and `/ModDate` into every print, so two runs over
identical HTML differ. That is why the PDF is absent from the MD5 table below, and why
step 12 compares `submission/report.pdf` against `report/report.pdf` rather than against
a recorded hash.

```bash
chrome --headless=new --disable-gpu --no-pdf-header-footer \
       --print-to-pdf=report/report.pdf \
       "file:///ABSOLUTE/PATH/TO/report/report.html"
# Windows: "C:\Program Files\Google\Chrome\Application\chrome.exe" (msedge.exe also works)
```

The HTML is hand-maintained prose, but every figure and every quoted number in it comes from
the scripts above. `report_numbers.py` is the one to run last and read against the report: it
recomputes every figure the prose quotes, including the handful — Table 4's cross-season blend
comparison, the split-gain shares, the effective persistence weight over the submitted rows —
that no other script owns and that therefore go stale silently when the training table grows.
The result is 8 pages, against the guidelines' 3–8 limit.

## 4. Expected runtime

Wall clock on the reference machine (16 logical cores, 15 GB RAM), single run. The
*clean clone* column is what a reviewer actually faces and is the one to plan against: it
was measured end to end on 24 August 2026 in the reproduction test described below — fresh
checkout, fresh virtualenv, `data/raw` copied in, nothing else warm. The *repeat* column is
the same step run again in a working directory whose OS file cache is already hot, which is
the situation while developing. Steps 1, 2 and 6 are downloads and are timed from the runs
that originally produced their outputs, so treat those as approximate.

| Step | Clean clone | Repeat (warm) | Note |
|---|---|---|---|
| 0. `venv` + `pip install` | ~2.5 min | — | 8 direct dependencies, wheels only |
| 1. EAGLE-I download | ~25 min | instant | 2.9 GB over HTTPS, bandwidth-bound |
| 2. Gazetteer | ~5 s | instant | 140 KB |
| 3. Denominator reconciliation | ~4.5 min | ~10 s | scans 2024 + 2025 once, then caches both scans |
| 4. Select 5 counties | ~1.7 min | ~45 s | full scan of the 2025 annual file |
| 5. Training sample | ~1 s | ~1 s | keeps the versioned sample; `--refresh` re-selects |
| 6. Weather download | **days** | instant | quota-bound, not bandwidth-bound — see section 2 |
| 7. Build training table | ~4 min | ~1.8 min | 388 runs × 102 counties → 2.80 M rows |
| 8. Train | ~1 min | ~27 s | LightGBM, CPU, 2.30 M training rows |
| 9. Blend fit | ~1.3 min | ~36 s | weight × decay half-life grid, per lead bucket |
| 10. Generate submission | ~25 min | ~26 min | 93 IFS runs, every one a cache hit; Task A evaluates four quarter-hours per row (see `predict.py`), so the loop does ~2x the work of a per-hour pass |
| 11. Validate | ~1 s | ~1 s | |
| 12. Build the package | ~2 s | ~2 s | copies code/ and zips |

**End to end from a clean checkout: about 38 minutes of compute for steps 3–12, plus ~2.5
minutes to build the environment, the one-off downloads, and however many days the weather
quota takes.** A reproducer who only wants to re-derive the model from an existing `data/`
directory needs steps 7–11, about 31 minutes cold.

Take those as an upper bound rather than a stopwatch. The clean-clone run certified below
did steps 3–12 in **17 minutes** on the same machine — step 3 in 2.0, step 4 in 0.7, step 7
in 1.9 and step 10 in 11.3 — because the archive it read had just been copied and was still
in the OS file cache, and because it ran on a local disk rather than a OneDrive-synced one.
Every step here is I/O-bound or cache-bound, so the spread between two runs of the same
commit is a factor of two.

**This was tested, not asserted.** On 24 August 2026, at commit `f79125d`, the repository was
cloned into an empty directory outside the working tree, a fresh virtualenv was built from
`code/requirements.txt`, `data/raw` was supplied from the existing archive (steps 1, 2 and 6
are downloads, and step 6 is quota-bound over days), and steps 3–12 were run in the order
above. Every artefact came back **byte-identical** to the committed one.

The model was reformulated on 27 August 2026 (step 8 above). That changes what steps 8–10
produce and nothing else: the four artefacts from steps 3–7 still hash exactly as the clean-clone
test certified them, which is what the first block below records. The three model artefacts are
new, and the second block is what steps 8–10 produce on this machine — verified to be
reproducible by re-running those steps against the same table and comparing, but **not** re-run
from a fresh clone since the reformulation.

Unchanged since the 24 August clean-clone test:

| Artefact | MD5 |
|---|---|
| `data/processed/total_customers_reconciled.csv` | `534f2888c9bb03dfd206ec90d30937bb` |
| `data/processed/selected_counties.csv` | `c406c65cd6e685641ab2567c2317e1b8` |
| `data/processed/training_table_partial.parquet` | `f3118a0926e282d18a4432e4ec9056fd` |

Produced by the reformulated model, 27 August 2026:

| Artefact | MD5 |
|---|---|
| `submission/predictions.csv` | `6ba64b0cccc32a10536ff6518394a576` |
| `data/processed/model_bundle/model.txt` | `e763400088e8cd3341809f824195164c` |
| `data/processed/model_bundle/blend_weights.json` | `3b91aaff07566a1486bff0fa5f4a6e36` |

`data/processed/training_counties.csv` also matches (`c7aff04c903d502023e01dccdb346573`), but
it is committed rather than derived (step 5 keeps the versioned sample on purpose, because the
archived forecast runs were fetched for exactly those 102 counties), so it is evidence that the
file travels with the repository, not that the pipeline reproduces it. Step 12 in the clone
built the package with that file and `baseline_runs_20260822.json` inside it, which is what
makes the same reasoning hold for a reviewer who has only the zip.

Step 10 reported *"All runs cached — the prediction loop below makes no further API calls"*
for all 93 runs, so a reviewer reproducing this spends no Open-Meteo quota. An earlier run of
the same test, against the pre-densification archive, is what surfaced the two defects described
under *The training sample is an input* below.

`report/report.pdf` is deliberately not in that table: it is printed by a headless browser that
stamps a timestamp into the file, so it does not reproduce byte for byte (see section 3). Its
content does — the HTML it is printed from is versioned, and every number in it comes from
`report_numbers.py`.

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
│   ├── densification_probe.py # go/no-go on buying more training runs with the quota
│   ├── report_numbers.py   # recomputes every number the report quotes
│   ├── validate_submission.py
│   ├── make_submission_package.py # assembles dist/ in the guidelines' layout
│   └── requirements.txt
├── data/                   # git-ignored except the pinned inference inputs (section 2.1)
├── docs/                   # competition rules and reference material
├── report/                 # report source (HTML), figures, built PDF
├── submission/             # built deliverables: predictions.csv, report.pdf
├── dist/                   # git-ignored: the assembled package, rebuilt on demand
└── PLAN.md                 # internal working plan
```

The guidelines ask for three components — report, predictions, and the complete codebase
with this guide — and suggest a package with `code/` and `README.md` nested inside
`submission/`. They live at the root here instead, because that is the working copy and a
second copy inside `submission/` could drift from it without anyone noticing. Step 12
above resolves the difference at the end: it derives `dist/submission/` in exactly the
suggested layout, so what is handed in matches the guidelines and what is edited stays
in one place. **Hand in `dist/submission.zip`, not the `submission/` folder** — that
folder alone is missing the codebase, which the guidelines say makes a submission
unscoreable.

The package carries two files from under `data/` as well, at their repo-relative paths:
`data/processed/training_counties.csv`, without which step 5 re-derives a different
training sample and step 7 has no weather for the counties it adds (the note in section 6
has that failure in full), and `data/processed/baseline_runs_20260822.json`, the frozen
run list `densification_probe.py` compares against. Everything else under `data/` is
downloaded or derived by the steps above.

## 6. Reproducibility notes

**Causality.** Predictions for a given `issue_time` use only information available at that
time. Weather input comes from the archived ECMWF IFS HRES run that would genuinely have
been disseminated by `issue_time` — never observed weather at `target_time`. The mapping
is asserted at runtime by [code/features/causality.py](code/features/causality.py), not
just respected by construction. See [PLAN.md section 2.2](PLAN.md) for the mapping and its
rationale.

**The training sample is an input, not an output.** `data/processed/training_counties.csv`
is versioned here and shipped inside the submission package, because the archived forecast
runs were fetched for exactly those 102 counties at a quota that makes fetching a different
sample a multi-day operation. Left unversioned, a clean checkout re-derived a *different* sample from the
climatology, and the counties it added had no weather: they merged to nothing and the
pipeline trained on 59 counties while reporting success at every step. Step 7 now treats
that mismatch as fatal, and step 5 will not overwrite the pinned file without `--refresh`.

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
