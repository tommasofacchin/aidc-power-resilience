"""
Bulk-download ECMWF IFS HRES Single Runs weather for the training county sample, across
the training window (2025-01-01 .. 2025-08-31), for the 00Z and 12Z runs each day.

Why 2/day instead of all 4 (00/06/12/18Z): this is TRAINING data, not the actual
issue_time -> run mapping used to generate submitted predictions (that mapping — see
PLAN.md section 2.2 — is applied later, only for the 5 reporting counties, when
building the real submission). For training we just need representative
(weather, lead_time, outcome) pairs across a range of lead times; the model learns
lead_time as an explicit feature, so it doesn't need every run replayed. Two runs a day
already covers short and long lead times per PLAN.md's mapping rule (task A wants
D-1 12Z), and halves the download cost.

forecast_hours=72 caps the response to 3 days per call instead of the default 10-day
horizon — more than enough margin over the longest lead time we actually use (Task A
tops out around lead +60h under the D-1 12Z mapping), and meaningfully cuts response
size/time per call.

Rate limiting: the Single Runs API enforces some undocumented per-minute request cap
(empirically hit around 200-250 locations' worth of requests within under a minute —
see PLAN.md/session notes; not batch-size-dependent, request-volume-dependent). This
script backs off 65s and retries on HTTP 429 rather than trying to precompute a safe
rate.

Resumable by construction: every (locations-batch, run_time) call is cached forever by
open_meteo.fetch_single_run's requests-cache layer, and this script also skips a run
entirely if its parquet output already exists. Killing and re-running it loses at most
the batch in flight.

DAY_STRIDE=2 (every other day rather than every day): a smoke test on one run (254
locations, 3 batches, 12 vars, forecast_hours=72) took 57.6s with no rate-limit hits.
Every day at 2 runs/day would be ~486 runs, ~7-8 hours — more than the time budget
allows. Every other day halves that to ~4 hours while EAGLE-I labels stay computed from
the full, non-subsampled 15-minute grid (this only thins which weather snapshots we
pull as training features, not the outage ground truth). This trades away some coverage
of short, single-day weather transients; worth revisiting as an ablation if time allows
(see PLAN.md).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_acquisition.county_coordinates import load_county_coordinates
from data_acquisition.open_meteo import RateLimitError, fetch_single_run

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WEATHER_DIR = PROJECT_ROOT / "data" / "raw" / "ifs_training_runs"

TRAIN_START = pd.Timestamp("2025-01-01")
TRAIN_END = pd.Timestamp("2025-08-31")
RUN_HOURS = [0, 12]
DAY_STRIDE = 2  # every other day — see module docstring for why
BATCH_SIZE = 100
FORECAST_HOURS = 72
MAX_RETRIES = 5
RETRY_SLEEP_SECONDS = 65


def load_training_locations() -> pd.DataFrame:
    counties = pd.read_csv(PROCESSED_DIR / "training_counties.csv", dtype={"fips_code": str})
    coords = load_county_coordinates()
    merged = counties.merge(coords, on="fips_code", how="left")
    missing = merged[merged["latitude"].isna()]
    if len(missing):
        print(f"Dropping {len(missing)} counties with no gazetteer coordinates: "
              f"{missing['fips_code'].tolist()}")
        merged = merged.dropna(subset=["latitude"])
    return merged[["fips_code", "latitude", "longitude"]].reset_index(drop=True)


def fetch_with_retry(locations: pd.DataFrame, run_time: pd.Timestamp) -> pd.DataFrame:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fetch_single_run(locations, run_time, forecast_hours=FORECAST_HOURS)
        except RateLimitError:
            if attempt == MAX_RETRIES:
                raise
            print(f"    rate limited (attempt {attempt}/{MAX_RETRIES}), "
                  f"sleeping {RETRY_SLEEP_SECONDS}s...")
            time.sleep(RETRY_SLEEP_SECONDS)
    raise RuntimeError("unreachable")


def run_output_path(run_time: pd.Timestamp) -> Path:
    return WEATHER_DIR / f"run_{run_time.strftime('%Y%m%dT%H%M')}.parquet"


def main():
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    locations = load_training_locations()
    print(f"Training locations: {len(locations)}")

    run_times = [
        pd.Timestamp(d) + pd.Timedelta(hours=h)
        for d in pd.date_range(TRAIN_START, TRAIN_END, freq="D")[::DAY_STRIDE]
        for h in RUN_HOURS
    ]
    print(f"Runs to fetch: {len(run_times)} ({run_times[0]} .. {run_times[-1]})")

    batches = [locations.iloc[i : i + BATCH_SIZE] for i in range(0, len(locations), BATCH_SIZE)]
    print(f"Batches per run: {len(batches)} (batch size {BATCH_SIZE})")

    t_start = time.time()
    n_done = 0
    for i, run_time in enumerate(run_times):
        out_path = run_output_path(run_time)
        if out_path.exists():
            n_done += 1
            continue

        frames = [fetch_with_retry(batch, run_time) for batch in batches]
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(out_path)
        n_done += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(run_times):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(run_times) - (i + 1)) / rate if rate > 0 else float("inf")
            print(f"[{i+1}/{len(run_times)}] {run_time} done "
                  f"({elapsed/60:.1f}m elapsed, ~{eta/60:.1f}m remaining)")

    print(f"\nDone. {n_done}/{len(run_times)} runs available in {WEATHER_DIR}")


if __name__ == "__main__":
    main()
