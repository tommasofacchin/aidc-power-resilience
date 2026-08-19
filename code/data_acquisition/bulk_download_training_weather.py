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

Rate limiting: the Single Runs API enforces THREE separate undocumented caps, found
one at a time as each was hit in turn — per-minute, per-hour, and per-day (see
open_meteo.RateLimitError for the details and how each is classified). The first two
are worth retrying through (65s and ~62min backoff respectively); the daily one isn't
— see the try/except around the main loop below, which stops the script cleanly
instead of retrying into a crash the way it did the first time this was hit.

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

import numpy as np
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
# The just-discovered DAILY quota (see module docstring) makes call COUNT the scarce
# resource, not per-call data volume — a single call with all 102 training counties
# was already proven to work well within URL-length limits (n=200 locations x 12
# variables succeeded cleanly during earlier reconnaissance). Batching all counties
# into one call per run halves call consumption vs. the previous 2-batches-per-run
# split, directly doubling how many runs fit in one day's quota.
BATCH_SIZE = 110
FORECAST_HOURS = 72
MAX_RETRIES = 5
MINUTE_RETRY_SLEEP_SECONDS = 65
HOUR_RETRY_SLEEP_SECONDS = 62 * 60  # a 65s sleep against an hourly cap just spins uselessly
MAX_HOUR_RETRIES = 10  # generous ceiling (~10h) so an unattended overnight run survives

# Discovered 19 Aug night: the DAILY quota (~110-130 total calls/day, see module
# docstring) is shared with the ~367 separate calls the real submission generation
# needs later (code/predict.py — one call per distinct issue_time->run mapping across
# both tasks, verified by actually building that schedule, not estimated). Generation
# is close to a hard floor: the guidelines fix a *minimum* issuance frequency, so it
# can't be thinned the way training can. Training gets a firm, conservative budget for
# ADDITIONAL runs (on top of whatever's already downloaded) so there's quota left for
# generation instead of the two competing for the same days. See PLAN.md section 2.5's
# 19 Aug addendum for the reasoning and PLAN.md section 2.1 for the generation math.
ADDITIONAL_RUN_BUDGET = 80


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
    minute_attempt = 0
    hour_attempt = 0
    while True:
        try:
            return fetch_single_run(locations, run_time, forecast_hours=FORECAST_HOURS)
        except RateLimitError as e:
            if e.scope == "day":
                # Waiting this out inside a live process isn't practical (resets on
                # Open-Meteo's clock, not ours, and "tomorrow" is not a bounded
                # sleep). Earlier this exact case fell through to the minute-retry
                # branch, burned 5 retries x 65s uselessly, and crashed with a raw
                # traceback — this script is resumable by construction (skips
                # existing run_*.parquet files), so the right move is to stop
                # cleanly now and let a later invocation continue once the quota
                # resets, not to keep hammering an API that just told us to stop.
                raise
            if e.scope == "hour":
                hour_attempt += 1
                if hour_attempt > MAX_HOUR_RETRIES:
                    raise
                print(f"    HOURLY rate limit hit ({e.reason!r}), attempt "
                      f"{hour_attempt}/{MAX_HOUR_RETRIES} — sleeping "
                      f"{HOUR_RETRY_SLEEP_SECONDS/60:.0f}m...")
                time.sleep(HOUR_RETRY_SLEEP_SECONDS)
            else:
                minute_attempt += 1
                if minute_attempt > MAX_RETRIES:
                    raise
                print(f"    minute rate limited (attempt {minute_attempt}/{MAX_RETRIES}), "
                      f"sleeping {MINUTE_RETRY_SLEEP_SECONDS}s...")
                time.sleep(MINUTE_RETRY_SLEEP_SECONDS)


def run_output_path(run_time: pd.Timestamp) -> Path:
    return WEATHER_DIR / f"run_{run_time.strftime('%Y%m%dT%H%M')}.parquet"


def main():
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    locations = load_training_locations()
    print(f"Training locations: {len(locations)}")

    full_schedule = [
        pd.Timestamp(d) + pd.Timedelta(hours=h)
        for d in pd.date_range(TRAIN_START, TRAIN_END, freq="D")[::DAY_STRIDE]
        for h in RUN_HOURS
    ]
    already_done = [rt for rt in full_schedule if run_output_path(rt).exists()]
    still_needed = [rt for rt in full_schedule if not run_output_path(rt).exists()]

    if len(still_needed) <= ADDITIONAL_RUN_BUDGET:
        run_times = full_schedule
    else:
        # Evenly-spaced subsample of what's left, not the next ADDITIONAL_RUN_BUDGET
        # in chronological order — the 40 runs already on disk cluster in Jan-Mar
        # (see PLAN.md), so taking the next chunk sequentially would spend the whole
        # budget extending that same cluster and never reach the summer/early
        # hurricane-season months the Jan-Aug window was originally chosen to cover.
        idx = np.linspace(0, len(still_needed) - 1, ADDITIONAL_RUN_BUDGET, dtype=int)
        sampled = sorted({still_needed[i] for i in idx})
        run_times = already_done + sampled
        print(f"Budget cap: {len(still_needed)} runs still needed for the full schedule, "
              f"capped to {len(sampled)} more (evenly spread across the remaining span) "
              f"-> {len(run_times)} total this pass.")

    print(f"Runs to fetch: {len(run_times)} ({run_times[0]} .. {run_times[-1]})")

    batches = [locations.iloc[i : i + BATCH_SIZE] for i in range(0, len(locations), BATCH_SIZE)]
    print(f"Batches per run: {len(batches)} (batch size {BATCH_SIZE})")

    t_start = time.time()
    n_done = 0
    try:
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
    except RateLimitError as e:
        if e.scope != "day":
            raise
        remaining = len(run_times) - n_done
        print(f"\nSTOPPING (not a crash): {e.reason!r}")
        print(f"{n_done}/{len(run_times)} runs done, {remaining} remaining. "
              f"This script is resumable — just re-run it once Open-Meteo's daily "
              f"quota resets; it skips every run already saved in {WEATHER_DIR}.")
        return

    print(f"\nDone. {n_done}/{len(run_times)} runs available in {WEATHER_DIR}")


if __name__ == "__main__":
    main()
