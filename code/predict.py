"""
Generate the actual submission: predictions.csv for the rolling issuance schedule
(PLAN.md section 2.1), applying the real issue_time -> IFS run mapping (section 2.2)
that respects dissemination lag — this is the one place in the whole pipeline where
that mapping matters; training (build_training_table.py) deliberately does not use it
(see that file's docstring for why that's not a train/serve inconsistency that matters).

Schedule (PLAN.md section 2.1):
- Task A: daily 00:00Z, 2025-08-30 .. 2025-11-29, lead +1h..+48h (48 rows/batch)
- Task B: every 6h (00/06/12/18Z), 2025-08-31T18:00Z .. 2025-11-30T18:00Z,
  lead +15m..+6h (24 rows/batch)

issue_time -> run mapping (PLAN.md section 2.2), accounting for ~5-7h IFS dissemination
lag:
- Task A: issue D 00:00Z -> run (D-1) 12:00Z  (so target lead-from-run is +12h..+60h)
- Task B: issue T        -> run T - 6h         (so target lead-from-run is +6h15m..+12h)

Autoregressive EAGLE-I features are computed from data at or before issue_time — this
is legitimate per the submission guidelines (only the weather input is restricted to
what a forecast could show; observed outage history is realistic, see PLAN.md 2.4) and
is checked with the shared causality assert, not by construction alone this time, since
here issue_time genuinely differs from run_time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_acquisition.county_coordinates import load_county_coordinates
from data_acquisition.eagle_i import PROCESSED_DIR, load_max_customer_counts, load_outage_events
from data_acquisition.open_meteo import fetch_single_run
from features.autoregressive import add_autoregressive_features
from features.build_training_table import add_temporal_features
from features.causality import assert_causal
from features.weather_features import (
    add_climatology_features,
    add_derived_weather_features,
    interpolate_to_15min,
    pivot_weather,
)
from model_bundle import ModelBundle, load_bundle
from preprocessing.build_target_grid import build_dense_grid
from validate_submission import validate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = PROJECT_ROOT / "submission"
BUNDLE_DIR = PROCESSED_DIR / "model_bundle"
SELECTED_COUNTIES_PATH = PROCESSED_DIR / "selected_counties.csv"

TASK_A_ISSUE_START = pd.Timestamp("2025-08-30T00:00:00")
TASK_A_ISSUE_END = pd.Timestamp("2025-11-29T00:00:00")
TASK_B_ISSUE_START = pd.Timestamp("2025-08-31T18:00:00")
TASK_B_ISSUE_END = pd.Timestamp("2025-11-30T18:00:00")

TASK_A_LEADS = [pd.Timedelta(hours=h) for h in range(1, 49)]
TASK_B_LEADS = [pd.Timedelta(minutes=15 * q) for q in range(1, 25)]


def task_a_run_time(issue_time: pd.Timestamp) -> pd.Timestamp:
    return issue_time - pd.Timedelta(days=1) + pd.Timedelta(hours=12)


def task_b_run_time(issue_time: pd.Timestamp) -> pd.Timestamp:
    return issue_time - pd.Timedelta(hours=6)


def build_schedule(
    task_a_range: tuple[pd.Timestamp, pd.Timestamp] = (TASK_A_ISSUE_START, TASK_A_ISSUE_END),
    task_b_range: tuple[pd.Timestamp, pd.Timestamp] = (TASK_B_ISSUE_START, TASK_B_ISSUE_END),
) -> pd.DataFrame:
    a_issues = pd.date_range(task_a_range[0], task_a_range[1], freq="D")
    b_issues = pd.date_range(task_b_range[0], task_b_range[1], freq="6h")

    rows = []
    for issue in a_issues:
        rows.append({"task_id": "A", "issue_time": issue, "run_time": task_a_run_time(issue),
                      "leads": TASK_A_LEADS})
    for issue in b_issues:
        rows.append({"task_id": "B", "issue_time": issue, "run_time": task_b_run_time(issue),
                      "leads": TASK_B_LEADS})
    return pd.DataFrame(rows)


def load_reporting_counties() -> pd.DataFrame:
    counties = pd.read_csv(SELECTED_COUNTIES_PATH, dtype={"fips_code": str})
    coords = load_county_coordinates()
    return counties.merge(coords, on="fips_code", how="left")


def build_weather_for_batch(
    locations: pd.DataFrame, run_time: pd.Timestamp, max_lead_hours: int, need_15min: bool,
    climatology: pd.DataFrame,
) -> pd.DataFrame:
    long_df = fetch_single_run(locations, run_time, forecast_hours=max_lead_hours + 1)
    long_df["run_time"] = run_time  # fetch_single_run already sets this, kept explicit for clarity
    wide = pivot_weather(long_df)
    wide = add_derived_weather_features(wide)
    # Training-fitted thresholds, never refitted from this batch — refitting from a
    # single ~60h window roughly halves the p95 gust and redefines the feature.
    wide = add_climatology_features(wide, climatology)
    if need_15min:
        # Task B: IFS has no native 15-minute output — interpolate (see
        # features/weather_features.interpolate_to_15min for why this is the right
        # place to do it). Task A's leads all fall exactly on the hour and skip this.
        wide = interpolate_to_15min(wide)
    return wide


def build_autoregressive_for_issue(
    fips_codes: list[str], issue_time: pd.Timestamp, events: pd.DataFrame, mcc: pd.DataFrame
) -> pd.DataFrame:
    """AR features as of issue_time, for a handful of counties — cheap enough to
    recompute per batch rather than caching a rolling multi-month grid, since the
    submission generation only touches 5 counties."""
    lookback_start = issue_time - pd.Timedelta(hours=48)  # margin beyond the 24h rolling max
    grid = build_dense_grid(
        events[events["fips_code"].isin(fips_codes)], mcc, fips_codes=fips_codes,
        start=lookback_start, end=issue_time,
    )
    grid = add_autoregressive_features(grid)
    # Causal by construction: the grid was built with end=issue_time, so no row in it
    # — and therefore no lag/rolling feature derived from it — can reference a
    # timestamp after issue_time. See features/autoregressive.py's docstring.
    at_issue = grid[grid["run_start_time"] == issue_time]

    return at_issue.rename(columns={"run_start_time": "issue_time", "x": "x_at_issue"})


def generate_predictions(schedule: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    counties = load_reporting_counties()
    fips_codes = counties["fips_code"].tolist()
    events = load_outage_events(years=[2025])
    mcc = load_max_customer_counts()

    feature_cols = bundle.feature_names
    batches = []

    for _, sched_row in schedule.iterrows():
        task_id, issue_time, run_time, leads = (
            sched_row["task_id"], sched_row["issue_time"], sched_row["run_time"], sched_row["leads"]
        )
        assert_causal(pd.Series([run_time]), issue_time, f"Task {task_id} IFS run_time vs issue_time")

        # Forecast horizon needed from this run: from run_time out to the batch's
        # furthest target_time (issue_time + last lead). E.g. Task A's run sits 12h
        # before issue_time and the last lead is +48h, so this run needs +60h of
        # forecast; Task B needs +12h. +1h margin for the ceiling rounding below.
        last_target_time = issue_time + leads[-1]
        max_lead_h = int(np.ceil((last_target_time - run_time).total_seconds() / 3600)) + 1
        weather = build_weather_for_batch(
            counties[["fips_code", "latitude", "longitude"]], run_time, max_lead_h,
            need_15min=(task_id == "B"), climatology=bundle.climatology,
        )

        ar = build_autoregressive_for_issue(fips_codes, issue_time, events, mcc)

        for lead in leads:
            target_time = issue_time + lead
            rows = weather[weather["time"] == target_time.tz_localize("UTC")].copy()
            if rows.empty:
                raise RuntimeError(
                    f"No weather row for target_time={target_time} from run_time={run_time} "
                    f"(task {task_id}, issue_time={issue_time}) — forecast_hours was too short."
                )
            rows = rows.merge(ar.drop(columns=["issue_time"]), on="fips_code", how="left")
            rows["issue_time"] = issue_time
            rows["target_time"] = target_time
            # NOTE: lead_hours is deliberately NOT overwritten with (target - issue).
            # pivot_weather already set it to (target - run_time), i.e. forecast age,
            # which is exactly what it meant during training (where issue == run).
            # Overwriting it here would tell the model "this forecast is 1 hour old"
            # about weather that is really 13 hours old for Task A, making it
            # over-trust stale forecasts. Keeping forecast-age semantics also puts
            # Task B's leads at 6.25-12h — inside the training range — instead of
            # 0.25-6h, which extrapolated below anything training ever saw.
            rows = add_temporal_features(rows, "target_time")
            rows["total_customers"] = rows["fips_code"].map(mcc.set_index("fips_code")["total_customers"])

            missing_features = [c for c in feature_cols if c not in rows.columns]
            if missing_features:
                raise RuntimeError(
                    f"Features the model expects are absent at prediction time: "
                    f"{missing_features}. Filling them with NaN would silently degrade "
                    f"predictions instead of surfacing the mismatch."
                )
            X = rows[feature_cols].copy()
            # Encode with the TRAINING categories: LightGBM consumes categoricals by
            # integer code, so re-deriving categories from these 5 counties alone would
            # remap every county onto a different identity.
            X["fips_code"] = bundle.encode_fips(X["fips_code"])
            pred = np.clip(bundle.booster.predict(X), 0, 1)

            batch_out = rows[["fips_code"]].merge(counties[["fips_code", "county", "state"]], on="fips_code", how="left")
            batch_out["task_id"] = task_id
            batch_out["issue_time"] = issue_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            batch_out["target_time"] = target_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            batch_out["predicted_x"] = pred
            batches.append(
                batch_out[["task_id", "fips_code", "county", "state", "issue_time", "target_time", "predicted_x"]]
            )

    return pd.concat(batches, ignore_index=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                         help="Generate just 2 issue_times per task instead of the full schedule.")
    args = parser.parse_args()

    if args.smoke_test:
        schedule = build_schedule(
            task_a_range=(TASK_A_ISSUE_START, TASK_A_ISSUE_START + pd.Timedelta(days=1)),
            task_b_range=(TASK_B_ISSUE_START, TASK_B_ISSUE_START + pd.Timedelta(hours=6)),
        )
        print(f"SMOKE TEST: {len(schedule)} issue_times only")
    else:
        schedule = build_schedule()
        print(f"FULL SCHEDULE: {len(schedule)} issue_times "
              f"({(schedule['task_id']=='A').sum()} Task A, {(schedule['task_id']=='B').sum()} Task B)")

    bundle = load_bundle(BUNDLE_DIR)
    predictions = generate_predictions(schedule, bundle)

    print(f"\nGenerated {len(predictions)} rows")
    out_path = SUBMISSION_DIR / ("predictions_smoke_test.csv" if args.smoke_test else "predictions.csv")
    predictions.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    print("\n=== Validating ===")
    counties = pd.read_csv(SELECTED_COUNTIES_PATH, dtype={"fips_code": str})
    result = validate(out_path, expected_counties=set(counties["fips_code"]))
    result.report()
