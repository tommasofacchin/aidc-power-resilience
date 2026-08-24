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

A Task A row is an hourly MEAN, not the value at the top of the hour — the organisers
asked for hourly mean aggregation and the two are measurably different quantities. See
TASK_A_SUBHOUR_OFFSETS below for the measurement and for why this needs no retraining.

issue_time -> run mapping (PLAN.md section 2.2), accounting for ~5-7h IFS dissemination
lag. Both tasks deliberately share ONE run per calendar day:
- Task A: issue D 00:00Z          -> run (D-1) 12:00Z  (lead-from-run +12h..+60h)
- Task B: issue D 00/06/12/18Z    -> run (D-1) 12:00Z  (lead-from-run +12h15m..+36h)

Task B could legitimately use a fresher run (its issue_time minus 6h), and an earlier
version of this file did. It does not, because the Open-Meteo Single Runs archive is
rate-limited to roughly 110-130 calls/day and that quota is the binding constraint on
whether this submission can be generated at all: one run per day costs 93 distinct
calls for the whole test window, a fresh run every 6h costs 367 — three-plus days of
quota versus under one. The guidelines only require that the weather input be a
forecast genuinely available at issue_time, and a run from 12-30h earlier always is.
The skill cost is small by design: PLAN.md section 2.3 argues Task B's genuine 15-minute
signal comes from the autoregressive outage state, not from weather, which the model
only uses as a slowly-varying risk envelope. If quota ever stops being scarce, reverting
is a one-line change to task_b_run_time().

predicted_x is emitted in the denominator the organisers grade against -- MCC.csv as
published -- which is NOT the reconciled denominator everything upstream of the output
divides by. The conversion is one constant per county and happens in exactly one place;
see to_grading_units() for the constants and for the forum answer that fixes them.

Autoregressive EAGLE-I features are computed from data at or before issue_time — this
is legitimate per the submission guidelines (only the weather input is restricted to
what a forecast could show; observed outage history is realistic, see PLAN.md 2.4) and
is checked with the shared causality assert, not by construction alone this time, since
here issue_time genuinely differs from run_time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_acquisition.bulk_download_training_weather import fetch_with_retry
from data_acquisition.county_coordinates import load_county_coordinates
from data_acquisition.eagle_i import (
    PROCESSED_DIR,
    load_outage_events,
    load_published_total_customers,
    load_total_customers,
)
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
from blend import apply_blend
from model_bundle import ModelBundle, load_bundle
from preprocessing.build_target_grid import build_dense_grid
from validate_submission import validate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = PROJECT_ROOT / "submission"
BUNDLE_DIR = PROCESSED_DIR / "model_bundle"
SELECTED_COUNTIES_PATH = PROCESSED_DIR / "selected_counties.csv"
BLEND_WEIGHTS_PATH = BUNDLE_DIR / "blend_weights.json"

TASK_A_ISSUE_START = pd.Timestamp("2025-08-30T00:00:00")
TASK_A_ISSUE_END = pd.Timestamp("2025-11-29T00:00:00")
TASK_B_ISSUE_START = pd.Timestamp("2025-08-31T18:00:00")
TASK_B_ISSUE_END = pd.Timestamp("2025-11-30T18:00:00")

TASK_A_LEADS = [pd.Timedelta(hours=h) for h in range(1, 49)]
TASK_B_LEADS = [pd.Timedelta(minutes=15 * q) for q in range(1, 25)]

# Task A is scored at 1-hour resolution against a ground truth EAGLE-I only records
# every 15 minutes, so the hourly value has to be an aggregate of four readings. The
# organisers' guidance on the discussion wall is explicit: "For task A we will suggest
# hourly mean aggregation." Predicting the instantaneous value at :00 instead is a
# measurably different quantity — on the five reporting counties over Jan-Aug 2025, the
# two disagree by 0.00206 on average across hours that contain an event, about 20% of
# the value being predicted, and the RMSE of the disagreement is 0.0043, roughly 18% of
# this system's autumn RMSE. That is error bought for nothing.
#
# So a Task A row is the mean of the four quarter-hours the label opens: 03:00 means the
# mean over [03:00, 04:00). Left-labelled, because that is what pandas' resample("1h")
# does by default and therefore what an aggregation of the 15-minute series most likely
# is on the scoring side. Task B is unaffected: its rows are 15-minute instants already,
# which is the source's native granularity.
#
# No retraining is needed for this and none was done. The mean of unbiased estimates of
# the four instants is an unbiased estimate of their mean, so the existing model and
# blend are applied unchanged at each quarter and averaged afterwards. It costs no API
# quota either: the 15-minute weather is interpolated locally from the same cached run.
TASK_A_SUBHOUR_OFFSETS = [pd.Timedelta(minutes=m) for m in (0, 15, 30, 45)]

# Every submission batch asks for exactly this many forecast hours, Task A and Task B
# alike. This is NOT a per-batch minimum, and must not be recomputed per batch:
# forecast_hours goes into the request URL and requests-cache keys on the URL, so
# asking 62h for Task A and 38h for Task B would put the two tasks on different cache
# entries for the very run they were changed to share — doubling the call count and
# undoing the whole point of task_b_run_time(). Task A needs +60h from its run and
# Task B +36h; 62 covers both, and is the exact value Task A already used before this
# constant existed, so responses cached earlier stay valid.
SUBMISSION_FORECAST_HOURS = 62


def task_a_run_time(issue_time: pd.Timestamp) -> pd.Timestamp:
    return issue_time - pd.Timedelta(days=1) + pd.Timedelta(hours=12)


def task_b_run_time(issue_time: pd.Timestamp) -> pd.Timestamp:
    """The same (D-1) 12Z run Task A uses for this calendar day — see module docstring.

    Deliberately not issue_time - 6h: sharing Task A's run cuts the submission from 367
    distinct API calls to 93, and the forecast is still 12-30h old at issue_time rather
    than 6h, which is a legitimate "available at issue_time" input.
    """
    return task_a_run_time(issue_time.normalize())


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
    locations: pd.DataFrame, run_time: pd.Timestamp, forecast_hours: int, need_15min: bool,
    climatology: pd.DataFrame,
) -> pd.DataFrame:
    long_df = fetch_single_run(locations, run_time, forecast_hours=forecast_hours)
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


def prefetch_runs(schedule: pd.DataFrame, counties: pd.DataFrame) -> None:
    """Pull every distinct IFS run the schedule needs into the on-disk cache first.

    Separating the API phase from the inference phase matters here for one practical
    reason: the Single Runs endpoint enforces minute, hour and day rate limits, and the
    day one cannot be waited out inside a live process. Fetching up front means a
    day-quota stop costs only the runs not yet fetched — everything already cached is
    permanent, so re-running resumes for free. Doing it inline in generate_predictions()
    instead would abandon the loop's progress on every rate-limit stop, and that loop is
    the slow part (457 batches of autoregressive feature construction).
    """
    runs = sorted(set(schedule["run_time"]))
    locations = counties[["fips_code", "latitude", "longitude"]]
    print(f"Prefetching {len(runs)} distinct IFS runs for {len(locations)} counties "
          f"at forecast_hours={SUBMISSION_FORECAST_HOURS}...")
    for i, run_time in enumerate(runs, 1):
        fetch_with_retry(locations, run_time, forecast_hours=SUBMISSION_FORECAST_HOURS)
        print(f"  [{i}/{len(runs)}] {run_time}", flush=True)
    print("All runs cached — the prediction loop below makes no further API calls.\n")


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


def to_grading_units(predictions: pd.DataFrame, mcc: pd.DataFrame) -> pd.DataFrame:
    """Rescale predicted_x from the reconciled denominator to MCC.csv as published.

    Everything upstream of here — the target grid, the training table, the model, the
    blend — divides customers_out by the reconciled denominator, because MCC is wrong
    by up to 21x for some counties and a target that saturates at 1 for whole storms is
    not something to fit. That reasoning is unchanged and is still the right way to
    build the model (preprocessing/reconcile_denominators.py has the evidence).

    It is not, however, the unit the submission is read in. Asked directly whether
    scoring uses total_customers exactly as MCC.csv gives it even where the recorded
    customers_out exceeds it, the organisers answered on 24 Aug 2026:

        "Note even the official dataset could have mistake, when grading we will ignore
        such timestamp"

    — which settles it: the grader divides by MCC as published (nothing else could
    produce a timestamp worth ignoring) and drops the timestamps that come out above 1.
    So x is a ratio in *their* denominator, and a prediction in ours is simply in the
    wrong units. Since x = customers_out / denominator is linear in 1/denominator, the
    conversion is one constant per county:

        predicted_x_published = predicted_x_reconciled * reconciled / published

    Three of the five reporting counties reconcile to MCC already and are multiplied by
    exactly 1. The other two are not small corrections: Mecklenburg x20.89 (28,172 vs
    588,615) and Arecibo x4.66 (41,122 vs 191,803). Leaving them unconverted was not the
    safe side of the argument it looked like — on autumn 2024, the grader ignores 1.7%
    of Mecklenburg's intervals and 0.2% of Arecibo's for exceeding 1, and on all the
    rest the truth in MCC units averages 0.0151 and 0.0300 against the 0.00072 and
    0.00643 this pipeline would have submitted. That is a systematic 21x/4.7x
    under-prediction across ~99% of the rows those two counties are actually scored on.

    Clipping back into [0, 1] after the multiplication costs nothing here — the largest
    converted prediction is ~0.58 — but it is kept because the guidelines require the
    range and a future, more confident model could cross it on a county whose published
    denominator is too small to hold the event.
    """
    reconciled = predictions["fips_code"].map(
        mcc.set_index("fips_code")["total_customers"]
    )
    published = predictions["fips_code"].map(
        load_published_total_customers().set_index("fips_code")["total_customers"]
    )
    missing = predictions.loc[reconciled.isna() | published.isna(), "fips_code"].unique()
    if len(missing):
        raise RuntimeError(
            f"Missing a denominator for {sorted(missing)} — in the reconciled table, in "
            f"MCC.csv, or in both. The conversion needs both to be numbers; without them "
            f"the county goes out either as NaN or, worse, silently in the wrong unit."
        )
    factor = reconciled / published
    print("")
    print("Converting predicted_x to the grading denominator (MCC.csv as published):")
    for fips, f in sorted(dict(zip(predictions["fips_code"], factor)).items()):
        print(f"  {fips}  x{f:.4f}")
    out = predictions.copy()
    out["predicted_x"] = np.clip(out["predicted_x"] * factor, 0, 1)
    return out


def generate_predictions(schedule: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    counties = load_reporting_counties()
    fips_codes = counties["fips_code"].tolist()
    events = load_outage_events(years=[2025], fips_codes=set(fips_codes))
    mcc = load_total_customers()

    # Absent weights mean "model only" — an explicit, reportable state, unlike a stale
    # file silently applying weights fitted against a different model.
    blend_weights = {}
    if BLEND_WEIGHTS_PATH.exists():
        blend_weights = json.loads(BLEND_WEIGHTS_PATH.read_text(encoding="utf8"))
        print(f"Blending with persistence, weights by operational lead: {blend_weights}")
    else:
        print(f"No {BLEND_WEIGHTS_PATH.name} — predicting from the model alone. "
              f"Run `python blend.py` to fit it.")

    feature_cols = bundle.feature_names
    batches = []

    for _, sched_row in schedule.iterrows():
        task_id, issue_time, run_time, leads = (
            sched_row["task_id"], sched_row["issue_time"], sched_row["run_time"], sched_row["leads"]
        )
        assert_causal(pd.Series([run_time]), issue_time, f"Task {task_id} IFS run_time vs issue_time")

        # One fixed horizon for every batch so Task A and Task B hit the same cache
        # entry for the run they share (see SUBMISSION_FORECAST_HOURS). Verify the
        # batch really is covered rather than trusting the constant silently: falling
        # short would surface far downstream as an empty weather slice.
        sub_offsets = TASK_A_SUBHOUR_OFFSETS if task_id == "A" else [pd.Timedelta(0)]
        last_target_time = issue_time + leads[-1] + sub_offsets[-1]
        needed_h = (last_target_time - run_time).total_seconds() / 3600
        if needed_h >= SUBMISSION_FORECAST_HOURS:
            raise RuntimeError(
                f"Task {task_id} batch at issue_time={issue_time} needs {needed_h:.2f}h "
                f"of forecast from run {run_time}, but SUBMISSION_FORECAST_HOURS is "
                f"{SUBMISSION_FORECAST_HOURS}. Raising the constant is correct, but note "
                f"every response cached at the old value would be re-requested at the "
                f"new one, spending Open-Meteo quota that may not be there."
            )
        # Both tasks need the interpolated 15-minute grid now: Task B because its rows
        # are 15-minute instants, Task A because its hourly mean is built from four of
        # them. Interpolation is local, so this spends no extra quota.
        weather = build_weather_for_batch(
            counties[["fips_code", "latitude", "longitude"]], run_time,
            SUBMISSION_FORECAST_HOURS, need_15min=True,
            climatology=bundle.climatology,
        )

        ar = build_autoregressive_for_issue(fips_codes, issue_time, events, mcc)

        for lead in leads:
            target_time = issue_time + lead
            # One pass per quarter-hour for Task A (see TASK_A_SUBHOUR_OFFSETS), a single
            # pass at the instant itself for Task B. Each pass is the full predictor —
            # model then blend, at that quarter's own operational lead — because the
            # quantity being averaged is the system's genuine estimate of each instant,
            # not an intermediate.
            sub_predictions: list[np.ndarray] = []
            emitted_fips: pd.DataFrame | None = None
            for offset in sub_offsets:
                sub_target_time = target_time + offset
                rows = weather[weather["time"] == sub_target_time.tz_localize("UTC")].copy()
                if rows.empty:
                    raise RuntimeError(
                        f"No weather row for target_time={sub_target_time} from "
                        f"run_time={run_time} (task {task_id}, issue_time={issue_time}) — "
                        f"forecast_hours was too short."
                    )
                # Sorted so the four quarter-hour prediction vectors line up county by
                # county; averaging them relies on a shared row order, and nothing else
                # in this loop guarantees one.
                rows = rows.sort_values("fips_code").reset_index(drop=True)
                rows = rows.merge(ar.drop(columns=["issue_time"]), on="fips_code", how="left")
                rows["issue_time"] = issue_time
                rows["target_time"] = sub_target_time
                # NOTE: lead_hours is deliberately NOT overwritten with (target - issue).
                # pivot_weather already set it to (target - run_time), i.e. forecast age,
                # which is exactly what it meant during training (where issue == run).
                # Overwriting it here would tell the model "this forecast is 1 hour old"
                # about weather that is really 13 hours old for Task A, making it
                # over-trust stale forecasts. Keeping forecast-age semantics also puts
                # Task B's leads at 12.25-36h — inside the 1-71h range training actually
                # saw — instead of 0.25-6h, which extrapolated below anything training
                # ever saw. (Those figures are for the shared-run mapping; they were
                # 6.25-12h when Task B still fetched its own run every 6h.)
                rows = add_temporal_features(rows, "target_time")
                rows["total_customers"] = rows["fips_code"].map(
                    mcc.set_index("fips_code")["total_customers"])

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

                # Blend toward persistence at short OPERATIONAL lead — the quarter's own
                # lead here, not the `lead_hours` feature, which carries forecast age and
                # is 12-36h for every Task B row. See blend.py for why the two axes
                # diverge at serving time and why persistence wins below ~12h (held-out
                # RMSE -3.9% in the 0-6h bucket, which is the whole of Task B).
                if blend_weights:
                    operational_lead = (lead + offset).total_seconds() / 3600
                    pred = apply_blend(
                        pred, rows["x_at_issue"].to_numpy(),
                        np.full(len(rows), operational_lead), blend_weights,
                    )
                sub_predictions.append(pred)
                if emitted_fips is None:
                    emitted_fips = rows[["fips_code"]]

            # One row per label: the instant for Task B, the mean of its four quarters
            # for Task A.
            pred = np.mean(sub_predictions, axis=0)

            batch_out = emitted_fips.merge(counties[["fips_code", "county", "state"]], on="fips_code", how="left")
            batch_out["task_id"] = task_id
            batch_out["issue_time"] = issue_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            batch_out["target_time"] = target_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            batch_out["predicted_x"] = pred
            batches.append(
                batch_out[["task_id", "fips_code", "county", "state", "issue_time", "target_time", "predicted_x"]]
            )

    return to_grading_units(pd.concat(batches, ignore_index=True), mcc)


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
    prefetch_runs(schedule, load_reporting_counties())
    predictions = generate_predictions(schedule, bundle)

    print(f"\nGenerated {len(predictions)} rows")
    out_path = SUBMISSION_DIR / ("predictions_smoke_test.csv" if args.smoke_test else "predictions.csv")
    # Fixed-point rather than pandas' default float repr, which switches to scientific
    # notation below 1e-4 and would render ~a third of the file as 9.95e-05. Every parser
    # reads that correctly, so it is not a spec violation, but PLAN.md's pre-submission
    # checklist asks for a file a reviewer can read without decoding exponents. Ten
    # decimals is lossless here: predictions top out below 0.6 and the metric that scores
    # them lives around 1e-2, so 1e-10 is far below anything that could matter.
    predictions.to_csv(out_path, index=False, float_format="%.10f")
    print(f"Saved to {out_path}")

    print("\n=== Validating ===")
    counties = pd.read_csv(SELECTED_COUNTIES_PATH, dtype={"fips_code": str})
    result = validate(out_path, expected_counties=set(counties["fips_code"]))
    result.report()
