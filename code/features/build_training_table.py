"""
Assemble the final training table: weather features + autoregressive EAGLE-I state,
joined at issue_time, against the future outcome (target_x) at target_time.

Training simplification, stated explicitly: issue_time := run_time (the weather run's
own initialization time), not the D-1 12Z / T-6h issue_time -> run mapping from
PLAN.md section 2.2. That mapping is a deployment-time choice for generating the actual
submission (a later step — see code/predict.py, not yet written) accounting for IFS
dissemination lag; it does not need to be replicated during training, because training
never pulls weather for a target from any run other than the one whose own trajectory
that target row is drawn from. There is no future-peeking either way: autoregressive
features only look at EAGLE-I data at or before run_time by construction (see
features/autoregressive.py), and weather features only use hours from that single,
already-fully-available run. The only effect of this simplification is that lead_hours
seen in training runs slightly longer than what the real D-1 12Z mapping will use at
serving time for Task A's shortest lead times — training still covers that range (every
run has hours out to lead_hours=72), so the model generalizes to it; predict.py's own
issue_time will be computed the real way when the time comes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_acquisition.eagle_i import PROCESSED_DIR, load_max_customer_counts, load_outage_events
from features.autoregressive import add_autoregressive_features
from features.weather_features import add_climatology_features, build_weather_base, fit_climatology
from preprocessing.build_target_grid import build_dense_grid

AR_FEATURE_COLS = [
    "x_lag_15m", "x_lag_30m", "x_lag_1h", "x_lag_2h", "x_lag_6h",
    "x_trend_1h", "x_trend_2h", "x_max_24h",
    "outage_duration_15m_periods", "ongoing_outage",
]


def add_temporal_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    t = df[time_col]
    hour_frac = t.dt.hour + t.dt.minute / 60
    doy = t.dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def build_training_table(
    fips_codes: list[str], climatology_cutoff: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (training_table, fitted_climatology).

    The climatology is returned rather than just applied, because prediction time must
    reuse these exact thresholds — see code/model_bundle.py. `climatology_cutoff`
    should be the temporal-split boundary so the fit doesn't see validation data.
    """
    base = build_weather_base()
    base = base[base["fips_code"].isin(fips_codes)]
    climatology = fit_climatology(base, cutoff=climatology_cutoff)
    weather = add_climatology_features(base, climatology)
    weather = weather.rename(columns={"run_time": "issue_time", "time": "target_time"})
    # weather's `time` came back tz-aware (UTC); issue_time (ex run_time) is tz-naive.
    # Align both to tz-naive UTC so they merge cleanly against the EAGLE-I grid, which
    # is tz-naive throughout.
    weather["target_time"] = weather["target_time"].dt.tz_localize(None)

    events = load_outage_events(years=[2025])
    events = events[events["fips_code"].isin(fips_codes)]
    mcc = load_max_customer_counts()

    grid_start = weather["issue_time"].min() - pd.Timedelta(hours=24)  # margin for 24h lookback
    grid_end = weather["target_time"].max()
    grid = build_dense_grid(events, mcc, fips_codes=fips_codes, start=grid_start, end=grid_end)
    grid = add_autoregressive_features(grid)

    ar_at_issue = grid.rename(columns={"run_start_time": "issue_time", "x": "x_at_issue"})
    ar_at_issue = ar_at_issue[["fips_code", "issue_time", "x_at_issue", *AR_FEATURE_COLS]]

    label = grid[["fips_code", "run_start_time", "x"]].rename(
        columns={"run_start_time": "target_time", "x": "target_x"}
    )

    table = weather.merge(ar_at_issue, on=["fips_code", "issue_time"], how="inner")
    table = table.merge(label, on=["fips_code", "target_time"], how="inner")

    # --- Missing-value treatment (report section: preprocessing strategy) -----------
    # Two distinct sources of NaN were found in this pipeline, and they get different
    # treatment because they mean different things:
    #
    # 1. lead_hours == 0 rows: Open-Meteo's accumulator-type variables (precipitation,
    #    wind_gusts_10m, ...) are null at a run's own t=0 instant by API convention —
    #    there's no prior interval to have accumulated over. This isn't missing data,
    #    it's a well-defined "not applicable"; also lead_hours=0 doesn't correspond to
    #    any real task (Task A's shortest lead is +1h, Task B's is +15m — see
    #    submission_guidelines_phase1-AIDC.docx), so these rows are simply dropped.
    #
    # 2. target_x / autoregressive x_* features NaN: traced to the *raw EAGLE-I source
    #    file itself* containing explicit NaN customers_out values (~6.5% of all rows,
    #    not a pipeline bug) — EAGLE-I marks a (county, timestamp) as NaN when its ETL
    #    couldn't get a reading, which is meaningfully different from a row being
    #    entirely absent (customers_out implicitly 0, no outage). Silently coercing
    #    NaN to 0 would misrepresent "we don't know" as "confirmed no outage" and bias
    #    the model toward under-predicting risk during exactly the periods when
    #    utility reporting itself is degraded — plausibly correlated with the same
    #    severe weather we're trying to predict.
    #    - Target (target_x): dropped when NaN — there is no ground truth to train or
    #      score against for that row, and the real evaluation would face the same
    #      ambiguity.
    #    - Autoregressive input features (x_at_issue, x_lag_*, x_trend_*, x_max_24h):
    #      left as NaN. LightGBM natively learns a default split direction for missing
    #      values per split, which is the standard, correct way to handle this in a
    #      tree model — imputing a fabricated number here would be strictly worse.
    before = len(table)
    table = table[table["lead_hours"] > 0]
    table = table[table["target_x"].notna()]
    dropped = before - len(table)
    print(f"Dropped {dropped} rows ({100*dropped/before:.2f}%): lead_hours==0 or NaN target_x "
          f"(source EAGLE-I data-quality gaps — see comment above)")
    # ----------------------------------------------------------------------------------

    table = add_temporal_features(table, "target_time")
    table["total_customers"] = table["fips_code"].map(
        mcc.set_index("fips_code")["total_customers"]
    )

    return table, climatology


if __name__ == "__main__":
    from train import VAL_START

    training_counties = pd.read_csv(
        PROCESSED_DIR / "training_counties.csv", dtype={"fips_code": str}
    )
    fips_codes = training_counties["fips_code"].tolist()

    table, climatology = build_training_table(fips_codes, climatology_cutoff=VAL_START)
    climatology.to_parquet(PROCESSED_DIR / "climatology.parquet")
    print(f"Fitted climatology on run_time < {VAL_START.date()} for "
          f"{len(climatology)} counties -> {PROCESSED_DIR / 'climatology.parquet'}")

    print(f"Training table: {len(table)} rows, {len(table.columns)} columns")
    print(f"Counties: {table['fips_code'].nunique()}, issue_times: {table['issue_time'].nunique()}")
    print(f"lead_hours range: {table['lead_hours'].min()} .. {table['lead_hours'].max()}")
    print(f"target_x: zero={100*(table['target_x']==0).mean():.2f}%, "
          f"mean={table['target_x'].mean():.6f}, max={table['target_x'].max():.4f}")
    print(f"\nColumns: {list(table.columns)}")

    out_path = PROCESSED_DIR / "training_table_partial.parquet"
    table.to_parquet(out_path)
    print(f"\nSaved to {out_path} (partial — weather download still running in background)")
