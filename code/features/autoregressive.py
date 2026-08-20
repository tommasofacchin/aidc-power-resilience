"""
Autoregressive EAGLE-I features: the observed outage state up to and including
issue_time. Per PLAN.md section 2.3, this is the main source of skill for Task B (the
15-minute-resolution weather input is interpolated from hourly, so it doesn't carry
genuine sub-hourly signal on its own — the autoregressive state does).

Legitimate per the submission guidelines (section 3.3): only the *weather* input is
restricted to what a forecast could have provided at issue_time. Observed outage
history up to issue_time is realistic (EAGLE-I/ODIN is near-real-time) and not
leakage — see PLAN.md section 2.4.

Requires a dense (zero-filled) per-county 15-minute grid, sorted by (fips_code,
run_start_time), as produced by preprocessing.build_target_grid.build_dense_grid — the
lag/rolling operations below assume a complete, regular time index per county and will
silently produce wrong offsets (real gaps read as if they were smaller) if the grid has
holes.
"""

from __future__ import annotations

import pandas as pd

# 15-minute native resolution: 4 periods/hour
LAG_PERIODS = {
    "x_lag_15m": 1,
    "x_lag_30m": 2,
    "x_lag_1h": 4,
    "x_lag_2h": 8,
    "x_lag_6h": 24,
}
ROLLING_MAX_WINDOW_24H = 96
ONGOING_OUTAGE_THRESHOLD = 1e-9  # x > 0, guarding against float noise at exactly 0


def add_autoregressive_features(dense_grid: pd.DataFrame) -> pd.DataFrame:
    """Add lag, trend, streak, and rolling-max features to a dense per-county x grid.

    Every added column is, by construction, a function only of rows at or before the
    row's own run_start_time within the same county — there is nothing here that needs
    a separate causality check when this grid is later joined at issue_time = the row's
    own run_start_time (the case for training). If a caller instead joins these features
    at a *different*, later issue_time, they must re-verify causality themselves — see
    code/features/causality.py.
    """
    df = dense_grid.sort_values(["fips_code", "run_start_time"]).reset_index(drop=True)
    grouped_x = df.groupby("fips_code")["x"]

    for col, periods in LAG_PERIODS.items():
        df[col] = grouped_x.shift(periods)

    df["x_trend_1h"] = df["x"] - df["x_lag_1h"]
    df["x_trend_2h"] = df["x"] - df["x_lag_2h"]

    df["x_max_24h"] = grouped_x.rolling(ROLLING_MAX_WINDOW_24H, min_periods=1).max().reset_index(
        level=0, drop=True
    )

    is_out = df["x"] > ONGOING_OUTAGE_THRESHOLD
    county_change = df["fips_code"] != df["fips_code"].shift()
    streak_id = (is_out.ne(is_out.shift()) | county_change).cumsum()
    duration = df.groupby(streak_id).cumcount() + 1
    df["outage_duration_15m_periods"] = duration.where(is_out, 0)
    df["ongoing_outage"] = is_out

    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_acquisition.eagle_i import load_total_customers, load_outage_events
    from preprocessing.build_target_grid import build_dense_grid

    events = load_outage_events(years=[2025])
    mcc = load_total_customers()

    # Quick correctness check on the 5 reporting counties, restricted to a short window
    # so it runs fast — the full training table will call this on the whole grid.
    sample_fips = ["72013", "37119", "54005", "26097", "22071"]
    events_sample = events[
        (events["fips_code"].isin(sample_fips))
        & (events["run_start_time"] >= "2025-08-01")
        & (events["run_start_time"] < "2025-08-08")
    ]
    grid = build_dense_grid(
        events_sample, mcc, fips_codes=sample_fips,
        start=pd.Timestamp("2025-08-01"), end=pd.Timestamp("2025-08-08"),
    )
    featured = add_autoregressive_features(grid)

    print(f"Rows: {len(featured)}")
    print(f"Columns added: {[c for c in featured.columns if c not in grid.columns]}")

    # Sanity check: pick a row with an ongoing outage and manually verify x_lag_1h.
    active = featured[featured["ongoing_outage"]].head(1)
    if len(active):
        row = active.iloc[0]
        fips, t = row["fips_code"], row["run_start_time"]
        expected = grid[(grid["fips_code"] == fips) & (grid["run_start_time"] == t - pd.Timedelta(hours=1))]
        print(f"\nSpot check fips={fips} t={t}:")
        print(f"  x={row['x']:.4f}, x_lag_1h={row['x_lag_1h']}, "
              f"expected (grid lookup)={expected['x'].iloc[0] if len(expected) else 'N/A'}")
        print(f"  outage_duration_15m_periods={row['outage_duration_15m_periods']}, "
              f"x_max_24h={row['x_max_24h']:.4f}")
