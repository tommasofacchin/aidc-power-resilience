"""
Weather features from archived ECMWF IFS HRES Single Runs (PLAN.md section 3.5).

Every derived feature here is computed *within* a single (fips_code, run_time) forecast
trajectory — i.e. from hours that were all disseminated together as one forecast. That
makes centered windows (looking at forecast hours both before and after a given target
hour) causally fine: causality is about issue_time vs. run_time, not about target_time
vs. neighboring forecast hours in the same already-available run. See
code/features/causality.py for the actual issue_time boundary check, applied where
run_time itself is compared against issue_time (in the training table assembly, not
here).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEATHER_RUNS_DIR = PROJECT_ROOT / "data" / "raw" / "ifs_training_runs"

GUST_COL = "wind_gusts_10m"
PRECIP_COL = "precipitation"
PRESSURE_COL = "surface_pressure"

CLIMATOLOGY_QUANTILES = {"p95": 0.95, "p99": 0.99, "p999": 0.999}


def load_weather_runs(run_dir: Path = WEATHER_RUNS_DIR) -> pd.DataFrame:
    """Concatenate every downloaded run's long-format parquet into one DataFrame."""
    paths = sorted(run_dir.glob("run_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No weather run files found in {run_dir}")
    return pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)


def pivot_weather(long_df: pd.DataFrame) -> pd.DataFrame:
    """Long (fips_code, run_time, time, variable, value) -> wide, one row per forecast hour."""
    wide = long_df.pivot_table(
        index=["fips_code", "run_time", "time"], columns="variable", values="value"
    ).reset_index()
    wide.columns.name = None
    wide = wide.sort_values(["fips_code", "run_time", "time"]).reset_index(drop=True)

    # The API returns some variables (e.g. cloud_cover) as JSON ints and others as
    # floats, plus nulls for the accumulator-at-lead-0 case (see module docstring) —
    # that mix makes pandas infer `value` as dtype object all the way through the
    # pivot. Harmless for elementwise arithmetic (gust_cubed etc. still "worked") but
    # breaks vectorized ops like groupby().quantile() outright. Force real numeric
    # dtype once, here, rather than downstream wherever it happens to bite.
    weather_var_cols = [c for c in wide.columns if c not in ("fips_code", "run_time", "time")]
    wide[weather_var_cols] = wide[weather_var_cols].apply(pd.to_numeric, errors="coerce")
    # `time` comes back tz-aware (UTC) from the API; `run_time` is whatever tz-naive
    # UTC Timestamp the caller passed to fetch_single_run. Align before subtracting.
    run_time_utc = wide["run_time"]
    if run_time_utc.dt.tz is None:
        run_time_utc = run_time_utc.dt.tz_localize("UTC")
    wide["lead_hours"] = (wide["time"] - run_time_utc).dt.total_seconds() / 3600
    return wide


def add_derived_weather_features(wide: pd.DataFrame) -> pd.DataFrame:
    """Add the section-3.5 derived features: gust^3, rolling extremes, cumulative
    precip, wind x rain interaction, temporal gradients."""
    df = wide.copy()
    grp_key = ["fips_code", "run_time"]

    df["gust_cubed"] = df[GUST_COL] ** 3
    df["wind_rain_interaction"] = df[GUST_COL] * df[PRECIP_COL]

    for hours, window in [(3, 7), (6, 13), (12, 25)]:
        # window sized so `hours` on each side of the target (hourly steps, centered).
        gust_grp = df.groupby(grp_key)[GUST_COL]
        df[f"gust_roll_max_{hours}h"] = gust_grp.rolling(
            window, center=True, min_periods=1
        ).max().reset_index(level=[0, 1], drop=True)

    precip_grp = df.groupby(grp_key)[PRECIP_COL]
    df["precip_cum_6h"] = precip_grp.rolling(6, min_periods=1).sum().reset_index(
        level=[0, 1], drop=True
    )
    df["precip_cum_24h"] = precip_grp.rolling(24, min_periods=1).sum().reset_index(
        level=[0, 1], drop=True
    )

    df["pressure_change_3h"] = df.groupby(grp_key)[PRESSURE_COL].diff(3)
    df["gust_jump_1h"] = df.groupby(grp_key)[GUST_COL].diff(1)

    return df


def fit_climatology(wide: pd.DataFrame, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """FIT per-county gust/precip quantiles. Call this once, on training data only.

    `cutoff` (exclusive, against run_time) restricts the fit to the training portion of
    a temporal split, so the quantiles don't absorb information from the validation
    period — the features derived from them would otherwise carry a mild optimistic
    bias into validation scores.

    Climatology basis is the ~8-month 2025 training window, NOT a true long-term
    climate normal: no multi-year weather archive is reachable under the ecmwf_ifs
    Single Runs constraint (archive starts 2024-03-14). Documented limitation
    (PLAN.md section 2.5), not an oversight.

    Returns a DataFrame indexed by fips_code with the *_clim_* threshold columns.
    """
    src = wide if cutoff is None else wide[wide["run_time"] < cutoff]
    if src.empty:
        raise ValueError(f"No weather rows before cutoff={cutoff}; cannot fit climatology.")

    parts = []
    for var, prefix in [(GUST_COL, "gust"), (PRECIP_COL, "precip")]:
        q = src.groupby("fips_code")[var].quantile(list(CLIMATOLOGY_QUANTILES.values())).unstack()
        q.columns = [f"{prefix}_clim_{name}" for name in CLIMATOLOGY_QUANTILES]
        parts.append(q)
    return pd.concat(parts, axis=1)


def add_climatology_features(wide: pd.DataFrame, climatology: pd.DataFrame) -> pd.DataFrame:
    """APPLY an already-fitted climatology (from fit_climatology).

    `climatology` is deliberately a required argument, not an optional one that
    defaults to fitting on the spot. Fitting on whatever happens to be in scope is
    exactly the bug this signature exists to make impossible: at prediction time the
    in-scope weather is a single ~60-hour batch, whose p95 gust is roughly half the
    training p95 (verified: Orleans LA 26.4 vs 54.4), which silently redefines every
    exceedance feature between training and serving.
    """
    missing = set(climatology.columns) & set(wide.columns)
    if missing:
        raise ValueError(
            f"`wide` already carries climatology columns {sorted(missing)} — "
            f"applying twice would produce _x/_y suffixed duplicates."
        )

    df = wide.merge(climatology, on="fips_code", how="left")
    for var, prefix in [(GUST_COL, "gust"), (PRECIP_COL, "precip")]:
        for name in CLIMATOLOGY_QUANTILES:
            df[f"{prefix}_exceeds_{name}"] = df[var] > df[f"{prefix}_clim_{name}"]
    return df


def interpolate_to_15min(wide: pd.DataFrame) -> pd.DataFrame:
    """Fill in the 15-minute target_times Task B needs, by linear interpolation
    between the hourly forecast points within each (fips_code, run_time) trajectory.

    IFS HRES has no native minutely_15 output (see PLAN.md section 2.3 — only HRRR/
    ICON-D2/AROME do, and those Single Runs archives don't cover the training/test
    window). This is the interpolation the plan already flags as necessary; genuine
    sub-hourly weather signal isn't available at all here, only a smoothed estimate —
    the actual sub-hourly timing skill for Task B is expected to come from the
    autoregressive EAGLE-I features, not this interpolation.

    Applied to the fully-featured (derived + climatology) wide table rather than to
    the raw variables before deriving features: simpler, and treats the whole feature
    trajectory as one smooth curve to resample rather than re-deriving rolling windows
    at two different step sizes.
    """
    non_numeric = ["fips_code", "run_time", "time"]
    numeric_cols = [c for c in wide.columns if c not in non_numeric]

    parts = []
    for (fips, run_time), group in wide.groupby(["fips_code", "run_time"]):
        g = group.set_index("time").sort_index()
        fine_index = pd.date_range(g.index.min(), g.index.max(), freq="15min")
        # Boolean exceedance flags (gust_exceeds_p95 etc.) interpolate fine once cast
        # to float — pandas' interpolate() otherwise rejects bool/object dtypes
        # outright. A fractional result (e.g. 0.3) is a reasonable soft signal for
        # "close to crossing this threshold" at the interpolated instant.
        g_numeric = g[numeric_cols].astype(float)
        g_fine = g_numeric.reindex(g.index.union(fine_index)).interpolate(method="time").loc[fine_index]
        g_fine["fips_code"] = fips
        g_fine["run_time"] = run_time
        parts.append(g_fine.reset_index(names="time"))

    return pd.concat(parts, ignore_index=True)


def build_weather_base(run_dir: Path = WEATHER_RUNS_DIR) -> pd.DataFrame:
    """Everything except climatology, which needs a separate fit/apply step so the
    training-fitted thresholds can be reused at prediction time."""
    long_df = load_weather_runs(run_dir)
    wide = pivot_weather(long_df)
    return add_derived_weather_features(wide)


def build_weather_features(
    climatology: pd.DataFrame, run_dir: Path = WEATHER_RUNS_DIR
) -> pd.DataFrame:
    return add_climatology_features(build_weather_base(run_dir), climatology)


if __name__ == "__main__":
    base = build_weather_base()
    climatology = fit_climatology(base)
    features = add_climatology_features(base, climatology)

    print(f"Rows: {len(features)}, columns: {len(features.columns)}")
    print(f"Counties: {features['fips_code'].nunique()}, runs: {features['run_time'].nunique()}")
    print(f"\nColumns: {list(features.columns)}")

    sample = features[
        (features["fips_code"] == features["fips_code"].iloc[0])
        & (features["lead_hours"].between(10, 15))
    ]
    cols = ["run_time", "time", "lead_hours", GUST_COL, "gust_cubed", "gust_roll_max_3h",
            "precip_cum_6h", "gust_exceeds_p95"]
    print(f"\nSample rows:\n{sample[cols].head(10).to_string(index=False)}")
