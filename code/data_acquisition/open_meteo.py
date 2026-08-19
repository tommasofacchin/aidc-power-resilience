"""
Open-Meteo Single Runs API client — archived ECMWF IFS HRES model initializations.

This is the ONLY weather source allowed for generating submitted predictions (see
submission_guidelines_phase1-AIDC.docx section 3.3): it retrieves the forecast that
was genuinely available at a specific past issue_time, rather than stitching together
observed/reanalysis data that wouldn't have existed yet. The Historical Forecast API
and the plain Weather Forecast API are both unsuitable for this and must not be used
for prediction-time features.

Archive coverage for models=ecmwf_ifs starts 2024-03-14 (verified in PLAN.md); other
Single Runs models only go back to 2026-04-02, which does not overlap the Phase 1
test/training window, so ecmwf_ifs is the only usable model here.

Every response is cached to disk (sqlite via requests-cache) so a given (coordinates,
run, variables) request is only ever fetched once.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests_cache

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = PROJECT_ROOT / "data" / "raw" / "open_meteo_cache"

BASE_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
MODEL = "ecmwf_ifs"
ARCHIVE_START = pd.Timestamp("2024-03-14")

DEFAULT_HOURLY_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "precipitation",
    "rain",
    "snowfall",
    "cape",
    "cloud_cover",
    "freezing_level_height",
]

class RateLimitError(Exception):
    """Raised on HTTP 429 from the Single Runs API so callers can back off and retry."""


_session = requests_cache.CachedSession(
    str(CACHE_PATH),
    backend="sqlite",
    expire_after=-1,  # archived model runs never change: cache forever
    allowable_codes=(200,),  # never cache 429s — a retry must hit the live server
)


def fetch_single_run(
    locations: pd.DataFrame,
    run_time: pd.Timestamp,
    hourly_vars: list[str] | None = None,
    forecast_hours: int | None = None,
) -> pd.DataFrame:
    """Fetch one archived ECMWF IFS HRES run for one or more locations.

    Parameters
    ----------
    locations : DataFrame with columns ["fips_code", "latitude", "longitude"] (one row
        per county). Order is preserved and used to re-attach fips_code to the
        response, since the API keys results by lat/lon (snapped to the model grid),
        not by any id we supply.
    run_time : the model initialization time (must be a valid ecmwf_ifs run: 00/06/12/18
        UTC), as a tz-naive UTC Timestamp.
    hourly_vars : variables to request; defaults to DEFAULT_HOURLY_VARS.

    Returns
    -------
    Tidy long DataFrame: fips_code, run_time, time, variable, value.
    """
    if run_time < ARCHIVE_START:
        raise ValueError(
            f"run_time {run_time} predates the ecmwf_ifs Single Runs archive "
            f"({ARCHIVE_START}); no data will be returned."
        )
    hourly_vars = hourly_vars or DEFAULT_HOURLY_VARS

    params = {
        "latitude": ",".join(f"{lat:.4f}" for lat in locations["latitude"]),
        "longitude": ",".join(f"{lon:.4f}" for lon in locations["longitude"]),
        "run": run_time.strftime("%Y-%m-%dT%H:%M"),
        "models": MODEL,
        "hourly": ",".join(hourly_vars),
    }
    if forecast_hours is not None:
        params["forecast_hours"] = forecast_hours

    resp = _session.get(BASE_URL, params=params, timeout=90)
    if resp.status_code == 429:
        raise RateLimitError(resp.json().get("reason", "rate limited"))
    resp.raise_for_status()
    payload = resp.json()

    # Single-location requests return one object; multi-location returns a list.
    results = payload if isinstance(payload, list) else [payload]
    if len(results) != len(locations):
        raise RuntimeError(
            f"Requested {len(locations)} locations but API returned {len(results)} — "
            f"response order may not match the input order."
        )

    frames = []
    for (_, loc_row), result in zip(locations.iterrows(), results):
        hourly = result["hourly"]
        df = pd.DataFrame(hourly)
        df = df.melt(id_vars=["time"], var_name="variable", value_name="value")
        df["fips_code"] = loc_row["fips_code"]
        df["run_time"] = run_time
        df["time"] = pd.to_datetime(df["time"], utc=True)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out[["fips_code", "run_time", "time", "variable", "value"]]


if __name__ == "__main__":
    test_locations = pd.DataFrame(
        {
            "fips_code": ["01001", "36061"],
            "latitude": [32.5322, 40.7128],
            "longitude": [-86.6449, -74.0060],
        }
    )
    df = fetch_single_run(test_locations, pd.Timestamp("2025-09-15T00:00"))
    print(f"Fetched {len(df)} rows for {df['fips_code'].nunique()} counties")
    print(df.head(10))
    print(f"\nVariables: {sorted(df['variable'].unique())}")
    print(f"Time range: {df['time'].min()} .. {df['time'].max()}")
    print(f"\nCache file: {CACHE_PATH}.sqlite")
