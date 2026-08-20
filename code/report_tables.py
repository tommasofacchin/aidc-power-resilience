"""
Per-county activity statistics for the five reporting counties, as quoted in section 2
of the technical report.

Separate from preprocessing/select_counties.py on purpose. That module ranked all ~3,050
EAGLE-I counties to make the selection, and it ran BEFORE the denominator problem in
preprocessing/reconcile_denominators.py was found, so every x it computed used raw MCC
counts. Mecklenburg's were wrong by 20.9x. Re-running the whole ranking would cost a
full pass over the raw annual file for a decision already made and already defended on
regime-coverage grounds; recomputing the five selected counties' statistics on the
corrected denominators is what the report actually needs, and is cheap.

Reads the 2025 annual EAGLE-I file (~1.4 GB, a few minutes) and writes
data/processed/report_county_profile.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_acquisition.eagle_i import PROCESSED_DIR, load_outage_events, load_total_customers
from preprocessing.build_target_grid import build_dense_grid

WINDOW_START = pd.Timestamp("2025-01-01")
WINDOW_END = pd.Timestamp("2025-09-01")  # exclusive: stops before the evaluated window
SIGNIFICANT_X = 0.01
OUT_PATH = PROCESSED_DIR / "report_county_profile.csv"


def main() -> None:
    counties = pd.read_csv(PROCESSED_DIR / "selected_counties.csv", dtype={"fips_code": str})
    fips_codes = counties["fips_code"].tolist()

    events = load_outage_events(years=[2025])
    events = events[events["fips_code"].isin(fips_codes)]
    grid = build_dense_grid(
        events, load_total_customers(), fips_codes=fips_codes,
        start=WINDOW_START, end=WINDOW_END - pd.Timedelta(minutes=15),
    )

    rows = []
    for fips, g in grid.groupby("fips_code"):
        rows.append({
            "fips_code": fips,
            "county": counties.loc[counties.fips_code == fips, "county"].iloc[0],
            "total_customers": int(g["total_customers"].iloc[0]),
            "n_intervals": len(g),
            "any_outage_rate": float((g["x"] > 0).mean()),
            "significant_rate": float((g["x"] > SIGNIFICANT_X).mean()),
            "mean_x": float(g["x"].mean()),
            "p99_x": float(g["x"].quantile(0.99)),
            "max_x": float(g["x"].max()),
        })

    profile = pd.DataFrame(rows).sort_values("significant_rate", ascending=False)
    profile.to_csv(OUT_PATH, index=False)
    print(profile.to_string(index=False))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
