"""
Build the dense 15-minute target grid from sparse EAGLE-I outage events.

EAGLE-I only emits a row when customers_out > 0 for a (county, timestamp). Everything
else is implicitly zero. Both training and evaluation need the *dense* grid — every
county at every native 15-minute timestamp, zero-filled where no event row exists —
because a model trained only on nonzero rows would never see the (overwhelming
majority) negative class it must learn to predict.

x = customers_out / total_customers, clipped to [0, 1] (a handful of source rows have
customers_out slightly exceeding the modeled county customer count from MCC.csv; this
is a known MCC/EAGLE-I mismatch, not a data error to silently drop).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_acquisition.eagle_i import PROCESSED_DIR, load_max_customer_counts, load_outage_events

NATIVE_FREQ = "15min"


def build_dense_grid(
    events: pd.DataFrame,
    mcc: pd.DataFrame,
    fips_codes: list[str] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Reindex sparse outage events onto a dense 15-minute grid, zero-filled, with x computed.

    Parameters
    ----------
    events : output of load_outage_events() — sparse rows with customers_out > 0.
    mcc : output of load_max_customer_counts() — the total_customers denominator.
    fips_codes : restrict to this set of counties; defaults to every county present in
        `events` (i.e. every county that had at least one outage in the loaded years —
        NOT every US county, since counties with zero outages in the source years
        never appear as event rows at all and can't be distinguished from "not covered
        by EAGLE-I" without a separate coverage source).
    start, end : grid bounds; defaults to the min/max run_start_time in `events`.

    Returns
    -------
    fips_code, county, state, run_start_time, customers_out, total_customers, x
    """
    if fips_codes is None:
        fips_codes = sorted(events["fips_code"].unique())
    if start is None:
        start = events["run_start_time"].min()
    if end is None:
        end = events["run_start_time"].max()

    full_index = pd.date_range(start, end, freq=NATIVE_FREQ)

    county_meta = (
        events[["fips_code", "county", "state"]]
        .drop_duplicates("fips_code")
        .set_index("fips_code")
    )

    grids = []
    for fips in fips_codes:
        county_events = events.loc[events["fips_code"] == fips, ["run_start_time", "customers_out"]]
        series = (
            county_events.set_index("run_start_time")["customers_out"]
            .reindex(full_index, fill_value=0)
            .rename("customers_out")
        )
        grid = series.to_frame()
        grid["fips_code"] = fips
        grids.append(grid)

    dense = pd.concat(grids).rename_axis("run_start_time").reset_index()
    dense = dense.merge(county_meta, on="fips_code", how="left")

    mcc_lookup = mcc.set_index("fips_code")["total_customers"] if "total_customers" in mcc.columns else None
    if mcc_lookup is None:
        raise KeyError(
            f"Expected a 'total_customers' column in MCC.csv; got columns {list(mcc.columns)}. "
            f"Inspect data/raw/MCC.csv and update this lookup accordingly."
        )
    dense["total_customers"] = dense["fips_code"].map(mcc_lookup)
    dense["x"] = (dense["customers_out"] / dense["total_customers"]).clip(0, 1)

    return dense[
        ["fips_code", "county", "state", "run_start_time", "customers_out", "total_customers", "x"]
    ]


if __name__ == "__main__":
    events = load_outage_events(years=[2014])
    mcc = load_max_customer_counts()
    print(f"MCC.csv columns: {list(mcc.columns)}")

    sample_fips = sorted(events["fips_code"].unique())[:3]
    print(f"Building dense grid for sample counties {sample_fips}...")
    grid = build_dense_grid(events, mcc, fips_codes=sample_fips)

    print(f"\nGrid shape: {grid.shape}")
    print(f"Zero rows: {(grid['x'] == 0).sum()} / {len(grid)} ({(grid['x'] == 0).mean():.4%})")
    print(f"\n{grid.head(10)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "target_grid_sample_2014.parquet"
    grid.to_parquet(out_path)
    print(f"\nSaved sample to {out_path}")
