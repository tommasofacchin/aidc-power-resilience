"""
Select a stratified sample of counties for training (distinct from the 5 counties in
select_counties.py, which are what we actually report on). Training on far more than 5
counties is deliberate (PLAN.md section 2.5): it's the main source of positive-event
volume and spatial generalization, given we only have one year (2025) of ground truth
instead of the multi-year history originally planned.

Downloading IFS weather for all ~3,048 EAGLE-I-covered counties over 8 months is not
worth the time budget (rate-limited at some per-minute cap on the free Open-Meteo
tier — confirmed empirically, not documented precisely). A stratified sample gives most
of the generalization benefit for a fraction of the download cost.

Stratification: US Census region x outage-activity tercile (from the Jan-Aug 2025
climatology), capped per state so no single state dominates a cell (Texas alone has 151
eligible counties — sampling naively would make this a Texas-weather model with a
long tail).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SAMPLE_SIZE = 250
MAX_PER_STATE_PER_CELL = 4
RANDOM_STATE = 42

# The 5 counties selected in select_counties.py are always included on top of this
# sample — they need training coverage too, so no double bookkeeping is needed
# downstream, just don't drop them if the stratified draw doesn't happen to pick them.
FORCE_INCLUDE_FIPS = {"72013", "37119", "54005", "26097", "22071"}

CENSUS_REGION = {
    # Northeast
    **{s: "Northeast" for s in [
        "Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont",
        "New Jersey", "New York", "Pennsylvania",
    ]},
    # Midwest
    **{s: "Midwest" for s in [
        "Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin",
        "Iowa", "Kansas", "Minnesota", "Missouri", "Nebraska", "North Dakota", "South Dakota",
    ]},
    # South
    **{s: "South" for s in [
        "Delaware", "Florida", "Georgia", "Maryland", "North Carolina", "South Carolina",
        "Virginia", "West Virginia", "District of Columbia",
        "Alabama", "Kentucky", "Mississippi", "Tennessee",
        "Arkansas", "Louisiana", "Oklahoma", "Texas",
    ]},
    # West
    **{s: "West" for s in [
        "Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah", "Wyoming",
        "Alaska", "California", "Hawaii", "Oregon", "Washington",
    ]},
    # Territories — not a Census region, own bucket so they don't get silently dropped
    "Puerto Rico": "Territories",
    "United States Virgin Islands": "Territories",
}


def stratified_sample(climatology: pd.DataFrame) -> pd.DataFrame:
    pool = climatology[
        (climatology["total_customers"] >= 5_000) & (climatology["span_days"] >= 200)
    ].copy()
    pool["region"] = pool["state"].map(CENSUS_REGION)
    unmapped = pool[pool["region"].isna()]["state"].unique()
    if len(unmapped):
        raise ValueError(f"States missing from CENSUS_REGION: {sorted(unmapped)}")

    pool["activity_tier"] = pd.qcut(
        pool["significant_rate"].rank(method="first"), 3, labels=["low", "mid", "high"]
    )

    rng = np.random.default_rng(RANDOM_STATE)

    # Cap per (state, cell) first, so no state can flood a stratum.
    capped_parts = []
    for _, group in pool.groupby(["region", "activity_tier", "state"], observed=True):
        n = min(len(group), MAX_PER_STATE_PER_CELL)
        capped_parts.append(group.sample(n=n, random_state=RANDOM_STATE))
    capped = pd.concat(capped_parts)

    cell_sizes = capped.groupby(["region", "activity_tier"], observed=True).size()
    cell_weights = cell_sizes / cell_sizes.sum()
    target_per_cell = (cell_weights * SAMPLE_SIZE).round().astype(int)

    sampled_parts = []
    for (region, tier), group in capped.groupby(["region", "activity_tier"], observed=True):
        n = min(len(group), target_per_cell.get((region, tier), 0))
        if n > 0:
            sampled_parts.append(group.sample(n=n, random_state=RANDOM_STATE))
    sample = pd.concat(sampled_parts)

    # Guarantee the 5 reporting counties are present even if the stratified draw missed them.
    forced = climatology[climatology["fips_code"].isin(FORCE_INCLUDE_FIPS)]
    sample = pd.concat([sample, forced]).drop_duplicates("fips_code")

    return sample.sort_values(["region", "activity_tier", "state"])


if __name__ == "__main__":
    climatology = pd.read_parquet(PROCESSED_DIR / "county_climatology_jan_aug_2025.parquet")
    sample = stratified_sample(climatology)

    print(f"Training sample: {len(sample)} counties")
    print("\nBy region:")
    print(sample["region"].value_counts())
    print("\nBy activity tier:")
    print(sample["activity_tier"].value_counts())
    print(f"\nStates represented: {sample['state'].nunique()}")
    print(f"5 reporting counties included: {FORCE_INCLUDE_FIPS.issubset(set(sample['fips_code']))}")

    out_path = PROCESSED_DIR / "training_counties.csv"
    sample[["fips_code", "county", "state", "region", "activity_tier", "total_customers"]].to_csv(
        out_path, index=False
    )
    print(f"\nSaved to {out_path}")
