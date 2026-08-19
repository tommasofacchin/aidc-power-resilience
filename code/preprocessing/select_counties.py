"""
Select the 5 counties this team reports on for the whole test window (submission
guidelines section 3.2: "each team selects its own 5 counties from those covered by
EAGLE-I ... document and justify your county selection in the technical report").

Climatology basis: Jan-Aug 2025, NOT Sep-Nov 2025 (the evaluated test window) and not
2014-2024 (inaccessible — see PLAN.md section 2.6 for why the criteria below are a
proxy for the originally-intended decade-long climatology, not the real thing).

Criteria, in the order applied:
1. Outage activity: enough nonzero x events in Jan-Aug 2025 to be a meaningful
   forecasting target at all (a county with zero activity all year gives the model
   nothing to learn and nothing to score).
2. Coverage continuity: EAGLE-I must actually be reporting for this county throughout
   the window, not silently missing chunks (a coverage gap looks identical to "no
   outage" in the sparse event log, which would corrupt both x and any climatology
   built on top of it).
3. total_customers not tiny: very small counties make x noisy (one household out of
   200 already swings x by 0.5%).
4. Weather-regime diversity across the final 5: not all coastal, not all inland.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_acquisition.eagle_i import PROCESSED_DIR, load_max_customer_counts, load_outage_events

CLIMATOLOGY_START = pd.Timestamp("2025-01-01")
CLIMATOLOGY_END = pd.Timestamp("2025-09-01")  # exclusive — stops before the test window
NATIVE_FREQ_PER_DAY = 96  # 15-minute native resolution
MIN_TOTAL_CUSTOMERS = 5_000  # below this, x is dominated by single-household noise


SIGNIFICANT_X_THRESHOLD = 0.01  # 1% of customers out — a real event, not background trickle

# Final selection (see PLAN.md section 2.6 and the technical report for the full
# rationale). Not a pure top-N by significant_rate: Gulf/Atlantic hurricane-exposed
# counties score low on that metric simply because peak hurricane season (Aug-Oct)
# falls mostly inside the held-out test window, not because that regime is
# unimportant — Orleans is included deliberately to cover it despite a thinner
# Jan-Aug signal, rather than let the climatology proxy silently exclude the regime
# the challenge is arguably most about.
SELECTED_COUNTIES = {
    "72013": ("Arecibo", "Puerto Rico", "Tropical/Caribbean — chronically fragile grid, highest signal in-window"),
    "37119": ("Mecklenburg", "North Carolina", "Piedmont/Southeast — highest mainland signal, hurricane remnants + severe convection"),
    "54005": ("Boone", "West Virginia", "Appalachian/interior — ice storms + severe thunderstorms"),
    "26097": ("Mackinac", "Michigan", "Great Lakes — winter storms, repeated near-total local outages (max_x=0.87)"),
    "22071": ("Orleans", "Louisiana", "Gulf Coast/hurricane — included deliberately for regime coverage despite thin Jan-Aug signal; peak season falls in the test window"),
}


def compute_county_climatology() -> pd.DataFrame:
    """One row per county: activity, coverage, and size stats over Jan-Aug 2025.

    Ranking on raw nonzero-row count is a trap: it just re-discovers the biggest,
    most populous counties, because a large county (Harris TX, LA county, ...) almost
    always has a *handful* of customers out somewhere from routine local faults —
    nonzero, but not the weather-driven risk spike this challenge is actually about.
    n_event_rows near 100% of intervals confirms this (verified on the first pass:
    Duval FL, Philadelphia PA, Harris TX all sat at ~99.9% nonzero — a permanent
    baseline trickle, not extreme-weather activity). Ranking instead on how often x
    crosses SIGNIFICANT_X_THRESHOLD separates "always a little bit broken" from
    "genuinely knocked out."
    """
    events = load_outage_events(years=[2025])
    events = events[
        (events["run_start_time"] >= CLIMATOLOGY_START) & (events["run_start_time"] < CLIMATOLOGY_END)
    ]
    mcc = load_max_customer_counts()
    events = events.merge(mcc, on="fips_code", how="left")
    events["x"] = (events["customers_out"] / events["total_customers"]).clip(0, 1)

    n_days = (CLIMATOLOGY_END - CLIMATOLOGY_START).days
    expected_intervals = n_days * NATIVE_FREQ_PER_DAY

    per_county = (
        events.groupby("fips_code")
        .agg(
            county=("county", "first"),
            state=("state", "first"),
            total_customers=("total_customers", "first"),
            n_event_rows=("customers_out", "size"),
            max_x=("x", "max"),
            p99_x=("x", lambda s: s.quantile(0.99)),
            n_significant_intervals=("x", lambda s: (s > SIGNIFICANT_X_THRESHOLD).sum()),
            first_event=("run_start_time", "min"),
            last_event=("run_start_time", "max"),
        )
        .reset_index()
    )

    # Coverage proxy: how much of the climatology window this county has ANY row
    # activity spanning (a crude but cheap signal — a county with events only in the
    # first two weeks and nothing after is more likely a reporting gap than eight
    # months of genuine silence).
    per_county["span_days"] = (per_county["last_event"] - per_county["first_event"]).dt.days
    per_county["event_rate"] = per_county["n_event_rows"] / expected_intervals
    per_county["significant_rate"] = per_county["n_significant_intervals"] / expected_intervals

    return per_county.sort_values("n_significant_intervals", ascending=False)


if __name__ == "__main__":
    climatology = compute_county_climatology()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "county_climatology_jan_aug_2025.parquet"
    climatology.to_parquet(out_path)

    print(f"Computed climatology for {len(climatology)} counties over "
          f"{CLIMATOLOGY_START.date()} .. {CLIMATOLOGY_END.date()}")
    print(f"Saved to {out_path}\n")

    eligible = climatology[climatology["total_customers"] >= MIN_TOTAL_CUSTOMERS]
    # span_days >= 200 out of 243 excludes counties whose activity looks front- or
    # back-loaded within the window (a likely coverage gap rather than genuine
    # eight-month quiet).
    eligible = eligible[eligible["span_days"] >= 200]
    print(f"Counties with total_customers >= {MIN_TOTAL_CUSTOMERS:,} and span_days >= 200: {len(eligible)}")

    print("\nTop 20 by count of significant intervals (x > 1%) — candidate pool:")
    cols = [
        "fips_code", "county", "state", "total_customers",
        "n_significant_intervals", "significant_rate", "max_x", "p99_x", "span_days",
    ]
    print(eligible[cols].head(20).to_string(index=False))

    print(f"\n=== Final selection ({len(SELECTED_COUNTIES)} counties) ===")
    selected = climatology[climatology["fips_code"].isin(SELECTED_COUNTIES)].copy()
    selected["rationale"] = selected["fips_code"].map(lambda f: SELECTED_COUNTIES[f][2])
    print(selected[cols + ["rationale"]].to_string(index=False))

    selected_out = PROCESSED_DIR / "selected_counties.csv"
    selected[["fips_code", "county", "state", "total_customers", "rationale"]].to_csv(
        selected_out, index=False
    )
    print(f"\nSaved final selection to {selected_out}")
