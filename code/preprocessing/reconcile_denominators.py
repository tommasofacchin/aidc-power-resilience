"""
Build a reconciled per-county total_customers — the denominator of x = customers_out /
total_customers, and therefore a multiplicative factor on every number this project
predicts.

Why this exists. The project used MCC.csv (the "modeled county customers" file shipped
with the EAGLE-I release) as the denominator throughout. MCC.csv is documented upstream
as modelled *as of 2022*, and on 20 Aug 2026 it turned out to be not merely stale but
wrong for entire states. Summed per state and checked against coverage_history.csv --
an independent file from the same EAGLE-I release, giving each state's total customer
count -- MCC reproduces the state total to within a few percent almost everywhere, but
for North Carolina it lands at 1.82M against a published 5.30M, missing roughly two
thirds of the state. Mecklenburg County (Charlotte, population 1.12M) carries 28,172
customers in MCC against 588,615 in EAGLE-I's own 2024 annual file: a factor of 20.9.
Mecklenburg is one of the five counties this team reports on, so every x for it -- in
the training target and in the submitted predictions -- was inflated ~21x.

Why not just switch to the 2024 file's column. Because it is wrong in the other
direction elsewhere. Puerto Rico's state total agrees between the two sources almost
exactly (1,485,262 vs 1,485,249), yet they allocate it differently across municipios:
Arecibo is 41,122 in MCC and 191,803 in the 2024 file, and the municipio's population is
about 87,000, so the 2024 figure exceeds the number of people living there. Neither
source is uniformly right, and a blanket swap would trade a broken NC for a broken PR.

The rule. For each state, take the county allocation from whichever candidate source
reproduces that state's independently-published total more closely, ties going to MCC as
the documented default. This is decided per state rather than per county because the
failure mode observed is a whole-state one, and because a per-county rule would have no
independent quantity to check itself against.

The second rule, and the evidence that forced it. A state total can be right while the
allocation under it is wrong, and the state rule is blind to that by construction. It
kept MCC for Puerto Rico on a 13-customer agreement in 1.48M — and EAGLE-I records up
to 139,095 customers out in Arecibo against MCC's 41,122 customers total. The numerator
is 3.4x the denominator it is divided into, so x saturated at 1 for 117 intervals in
2025 alone and the clip hid it. That is not a judgement call between two estimates: a
ratio the task defines on [0, 1] cannot exceed 1, so the observed numerator falsifies
that denominator outright. Every county therefore gets a second check against the
largest customers_out ever recorded for it, and where the state-level choice cannot
hold that numerator the other candidate is used instead. The population argument above
still stands — 191,803 customers in a municipio of ~87,000 people is not a customer
count either — which most likely means EAGLE-I's Puerto Rico rows report a service
region rather than the municipio they are keyed to. Both candidates are then wrong in
absolute terms, and the one that at least cannot be arithmetically impossible is the
only defensible choice.

The organisers keep their own fixed reference denominator and do not publish it, so this
cannot be validated against the real thing -- which is exactly why the choice is written
down here and reported, rather than left implicit in whichever file got loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_acquisition.eagle_i import PROCESSED_DIR, RAW_DIR

OUT_PATH = PROCESSED_DIR / "total_customers_reconciled.csv"
REPORT_PATH = PROCESSED_DIR / "denominator_reconciliation.md"
INFILE_CACHE = PROCESSED_DIR / "total_customers_2024_infile.parquet"
OBSERVED_MAX_CACHE = PROCESSED_DIR / "observed_max_customers_out.parquet"

# The annual file that carries a per-row total_customers column. Only 2024 does; see
# the schema-drift note in data_acquisition/eagle_i.py.
INFILE_YEAR = 2024
COVERAGE_YEAR = "1/1/22"  # latest year in coverage_history.csv

# Relative error in a state's MCC total, above which MCC is judged defective rather than
# merely stale and the 2024 allocation is used instead. See the comment at the switch.
MCC_TOLERANCE = 0.05

# Years scanned for the falsification check below. Both years the project actually
# trains and predicts on: a denominator only has to be exceeded once to be wrong.
OBSERVED_MAX_YEARS = (2024, 2025)

STATE_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "Puerto Rico": "PR",
}


def load_infile_totals(force: bool = False) -> pd.DataFrame:
    """Per-county total_customers from the 2024 annual file, cached as parquet.

    The source is a 1.4 GB CSV and the answer is ~3,000 rows, so it is read in chunks
    once and cached; re-reading it on every call would dominate the runtime of anything
    that needs the denominator.
    """
    if INFILE_CACHE.exists() and not force:
        return pd.read_parquet(INFILE_CACHE)

    path = RAW_DIR / f"eaglei_outages_{INFILE_YEAR}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required for denominator reconciliation (it is the only annual "
            f"file carrying a per-row total_customers column). Fetch it with "
            f"python -m data_acquisition.eagle_i {INFILE_YEAR}"
        )
    seen: dict[str, float] = {}
    for chunk in pd.read_csv(
        path, usecols=["fips_code", "total_customers"],
        dtype={"fips_code": str}, chunksize=2_000_000,
    ):
        chunk = chunk.dropna(subset=["total_customers"])
        for fips, tc in zip(chunk["fips_code"], chunk["total_customers"]):
            seen.setdefault(fips, tc)

    df = pd.DataFrame({"fips_code": list(seen), "infile": list(seen.values())})
    df["fips_code"] = df["fips_code"].str.zfill(5)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INFILE_CACHE)
    return df


def load_observed_max_out(force: bool = False) -> pd.DataFrame:
    """Per-county maximum customers_out ever recorded, cached as parquet.

    This is the only quantity available that can falsify a denominator without appealing
    to a third source. x = customers_out / total_customers is bounded to [0, 1] by the
    task definition, so a county whose recorded numerator exceeds a candidate
    denominator proves that candidate does not describe the population EAGLE-I counts
    against — whatever a state-level total or a census population says about it.
    """
    if OBSERVED_MAX_CACHE.exists() and not force:
        return pd.read_parquet(OBSERVED_MAX_CACHE)

    peak: dict[str, float] = {}
    for year in OBSERVED_MAX_YEARS:
        path = RAW_DIR / f"eaglei_outages_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is required for the denominator consistency check. Fetch it "
                f"with python -m data_acquisition.eagle_i {year}"
            )
        for chunk in pd.read_csv(
            path, usecols=["fips_code", "customers_out"],
            dtype={"fips_code": str}, chunksize=2_000_000,
        ):
            chunk = chunk.dropna(subset=["customers_out"])
            grouped = chunk.groupby("fips_code")["customers_out"].max()
            for fips, value in grouped.items():
                if value > peak.get(fips, -1.0):
                    peak[fips] = float(value)

    df = pd.DataFrame({"fips_code": list(peak), "observed_max": list(peak.values())})
    df["fips_code"] = df["fips_code"].str.zfill(5)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OBSERVED_MAX_CACHE)
    return df


def load_state_names() -> pd.DataFrame:
    """State FIPS prefix -> state name, read off the outage file itself."""
    path = RAW_DIR / f"eaglei_outages_{INFILE_YEAR}.csv"
    names: dict[str, str] = {}
    for chunk in pd.read_csv(
        path, usecols=["fips_code", "state"], dtype={"fips_code": str}, chunksize=2_000_000
    ):
        for fips, state in zip(chunk["fips_code"], chunk["state"]):
            names.setdefault(str(fips).zfill(5)[:2], state)
    return pd.DataFrame({"state_fips": list(names), "state_name": list(names.values())})


def reconcile() -> tuple[pd.DataFrame, pd.DataFrame]:
    mcc = pd.read_csv(RAW_DIR / "MCC.csv", dtype={"County_FIPS": str})
    mcc = mcc.rename(columns={"County_FIPS": "fips_code", "Customers": "mcc"})
    mcc["fips_code"] = mcc["fips_code"].str.zfill(5)

    infile = load_infile_totals()
    states = load_state_names()

    cov = pd.read_csv(RAW_DIR / "coverage_history.csv")
    cov = cov[cov["year"] == COVERAGE_YEAR][["state", "total_customers"]]
    cov = cov.rename(columns={"state": "abbrev", "total_customers": "published_total"})

    df = mcc.merge(infile, on="fips_code", how="outer")
    df["state_fips"] = df["fips_code"].str[:2]
    df = df.merge(states, on="state_fips", how="left")
    df["abbrev"] = df["state_name"].map(STATE_ABBREV)

    # Per-state sums of each candidate, against the published state total.
    agg = df.groupby("abbrev", dropna=True)[["mcc", "infile"]].sum().reset_index()
    agg = agg.merge(cov, on="abbrev", how="inner")
    agg["mcc_err"] = ((agg["mcc"] - agg["published_total"]).abs() / agg["published_total"])
    agg["infile_err"] = ((agg["infile"] - agg["published_total"]).abs() / agg["published_total"])
    # Override the documented default only where it demonstrably fails, not wherever it
    # is a hair worse. Comparing raw errors alone put Puerto Rico on the in-file
    # allocation over a 13-customer difference in 1.48M -- and the in-file allocation
    # gives Arecibo 191,803 customers in a municipio of ~87,000 people, so that
    # coin-flip would have corrupted a reported county to "fix" a rounding difference.
    # MCC_TOLERANCE is the band inside which MCC is treated as simply correct: 5%
    # comfortably exceeds plausible 2022->2024 customer growth, so exceeding it means a
    # real defect rather than staleness.
    agg["source"] = "mcc"
    switch = (agg["mcc_err"] > MCC_TOLERANCE) & (agg["infile_err"] < agg["mcc_err"])
    agg.loc[switch, "source"] = "infile"
    agg.loc[agg["infile"] <= 0, "source"] = "mcc"

    df = df.merge(agg[["abbrev", "source"]], on="abbrev", how="left")
    df["source"] = df["source"].fillna("mcc")
    df["total_customers"] = df["mcc"].where(df["source"] == "mcc", df["infile"])

    # Per-county falsification of the state-level choice.
    #
    # The state rule above is the right default — the failure mode it corrects is a
    # whole-state one — but it can only see sums, and a state whose total is right can
    # still allocate that total wrongly across its counties. Puerto Rico is exactly
    # that case: both candidates reproduce the state total to within 13 customers in
    # 1.48M, so the rule keeps MCC, which gives Arecibo 41,122 customers. EAGLE-I then
    # records up to 139,095 customers out in Arecibo. A ratio bounded to [0, 1] cannot
    # have a numerator 3.4x its denominator, so MCC's allocation is not merely a worse
    # estimate there, it is falsified by the numerator it will be divided into — 117
    # intervals were being clipped to x=1 to hide it.
    #
    # This check is deliberately narrow. It fires only where the data proves the
    # denominator impossible, never where the alternative merely looks nicer, so it
    # cannot quietly undo the state rule on a judgement call.
    observed = load_observed_max_out()
    df = df.merge(observed, on="fips_code", how="left")
    alternative = df["infile"].where(df["source"] == "mcc", df["mcc"])
    falsified = df["observed_max"].notna() & (df["total_customers"] < df["observed_max"])
    repairable = falsified & alternative.notna() & (alternative >= df["observed_max"])
    df.loc[repairable, "total_customers"] = alternative[repairable]
    df.loc[repairable, "source"] = df.loc[repairable, "source"] + "->consistency"
    # Neither candidate can hold the observed numerator. Take the larger of the two so
    # x is at least as close to the truth as the evidence allows, and flag it: this is
    # a county whose EAGLE-I numerator does not describe the same population either
    # candidate denominator counts, which no arithmetic here can repair.
    unresolved = falsified & ~repairable
    df.loc[unresolved, "total_customers"] = (
        df.loc[unresolved, ["mcc", "infile"]].max(axis=1, skipna=True)
    )
    df.loc[unresolved, "source"] = "unresolved"
    df.attrs["n_repaired"] = int(repairable.sum())
    df.attrs["n_unresolved"] = int(unresolved.sum())
    # A county absent from the chosen source still needs a denominator; fall back to
    # the other one rather than emitting NaN, which would silently drop the county.
    df["total_customers"] = df["total_customers"].fillna(df["mcc"]).fillna(df["infile"])
    df = df.dropna(subset=["total_customers"])
    return df, agg


def main() -> None:
    df, agg = reconcile()
    switched = agg[agg["source"] == "infile"]["abbrev"].tolist()

    print(f"Counties repaired by the observed-numerator check: {df.attrs['n_repaired']}")
    print(f"Counties where neither candidate holds the numerator: {df.attrs['n_unresolved']}")
    print(f"States reconciled against coverage_history {COVERAGE_YEAR}: {len(agg)}")
    print(f"States taking the 2024 in-file allocation instead of MCC: {switched or 'none'}\n")
    for _, r in agg[agg["abbrev"].isin(switched)].iterrows():
        print(f"  {r.abbrev}: published {r.published_total:>10,.0f} | "
              f"MCC {r.mcc:>10,.0f} | in-file {r.infile:>10,.0f}")

    out = df[["fips_code", "total_customers", "source"]].sort_values("fips_code")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    changed = df[df["source"] == "infile"].copy()
    changed["ratio"] = changed["infile"] / changed["mcc"]
    changed = changed.dropna(subset=["ratio"]).nlargest(10, "ratio")
    print(f"\nLargest denominator corrections:")
    for _, r in changed.iterrows():
        print(f"  {r.fips_code} {str(r.state_name):<16} {r.mcc:>10,.0f} -> "
              f"{r.infile:>10,.0f}  (x{r.ratio:.2f})")

    lines = [
        "# Denominator reconciliation",
        "",
        f"Per-state source selection against `coverage_history.csv` ({COVERAGE_YEAR}). "
        f"A state uses the 2024 in-file allocation only where that reproduces the "
        f"published state total more closely than MCC.csv does.",
        "",
        f"A second, per-county check then overrides that choice wherever the recorded "
        f"`customers_out` exceeds the chosen denominator, which would put x above 1: "
        f"{df.attrs['n_repaired']} counties repaired this way, "
        f"{df.attrs['n_unresolved']} where neither candidate can hold the observed "
        f"numerator (flagged `unresolved`, larger candidate used).",
        "",
        "| state | published total | MCC sum | in-file sum | chosen |",
        "|---|---|---|---|---|",
    ]
    for _, r in agg.sort_values("abbrev").iterrows():
        lines.append(f"| {r.abbrev} | {r.published_total:,.0f} | {r.mcc:,.0f} | "
                     f"{r.infile:,.0f} | {r.source} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf8")

    print(f"\nWrote {OUT_PATH.name} ({len(out):,} counties) and {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
