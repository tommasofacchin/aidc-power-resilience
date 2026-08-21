"""
EAGLE-I outage data acquisition.

Every file this project needs lives in a single figshare record — article 24237376,
"The Environment for Analysis of Geo-Located Energy Information's Recorded Electricity
Outages 2014-2025" (CC BY 4.0), surfaced by share token 417a4f147cf1357a5391. There is
no login step anywhere: each file is a plain anonymous HTTPS GET against
https://ndownloader.figshare.com/files/<id>.

The ORNL Globus collection the organisers also hand out (DOI 10.13139/ORNLNCCS/1975202)
covers only 2014-2022 — a strict subset of the record above — so its interactive OAuth
login is never worth paying. Re-derive the id table below at any time with:

    curl -s https://api.figshare.com/v2/articles/24237376

Annual files run 78 MB (2014, which only starts in November) to ~1.4 GB, ~11.6 GB for
the full set, so they are deliberately NOT part of download_support_files(); ask for
them by year via download_year_files().

Everything downloaded or derived is cached under data/raw and data/processed so the
figshare calls happen at most once.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FIGSHARE_ARTICLE = 24237376
_NDOWNLOADER = "https://ndownloader.figshare.com/files/{file_id}"

# Small companion files, ~57 KB all told — cheap enough to always fetch as a group.
FIGSHARE_SUPPORT_FILES = {
    "MCC.csv": 42547708,
    "DQI.csv": 42547705,
    "coverage_history.csv": 42547714,
}

# Annual outage files, year -> figshare file id. All twelve verified reachable
# anonymously on 20 Aug 2026 (2019/2023/2024 spot-checked by range request).
FIGSHARE_YEAR_FILES = {
    2014: 42547717,
    2015: 42547822,
    2016: 42547825,
    2017: 42547828,
    2018: 42547879,
    2019: 42547885,
    2020: 42547894,
    2021: 42547891,
    2022: 42547897,
    2023: 44574907,
    2024: 53581661,
    2025: 62164877,
}

# The annual files are NOT schema-stable, and the drift is silent rather than fatal:
#   2014-2022, 2025 : fips_code,county,state,customers_out,run_start_time
#   2023            : the fourth column is named `sum`, not `customers_out`
#   2024            : carries an extra sixth column, total_customers
# Concatenating the raw frames therefore hands you an all-NaN customers_out for the
# whole of 2023 without raising anything. Headers are normalised at read time instead.
CANONICAL_COLUMNS = ["fips_code", "county", "state", "customers_out", "run_start_time"]
_COLUMN_ALIASES = {"sum": "customers_out"}

# Rows per read chunk in load_outage_events. Only relevant to peak memory, not to the
# result: 5M rows of these five columns is a few hundred MB, small enough that two
# annual files can be read back to back inside 16 GB.
READ_CHUNK_ROWS = 5_000_000

_YEAR_FILE_RE = re.compile(r"eaglei_outages_(\d{4})\.csv$", re.IGNORECASE)


def _fetch(file_id: int, dest: Path, force: bool = False) -> Path:
    """Stream one figshare file to dest, skipping the download if it is already there."""
    if dest.exists() and not force:
        return dest
    with requests.get(_NDOWNLOADER.format(file_id=file_id), stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest


def download_support_files(force: bool = False) -> list[Path]:
    """Download MCC.csv, DQI.csv and coverage_history.csv into RAW_DIR."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return [
        _fetch(file_id, RAW_DIR / filename, force)
        for filename, file_id in FIGSHARE_SUPPORT_FILES.items()
    ]


def download_year_files(years: list[int], force: bool = False) -> list[Path]:
    """Download the given annual outage files into RAW_DIR.

    Kept separate from download_support_files() because these are large: budget roughly
    0.6-1.4 GB per year, ~11.6 GB for all twelve.
    """
    unknown = sorted(set(years) - set(FIGSHARE_YEAR_FILES))
    if unknown:
        raise ValueError(
            f"No known figshare file id for year(s) {unknown}. "
            f"Available: {sorted(FIGSHARE_YEAR_FILES)}. If EAGLE-I has published a "
            f"newer annual release, re-read the id table with "
            f"curl -s https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}"
        )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return [
        _fetch(FIGSHARE_YEAR_FILES[year], RAW_DIR / f"eaglei_outages_{year}.csv", force)
        for year in sorted(years)
    ]


def discover_local_year_files() -> dict[int, Path]:
    """Find every eaglei_outages_<year>.csv already sitting in RAW_DIR."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    found = {}
    for path in RAW_DIR.glob("eaglei_outages_*.csv"):
        m = _YEAR_FILE_RE.search(path.name)
        if m:
            found[int(m.group(1))] = path
    return found


def load_outage_events(
    years: list[int] | None = None, fips_codes: set[str] | None = None
) -> pd.DataFrame:
    """Load raw outage event rows (fips_code, county, state, customers_out, run_start_time).

    Per-year header drift is normalised to CANONICAL_COLUMNS before concatenation — see
    the note on _COLUMN_ALIASES above for what actually differs between years.

    The source files are sparse: a row exists only when customers_out > 0 for that
    county at that 15-minute timestamp. Building the dense zero-filled grid needed for
    the `x` target is preprocessing's job, not this loader's (see
    code/preprocessing/build_target_grid.py).

    `fips_codes` restricts the result to those counties, applied chunk by chunk *during*
    the read rather than after it. This is a memory constraint, not an optimisation:
    each annual file is ~30M rows, and the training table needs two of them at once
    while only 102 of the ~3,050 covered counties are ever used — holding both years
    whole first exhausts RAM on a 16 GB machine and dies mid-build. Filtering as we read
    keeps the peak proportional to one chunk plus the ~3% that survives. Leave it None
    (the default) for the whole-population scans that county selection needs.
    """
    year_files = discover_local_year_files()
    if years is not None:
        missing = [y for y in years if y not in year_files]
        if missing:
            raise FileNotFoundError(
                f"No local eaglei_outages_<year>.csv for years {missing} in {RAW_DIR}. "
                f"Fetch them with download_year_files({missing}) — they come straight "
                f"from figshare over anonymous HTTPS, no Globus login involved. "
                f"Budget ~0.6-1.4 GB per year."
            )
        year_files = {y: year_files[y] for y in years}

    frames = []
    for year, path in sorted(year_files.items()):
        chunks = []
        for chunk in pd.read_csv(
            path,
            dtype={"fips_code": str, "county": str, "state": str},
            parse_dates=["run_start_time"],
            chunksize=READ_CHUNK_ROWS,
        ):
            chunk = chunk.rename(columns=_COLUMN_ALIASES)
            absent = [c for c in CANONICAL_COLUMNS if c not in chunk.columns]
            if absent:
                raise ValueError(
                    f"{path.name} is missing column(s) {absent} after alias normalisation; "
                    f"its header is {list(chunk.columns)}. EAGLE-I has renamed a column "
                    f"again — extend _COLUMN_ALIASES rather than letting the concat below "
                    f"quietly produce an all-NaN year."
                )
            chunk["fips_code"] = chunk["fips_code"].str.zfill(5)
            # Selecting CANONICAL_COLUMNS also drops 2024's extra total_customers
            # column, on purpose: it exists for that one year only, so keeping it would
            # inject a mostly-NaN denominator into the concatenated frame, and it
            # disagrees with MCC.csv by ~20% anyway (Autauga 01001: 29,666 vs 24,619).
            # load_total_customers() stays the single source of the denominator; see
            # PLAN.md for the open question of which one scores.
            chunk = chunk[CANONICAL_COLUMNS]
            if fips_codes is not None:
                chunk = chunk[chunk["fips_code"].isin(fips_codes)]
            chunks.append(chunk)
        frames.append(pd.concat(chunks, ignore_index=True))
    if not frames:
        raise FileNotFoundError(
            f"No eaglei_outages_<year>.csv files found in {RAW_DIR}. "
            f"Run download_year_files([2025]) to fetch the test-window year."
        )
    return pd.concat(frames, ignore_index=True).sort_values("run_start_time")


def load_total_customers() -> pd.DataFrame:
    """Load the reconciled per-county total_customers — the denominator of x.

    NOT raw MCC.csv. MCC is wrong for entire states (North Carolina lands at 1.82M
    against a published 5.30M, putting Mecklenburg/Charlotte at 28,172 customers
    instead of 588,615 — a 21x error on a county this team reports on). The reconciled
    table picks, per state, whichever candidate source reproduces the independently
    published state total; see preprocessing/reconcile_denominators.py for the rule and
    the evidence.

    Missing input is fatal rather than a silent fall back to MCC: this value multiplies
    every prediction, so quietly reverting to the broken denominator is the worst
    possible failure mode.
    """
    path = PROCESSED_DIR / "total_customers_reconciled.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with "
            f"`python -m preprocessing.reconcile_denominators` (needs "
            f"eaglei_outages_2024.csv). Falling back to raw MCC.csv is deliberately not "
            f"done here — MCC is wrong by up to 21x for some counties, and the error "
            f"would be invisible in the output."
        )
    df = pd.read_csv(path, dtype={"fips_code": str})
    df["fips_code"] = df["fips_code"].str.zfill(5)
    return df[["fips_code", "total_customers"]]


if __name__ == "__main__":
    requested = [int(a) for a in sys.argv[1:]]

    downloaded = download_support_files()
    print(f"Support files in {RAW_DIR}:")
    for p in downloaded:
        print(f"  {p.name} ({p.stat().st_size:,} bytes)")

    if requested:
        print(f"\nDownloading annual files {requested} from figshare...")
        for p in download_year_files(requested):
            print(f"  {p.name} ({p.stat().st_size:,} bytes)")

    local_years = discover_local_year_files()
    print(f"\nYear files available locally: {sorted(local_years)}")
    missing_years = sorted(set(FIGSHARE_YEAR_FILES) - set(local_years))
    if missing_years:
        print(f"Not downloaded yet: {missing_years}")
        print(f"  python -m data_acquisition.eagle_i {' '.join(map(str, missing_years))}")
