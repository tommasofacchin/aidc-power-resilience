"""
EAGLE-I outage data acquisition.

Source layout is fragmented across two access paths (see PLAN.md / project memory):

- MCC.csv, DQI.csv, coverage_history.csv, and eaglei_outages_2014.csv are reachable
  with a plain HTTPS GET via figshare's ndownloader, resolved through the file ids
  discovered under share token 417a4f147cf1357a5391.
- Later annual releases (2015 onward, including the 2025 release that covers the
  Phase 1 test window) are Globus-only and require an interactive browser login.
  Those files must be downloaded manually and dropped into RAW_DIR; this module picks
  them up from there by filename pattern (eaglei_outages_<year>.csv).

Everything downloaded or derived is cached under data/raw and data/processed so the
API/figshare calls happen at most once.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# figshare file ids confirmed reachable via https://ndownloader.figshare.com/files/<id>
# (share token 417a4f147cf1357a5391 — see project memory for how these were found)
FIGSHARE_FILES = {
    "MCC.csv": 42547708,
    "DQI.csv": 42547705,
    "coverage_history.csv": 42547714,
    "eaglei_outages_2014.csv": 42547717,
}

GLOBUS_LINK = (
    "https://app.globus.org/file-manager"
    "?destination_id=57618e0a-2c99-45ff-9694-24141b92fa17"
    "&destination_path=%2Fgen101%2Fworld-shared%2Fdoi-data%2FORNLNCCS"
    "%2F202602%2F10.13139_ORNLNCCS_3012826%2F"
)

_YEAR_FILE_RE = re.compile(r"eaglei_outages_(\d{4})\.csv$", re.IGNORECASE)


def download_figshare_files(force: bool = False) -> list[Path]:
    """Download every known figshare-reachable EAGLE-I file into RAW_DIR."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, file_id in FIGSHARE_FILES.items():
        dest = RAW_DIR / filename
        if dest.exists() and not force:
            paths.append(dest)
            continue
        url = f"https://ndownloader.figshare.com/files/{file_id}"
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        paths.append(dest)
    return paths


def discover_local_year_files() -> dict[int, Path]:
    """Find every eaglei_outages_<year>.csv already sitting in RAW_DIR.

    This is how manually-downloaded Globus years (2015-2025) get picked up, on top of
    whatever download_figshare_files() fetched automatically.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    found = {}
    for path in RAW_DIR.glob("eaglei_outages_*.csv"):
        m = _YEAR_FILE_RE.search(path.name)
        if m:
            found[int(m.group(1))] = path
    return found


def load_outage_events(years: list[int] | None = None) -> pd.DataFrame:
    """Load raw outage event rows (fips_code, county, state, customers_out, run_start_time).

    The source files are sparse: a row exists only when customers_out > 0 for that
    county at that 15-minute timestamp. Building the dense zero-filled grid needed for
    the `x` target is preprocessing's job, not this loader's (see
    code/preprocessing/build_target_grid.py).
    """
    year_files = discover_local_year_files()
    if years is not None:
        missing = [y for y in years if y not in year_files]
        if missing:
            raise FileNotFoundError(
                f"No local eaglei_outages_<year>.csv for years {missing}. "
                f"2014 is fetched automatically by download_figshare_files(); "
                f"later years must be downloaded via Globus and placed in {RAW_DIR}. "
                f"Globus link: {GLOBUS_LINK}"
            )
        year_files = {y: year_files[y] for y in years}

    frames = []
    for year, path in sorted(year_files.items()):
        df = pd.read_csv(
            path,
            dtype={"fips_code": str, "county": str, "state": str},
            parse_dates=["run_start_time"],
        )
        df["fips_code"] = df["fips_code"].str.zfill(5)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No eaglei_outages_<year>.csv files found in {RAW_DIR}. "
            f"Run download_figshare_files() first, or fetch later years via Globus: {GLOBUS_LINK}"
        )
    return pd.concat(frames, ignore_index=True).sort_values("run_start_time")


def load_max_customer_counts() -> pd.DataFrame:
    """Load MCC.csv — modeled county customer counts, used as the total_customers denominator.

    Source columns are County_FIPS, Customers (not the fips_code/customers_out naming
    used by the outage event files) — renamed here so downstream code has one
    consistent schema to join against.
    """
    path = RAW_DIR / "MCC.csv"
    if not path.exists():
        download_figshare_files()
    df = pd.read_csv(path, dtype={"County_FIPS": str})
    df = df.rename(columns={"County_FIPS": "fips_code", "Customers": "total_customers"})
    df["fips_code"] = df["fips_code"].str.zfill(5)
    return df


if __name__ == "__main__":
    downloaded = download_figshare_files()
    print(f"Downloaded/verified {len(downloaded)} figshare files in {RAW_DIR}:")
    for p in downloaded:
        print(f"  {p.name} ({p.stat().st_size:,} bytes)")

    local_years = discover_local_year_files()
    print(f"\nYear files available locally: {sorted(local_years)}")
    missing_years = sorted(set(range(2015, 2026)) - set(local_years))
    if missing_years:
        print(f"Missing years (need manual Globus download into {RAW_DIR}): {missing_years}")
        print(f"Globus link: {GLOBUS_LINK}")
