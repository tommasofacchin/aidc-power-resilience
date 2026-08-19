"""
County centroid coordinates, from the US Census Bureau's Gazetteer file
(2025_Gaz_counties_national.txt, downloaded from
https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2025_Gazetteer/2025_Gaz_counties_national.zip).

GEOID is the 5-digit county FIPS code — the same join key used throughout this
pipeline (fips_code in EAGLE-I, County_FIPS in MCC.csv).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GAZETTEER_PATH = PROJECT_ROOT / "data" / "raw" / "2025_Gaz_counties_national.txt"


def load_county_coordinates() -> pd.DataFrame:
    """Return fips_code, latitude, longitude for every US county."""
    df = pd.read_csv(GAZETTEER_PATH, sep="|", dtype={"GEOID": str})
    df = df.rename(columns={"GEOID": "fips_code", "INTPTLAT": "latitude", "INTPTLONG": "longitude"})
    return df[["fips_code", "latitude", "longitude"]]


if __name__ == "__main__":
    coords = load_county_coordinates()
    print(f"Loaded coordinates for {len(coords)} counties")
    print(coords.head())
