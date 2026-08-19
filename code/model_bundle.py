"""
A trained model is not just the booster: it also carries the exact fitted state that
its features were built with. Keeping those together in one bundle is what prevents
the train/serve skew class of bug — where prediction-time code recomputes something
that training had fitted, and silently means something different.

Two such pieces here, both of which caused real, verified bugs before this existed:

- `fips_categories`: the ordered category list `fips_code` was encoded with at training
  time. LightGBM consumes pandas categoricals by integer CODE, so re-encoding at
  predict time over a different (smaller) set of counties silently remaps every county
  to a different identity — e.g. Orleans LA was code 84 in training and code 0 at
  predict time.
- `climatology`: per-county gust/precip quantiles. Recomputing these from whatever
  weather happens to be in scope at predict time (one 60-hour batch) rather than from
  the training distribution changed Orleans' p95 gust from 54.4 to 26.4, redefining
  every `*_exceeds_*` feature.

Saved as a directory of three plain files rather than a pickle: LightGBM's own model
format for the booster (portable, inspectable), JSON for categories, parquet for the
climatology table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import pandas as pd

MODEL_FILENAME = "model.txt"
CATEGORIES_FILENAME = "fips_categories.json"
CLIMATOLOGY_FILENAME = "climatology.parquet"


@dataclass
class ModelBundle:
    booster: lgb.Booster
    fips_categories: list[str]
    climatology: pd.DataFrame

    @property
    def feature_names(self) -> list[str]:
        return self.booster.feature_name()

    def encode_fips(self, values: pd.Series) -> pd.Categorical:
        """Encode fips_code with the TRAINING categories, so codes match what the
        model learned. Counties absent from training encode as NaN, which LightGBM
        treats as missing — better than silently reusing another county's code."""
        return pd.Categorical(values, categories=self.fips_categories)


def save_bundle(
    directory: Path,
    booster: lgb.Booster,
    fips_categories: list[str],
    climatology: pd.DataFrame,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    # Plain Python file I/O rather than booster.save_model(): LightGBM's own C++ writer
    # fails on this project's OneDrive-synced path (see train.py for the details).
    (directory / MODEL_FILENAME).write_text(booster.model_to_string())
    (directory / CATEGORIES_FILENAME).write_text(json.dumps(list(fips_categories)))
    climatology.to_parquet(directory / CLIMATOLOGY_FILENAME)


def load_bundle(directory: Path) -> ModelBundle:
    booster = lgb.Booster(model_str=(directory / MODEL_FILENAME).read_text())
    fips_categories = json.loads((directory / CATEGORIES_FILENAME).read_text())
    climatology = pd.read_parquet(directory / CLIMATOLOGY_FILENAME)
    return ModelBundle(booster=booster, fips_categories=fips_categories, climatology=climatology)
