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

- `target_kind`: WHAT the booster's output means. The model predicts the residual from
  persistence, not the level, so raw output must be added to `x_at_issue` before it is a
  ratio at all. A bundle that did not carry this would let predict.py apply the wrong
  reconstruction against a booster that loads perfectly and predicts plausible-looking
  small numbers — the same silent-skew failure as the two above, and the reason this is
  recorded rather than assumed. LEVEL is kept so an older bundle still reads correctly.

Saved as a directory of plain files rather than a pickle: LightGBM's own model format for
the booster (portable, inspectable), JSON for categories and target kind, parquet for the
climatology table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

MODEL_FILENAME = "model.txt"
CATEGORIES_FILENAME = "fips_categories.json"
CLIMATOLOGY_FILENAME = "climatology.parquet"
TARGET_KIND_FILENAME = "target_kind.json"

# What the booster's raw output means.
LEVEL = "level"                       # predicts target_x directly (pre-2026-08-27 bundles)
DELTA = "delta_from_persistence"      # predicts target_x - x_at_issue


def to_level(raw: np.ndarray, x_at_issue, target_kind: str) -> np.ndarray:
    """Turn raw booster output into a predicted ratio in [0, 1].

    One function because three callers need to agree exactly: train.py's validation
    report, blend.py's weight fit, and predict.py's submission. Any of them
    reconstructing on its own is how the two bugs in the module docstring happened.

    Where `x_at_issue` is NaN the source EAGLE-I reading at issue_time was missing, so
    the delta has no base to sit on. Those rows fall back to the delta alone, i.e. the
    same arithmetic with the base read as zero — 0.12% of the training table, and the
    honest reading of "no outage recorded" when nothing else is known.
    """
    raw = np.asarray(raw, dtype=float)
    if target_kind == LEVEL:
        return np.clip(raw, 0, 1)
    if target_kind != DELTA:
        raise ValueError(f"Unknown target_kind {target_kind!r}; expected {LEVEL!r} or {DELTA!r}")
    base = np.asarray(x_at_issue, dtype=float)
    return np.clip(np.where(np.isnan(base), 0.0, base) + raw, 0, 1)


@dataclass
class ModelBundle:
    booster: lgb.Booster
    fips_categories: list[str]
    climatology: pd.DataFrame
    target_kind: str = LEVEL

    @property
    def feature_names(self) -> list[str]:
        return self.booster.feature_name()

    def predict_level(self, X: pd.DataFrame, x_at_issue) -> np.ndarray:
        """Predict a ratio in [0, 1], reconstructing from whatever the booster was
        trained to output. Callers should use this rather than `booster.predict`."""
        return to_level(self.booster.predict(X), x_at_issue, self.target_kind)

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
    target_kind: str = DELTA,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    # Plain Python file I/O rather than booster.save_model(): LightGBM's own C++ writer
    # fails on this project's OneDrive-synced path (see train.py for the details).
    (directory / MODEL_FILENAME).write_text(booster.model_to_string())
    (directory / CATEGORIES_FILENAME).write_text(json.dumps(list(fips_categories)))
    (directory / TARGET_KIND_FILENAME).write_text(json.dumps({"target_kind": target_kind}))
    climatology.to_parquet(directory / CLIMATOLOGY_FILENAME)


def load_bundle(directory: Path) -> ModelBundle:
    booster = lgb.Booster(model_str=(directory / MODEL_FILENAME).read_text())
    fips_categories = json.loads((directory / CATEGORIES_FILENAME).read_text())
    climatology = pd.read_parquet(directory / CLIMATOLOGY_FILENAME)
    # A bundle written before target_kind existed holds a level model. Defaulting the
    # other way would reinterpret it as a delta and quietly add persistence twice.
    kind_path = directory / TARGET_KIND_FILENAME
    target_kind = json.loads(kind_path.read_text())["target_kind"] if kind_path.exists() else LEVEL
    return ModelBundle(
        booster=booster, fips_categories=fips_categories,
        climatology=climatology, target_kind=target_kind,
    )
