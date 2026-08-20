"""
Fit a lead-dependent blend between the model and the persistence baseline.

Why. The lead-time skill curve shows persistence beating the LightGBM model below about
12 hours and losing beyond it, so a single predictor is the wrong shape for a submission
whose two tasks live on opposite sides of that crossover.

The axis that matters is the OPERATIONAL lead, target_time - issue_time, not the
forecast age target_time - run_time that the model consumes as `lead_hours`. The two
coincide during training, where issue_time is defined as the run's own initialisation
time, but they come apart at serving time and they come apart badly:

    Task A   issue D 00Z, run (D-1) 12Z   operational 1-48h,     forecast age 12-60h
    Task B   issue D 00/06/12/18Z, same run   operational 0.25-6h,  forecast age 12.25-36h

Task B therefore asks the model for a 15-minute-ahead prediction while handing it
`lead_hours` between 12 and 36. In training, a row with lead_hours=24 had autoregressive
features that were also 24 hours stale, so the model learned to discount them at that
lead. At serving time those same features are only minutes old and highly informative,
but the model cannot know that: nothing in its input distinguishes the two situations.
Persistence, which just carries x_at_issue forward, has no such confusion.

Blending is the cheap correction. Fitting the weight per operational-lead bucket on the
held-out period, where issue_time does equal run_time and the two axes agree, is the
closest honest proxy available for the serving-time behaviour.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_acquisition.eagle_i import PROCESSED_DIR
from model_bundle import load_bundle
from train import NON_FEATURE_COLS, VAL_START, load_table, temporal_split

BUNDLE_DIR = PROCESSED_DIR / "model_bundle"
WEIGHTS_PATH = BUNDLE_DIR / "blend_weights.json"
TABLE_PATH = PROCESSED_DIR / "training_table_partial.parquet"

# Upper edges, in operational lead hours. The first covers everything Task B ever asks
# for (0.25-6h); the rest track Task A across its 48-hour horizon.
LEAD_EDGES = [6, 12, 24, 48, 72]
WEIGHT_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)


def bucket_of(lead_hours: pd.Series) -> pd.Series:
    return pd.cut(lead_hours, bins=[0] + LEAD_EDGES, right=True)


def fit_weights(val_df: pd.DataFrame, model_pred: np.ndarray) -> dict[str, float]:
    """Per-bucket weight on persistence that minimises RMSE against the truth.

    RMSE rather than MAE: on a target that is 71.6% exact zeros, MAE is minimised by
    collapsing toward zero (the always-zero baseline is nearly unbeatable on it), so
    tuning a blend on MAE would just pick whichever component predicts less.
    """
    truth = val_df["target_x"].to_numpy()
    # A missing x_at_issue means EAGLE-I had no reading at issue_time, not "no outage" —
    # persistence is undefined there, so those rows fall back to the model alone.
    persistence = val_df["x_at_issue"].to_numpy()
    usable = ~np.isnan(persistence)
    buckets = bucket_of(val_df["lead_hours"])

    weights = {}
    for b in buckets.cat.categories:
        sel = (buckets == b).to_numpy() & usable
        if sel.sum() < 1000:
            weights[str(b)] = 0.0
            continue
        t, m, p = truth[sel], model_pred[sel], persistence[sel]
        rmses = [np.sqrt((((1 - w) * m + w * p - t) ** 2).mean()) for w in WEIGHT_GRID]
        best = int(np.argmin(rmses))
        weights[str(b)] = float(WEIGHT_GRID[best])
        print(f"  {str(b):<12} n={sel.sum():>7,}  best w_persistence={WEIGHT_GRID[best]:.2f}  "
              f"RMSE {rmses[-1] if best == len(WEIGHT_GRID)-1 else rmses[0]:.6f} -> {rmses[best]:.6f}")
    return weights


def apply_blend(
    model_pred: np.ndarray, persistence: np.ndarray, lead_hours: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    """Blend at prediction time. lead_hours here must be the OPERATIONAL lead."""
    out = np.asarray(model_pred, dtype=float).copy()
    buckets = bucket_of(pd.Series(lead_hours))
    for b, w in weights.items():
        if w == 0:
            continue
        sel = (buckets.astype(str) == b).to_numpy() & ~np.isnan(persistence)
        out[sel] = (1 - w) * out[sel] + w * np.asarray(persistence, dtype=float)[sel]
    return np.clip(out, 0, 1)


def main() -> None:
    df = load_table(TABLE_PATH)
    train_df, val_df = temporal_split(df, VAL_START)
    bundle = load_bundle(BUNDLE_DIR)

    X = val_df[bundle.feature_names].copy()
    X["fips_code"] = bundle.encode_fips(X["fips_code"])
    model_pred = np.clip(bundle.booster.predict(X), 0, 1)

    print(f"Fitting blend weights on {len(val_df):,} held-out rows "
          f"(split {VAL_START.date()}), by operational lead:")
    weights = fit_weights(val_df, model_pred)

    truth = val_df["target_x"].to_numpy()
    blended = apply_blend(
        model_pred, val_df["x_at_issue"].to_numpy(), val_df["lead_hours"].to_numpy(), weights
    )

    def rmse(p):
        return float(np.sqrt(((p - truth) ** 2).mean()))

    print(f"\nOverall RMSE  model {rmse(model_pred):.6f} -> blended {rmse(blended):.6f} "
          f"({100*(rmse(blended)-rmse(model_pred))/rmse(model_pred):+.2f}%)")

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps(weights, indent=2), encoding="utf8")
    print(f"Wrote {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
