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

Blending is the cheap correction. Fitting the weight per operational-lead bucket on a
held-out period, where issue_time does equal run_time and the two axes agree, is the
closest honest proxy available for the serving-time behaviour.

WHICH held-out period turned out to matter more than the blend itself. The weights were
originally fitted on May-June 2025, the tail of the 2025 span, and came out at
0.80/0.70/0.10/0.00/0.00 across the five lead buckets: persistence only for the shortest
leads, model alone beyond a day. Refitting the identical procedure on the autumn 2024
holdout (2024-09-01 .. 2024-11-30, seasonally matched to the Sep-Nov test window) gives
0.90/0.80/0.75/0.65/0.50 — persistence is worth *more* at every lead, and still worth
half the prediction two to three days out.

The reason is physical, not statistical. Late-spring outages are short convective events
that decay within hours, so yesterday's state says little about tomorrow. Autumn outages
are storm-restoration events lasting days, so the current state stays informative across
the whole 48-hour Task A horizon. Measured on that autumn holdout: model alone RMSE
0.036917, spring-fitted weights 0.034361 (-6.9%), autumn-fitted weights 0.028579
(-22.6%). The same autumn weights cost about 2% RMSE back on the May-June window — a
season the submission is never scored in. `--season autumn` is therefore the default.

Caveat worth stating in the report: autumn 2024 contains Helene and Milton, so the
weights are fitted on a window with two major landfalling systems. They would be too
persistence-heavy for an unusually quiet autumn. The downside is bounded, though —
during calm periods x_at_issue is ~0, so a heavy persistence weight shrinks predictions
toward zero, which is the direction this target rewards anyway.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_acquisition.eagle_i import PROCESSED_DIR
from model_bundle import DELTA, load_bundle, to_level
from train import NON_FEATURE_COLS, VAL_START, fit_delta_model, load_table, temporal_split

BUNDLE_DIR = PROCESSED_DIR / "model_bundle"
WEIGHTS_PATH = BUNDLE_DIR / "blend_weights.json"
TABLE_PATH = PROCESSED_DIR / "training_table_partial.parquet"

# The autumn block the seasonal fit uses. Sep-Nov 2024 is the only stretch of the IFS
# archive that is seasonally matched to the Sep-Nov 2025 test window.
AUTUMN_START = pd.Timestamp("2024-09-01")
AUTUMN_END = pd.Timestamp("2024-12-01")

# Upper edges, in operational lead hours. The first covers everything Task B ever asks
# for (0.25-6h); the rest track Task A across its 48-hour horizon.
LEAD_EDGES = [6, 12, 24, 48, 72]
WEIGHT_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)

# Half-lives in hours for the restoration decay, plus None for flat carry-forward — the
# unconditional assumption of the previous version. Flat persistence says a county with
# 4% of customers out now still has 4% out in two days, which no restoration crew makes
# true; the decay lets the fit discover the rate instead of asserting one. None being in
# the grid means the search can always fall back to the old behaviour, so this cannot
# make the fitted blend worse on the window it is fitted on.
HALFLIFE_GRID = [None, 96, 72, 48, 36, 24, 18, 12, 9, 6, 4, 3, 2]


def decay(persistence: np.ndarray, lead_hours: np.ndarray, halflife: float | None) -> np.ndarray:
    """Persistence carried forward with exponential restoration decay.

    halflife=None reproduces flat carry-forward exactly.
    """
    if halflife is None:
        return persistence
    return persistence * np.power(0.5, np.asarray(lead_hours, dtype=float) / float(halflife))


def bucket_of(lead_hours: pd.Series) -> pd.Series:
    return pd.cut(lead_hours, bins=[0] + LEAD_EDGES, right=True)


def fit_weights(val_df: pd.DataFrame, model_pred: np.ndarray, quiet: bool = False,
                halflives: list | None = None) -> dict[str, dict]:
    """Per-bucket persistence weight and decay half-life that minimise RMSE.

    RMSE rather than MAE: on a target that is 69.9% exact zeros, MAE is minimised by
    collapsing toward zero (the always-zero baseline is nearly unbeatable on it), so
    tuning a blend on MAE would just pick whichever component predicts less.

    Two parameters are fitted jointly per bucket rather than one, because they trade
    against each other: shrinking a stale persistence toward zero can be done either by
    weighting the model up or by decaying the carry-forward, and fitting one at a time
    would attribute to the first whatever the second could have explained. Two
    parameters against tens of thousands of rows per bucket is not a quantity of
    freedom that needs regularising.

    `halflives` overrides the search grid. Its only caller is report_numbers.py, which
    passes [None] to refit the flat-carry-forward version and measure what the decay
    actually buys against it — the negative result section 5.1 of the report reports.
    """
    truth = val_df["target_x"].to_numpy()
    # A missing x_at_issue means EAGLE-I had no reading at issue_time, not "no outage" —
    # persistence is undefined there, so those rows fall back to the model alone.
    persistence = val_df["x_at_issue"].to_numpy()
    usable = ~np.isnan(persistence)
    leads = val_df["lead_hours"].to_numpy(dtype=float)
    buckets = bucket_of(val_df["lead_hours"])

    weights = {}
    for b in buckets.cat.categories:
        sel = (buckets == b).to_numpy() & usable
        if sel.sum() < 1000:
            weights[str(b)] = {"w_persistence": 0.0, "halflife_h": None}
            continue
        t, m, p, lead = truth[sel], model_pred[sel], persistence[sel], leads[sel]

        model_only = float(np.sqrt(((m - t) ** 2).mean()))
        flat_best = min(
            float(np.sqrt((((1 - w) * m + w * p - t) ** 2).mean())) for w in WEIGHT_GRID
        )
        best_rmse, best_w, best_hl = None, 0.0, None
        for halflife in (HALFLIFE_GRID if halflives is None else halflives):
            decayed = decay(p, lead, halflife)
            for w in WEIGHT_GRID:
                rmse = float(np.sqrt((((1 - w) * m + w * decayed - t) ** 2).mean()))
                if best_rmse is None or rmse < best_rmse:
                    best_rmse, best_w, best_hl = rmse, float(w), halflife

        weights[str(b)] = {"w_persistence": best_w, "halflife_h": best_hl}
        halflife_label = "flat" if best_hl is None else f"{best_hl}h"
        if quiet:
            continue
        print(f"  {str(b):<12} n={sel.sum():>7,}  w_persistence={best_w:.2f} "
              f"halflife={halflife_label:>5}  RMSE model {model_only:.6f} -> "
              f"flat blend {flat_best:.6f} -> decayed blend {best_rmse:.6f}")
    return weights


def apply_blend(
    model_pred: np.ndarray, persistence: np.ndarray, lead_hours: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    """Blend at prediction time. lead_hours here must be the OPERATIONAL lead."""
    out = np.asarray(model_pred, dtype=float).copy()
    persistence = np.asarray(persistence, dtype=float)
    lead_hours = np.asarray(lead_hours, dtype=float)
    buckets = bucket_of(pd.Series(lead_hours))
    for b, spec in weights.items():
        w = float(spec["w_persistence"])
        if w == 0:
            continue
        decayed = decay(persistence, lead_hours, spec.get("halflife_h"))
        sel = (buckets.astype(str) == b).to_numpy() & ~np.isnan(persistence)
        out[sel] = (1 - w) * out[sel] + w * decayed[sel]
    return np.clip(out, 0, 1)


def autumn_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Out-of-sample autumn evaluation: train on 2025, predict Sep-Nov 2024.

    Why not just run the deployed bundle over the autumn rows: it was trained on them.
    Its autumn predictions are in-sample, would look far better than they will in the
    real test window, and the weights fitted against them would understate how much
    persistence is worth — the exact quantity this is trying to measure. So the model
    used here is retrained from scratch on 2025 alone. It is used only to fit the
    weights; the submitted predictions still come from the full bundle.

    The feature set is deliberately the deployed one, climatology features included,
    even though their quantiles were fitted on a window that overlaps this evaluation
    (see seasonal_holdout.py). That leak flatters the model, i.e. it biases the fitted
    weights *away* from persistence — against the conclusion this measurement reaches.
    A result that survives a bias pointing the other way needs no correction for it.
    """
    train_df = df[df["issue_time"] >= AUTUMN_END]
    val_df = df[(df["issue_time"] >= AUTUMN_START) & (df["issue_time"] < AUTUMN_END)]
    if val_df.empty:
        raise ValueError(
            f"No rows with issue_time in [{AUTUMN_START.date()}, {AUTUMN_END.date()}); "
            f"the table spans {df['issue_time'].min()} .. {df['issue_time'].max()}. "
            f"Rebuild it with `--years 2024 2025` first."
        )
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    model = fit_delta_model(train_df, feature_cols)
    pred = to_level(model.predict(val_df[feature_cols]), val_df["x_at_issue"].to_numpy(), DELTA)
    label = (f"{len(val_df):,} out-of-sample autumn rows "
             f"({AUTUMN_START.date()} .. {(AUTUMN_END - pd.Timedelta(days=1)).date()}, "
             f"model retrained on {len(train_df):,} rows of 2025)")
    return val_df, pred, label


def reference_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, str]:
    """The original fit: the deployed bundle over the post-VAL_START rows."""
    _, val_df = temporal_split(df, VAL_START)
    bundle = load_bundle(BUNDLE_DIR)
    X = val_df[bundle.feature_names].copy()
    X["fips_code"] = bundle.encode_fips(X["fips_code"])
    pred = bundle.predict_level(X, val_df["x_at_issue"].to_numpy())
    return val_df, pred, f"{len(val_df):,} held-out rows (split {VAL_START.date()})"


def main(season: str = "autumn") -> None:
    df = load_table(TABLE_PATH)
    val_df, model_pred, label = (
        autumn_holdout(df) if season == "autumn" else reference_holdout(df)
    )

    print(f"Fitting blend weights on {label}, by operational lead:")
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
    import argparse

    parser = argparse.ArgumentParser(description="Fit the model/persistence blend.")
    parser.add_argument(
        "--season", choices=["autumn", "reference"], default="autumn",
        help="'autumn' (default) fits on an out-of-sample Sep-Nov 2024 evaluation, the "
             "season the submission is actually scored in; 'reference' reproduces the "
             "original May-Jun 2025 fit.",
    )
    args = parser.parse_args()
    main(season=args.season)
