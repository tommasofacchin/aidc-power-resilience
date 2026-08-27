"""
Seasonal-transfer probe: how much skill survives when the evaluation window is autumn?

Why this exists. Every other number in this project is validated on the 2025 rows after
VAL_START, because that is the tail of the contiguous span the weather download covered
and a forward-in-time split is the only honest one inside a season. But the *evaluated*
window is 1 September - 30 November, and late spring is not autumn: no tropical system,
a different convective regime, and — visible in the target itself — a different base rate
of outages. A model that looks good in May tells you very little about November, and the
report should not pretend otherwise.

The 2024 autumn runs downloaded on 21 August make a direct measurement possible for the
first time. This script trains on 2025 only and evaluates on September-November 2024, so
the evaluation window is seasonally matched to the real test window while remaining
strictly disjoint from what the model saw. It then repeats the identical comparison on
the ordinary forward split, so the two seasons can be read side by side rather than
against different baselines.

Two deliberate departures from train.py, both to keep the probe honest:

1. The climatology feature group is dropped from BOTH sides of the comparison. Those
   twelve features are per-county gust/precipitation quantiles fitted, in
   build_training_table.py, on every run before VAL_START — which includes the autumn
   2024 rows this script evaluates on. Keeping them would let the autumn evaluation see
   quantiles fitted partly on itself. Dropping them costs almost nothing (the ablation
   puts the group's marginal RMSE contribution near zero) and removes the objection
   entirely. Everything else — raw forecast fields, derived weather, autoregressive
   state, temporal encodings, county identity — is untouched.

2. Direction of time is reversed for the autumn probe: it trains on 2025 and evaluates
   on 2024. That is a seasonal-transfer question, not a forecasting-into-the-future one,
   and no model selection is done on its result — the submitted model is still the one
   train.py fits. It is reported as what it is.

Output: data/processed/seasonal_holdout.{md,csv}, meant to be read alongside
ablation_results.md rather than replacing it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation import (
    EVENT_THRESHOLD,
    FEATURE_GROUPS,
    climatology_baseline,
    fit_predict,
    metrics,
    to_markdown,
)
from data_acquisition.eagle_i import PROCESSED_DIR
from train import NON_FEATURE_COLS, VAL_START, load_table

TABLE_PATH = PROCESSED_DIR / "training_table_partial.parquet"
OUT_CSV = PROCESSED_DIR / "seasonal_holdout.csv"
OUT_MD = PROCESSED_DIR / "seasonal_holdout.md"

AUTUMN_START = pd.Timestamp("2024-09-01")
AUTUMN_END = pd.Timestamp("2024-12-01")
YEAR_2025_START = pd.Timestamp("2025-01-01")


def season_label(df: pd.DataFrame, suffix: str) -> str:
    """Name a window by the dates it actually covers.

    Spelled out from the data rather than hardcoded because the reference window grows
    every time more weather runs finish downloading: it was May-June when first written
    and is already wider, and a stale label on a results table is the kind of small
    wrongness that quietly propagates into the report.
    """
    lo, hi = df["issue_time"].min().date(), df["issue_time"].max().date()
    return f"{lo} .. {hi} ({suffix})"


def evaluate_split(
    train_df: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str], season: str
) -> list[dict]:
    """Model plus the two non-learned baselines, all on the same evaluation rows."""
    y = eval_df["target_x"].to_numpy()
    rows = []

    preds = {
        "model (LightGBM delta)": fit_predict(train_df, eval_df, feature_cols),
        "always zero": np.zeros(len(eval_df)),
        "county climatology": climatology_baseline(train_df, eval_df),
    }

    # Persistence is scored only where x_at_issue exists: EAGLE-I leaves genuine NaNs
    # where its ETL got no reading (see build_training_table.py), and the baseline has
    # no defined value there. Scoring it on a subset while the others are scored on all
    # rows would be comparing different populations, so the subset is reported in n.
    has_issue = eval_df["x_at_issue"].notna().to_numpy()

    for label, pred in preds.items():
        rows.append({
            "season": season, "predictor": label, "n": len(eval_df),
            "event_rate": float((y > EVENT_THRESHOLD).mean()), **metrics(y, pred),
        })
    rows.append({
        "season": season, "predictor": "persistence (x at issue)", "n": int(has_issue.sum()),
        "event_rate": float((y[has_issue] > EVENT_THRESHOLD).mean()),
        **metrics(y[has_issue], eval_df["x_at_issue"].to_numpy()[has_issue]),
    })
    return rows


def main() -> None:
    df = load_table(TABLE_PATH)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    dropped = FEATURE_GROUPS["climatology"]
    feature_cols = [c for c in feature_cols if c not in dropped]
    print(f"Features: {len(feature_cols)} "
          f"(climatology group of {len(dropped)} dropped — see module docstring)")

    autumn = df[(df["issue_time"] >= AUTUMN_START) & (df["issue_time"] < AUTUMN_END)]
    y2025 = df[df["issue_time"] >= YEAR_2025_START]
    if autumn.empty:
        raise ValueError(
            f"No rows with issue_time in [{AUTUMN_START.date()}, {AUTUMN_END.date()}). "
            f"The table spans {df['issue_time'].min()} .. {df['issue_time'].max()} — "
            f"rebuild it with `--years 2024 2025` before running this probe."
        )

    print(f"\nAutumn probe: train on 2025 ({len(y2025):,} rows), "
          f"evaluate on Sep-Nov 2024 ({len(autumn):,} rows)")
    rows = evaluate_split(y2025, autumn, feature_cols, season_label(autumn, "seasonally matched"))

    spring_train = df[df["issue_time"] < VAL_START]
    spring_eval = df[df["issue_time"] >= VAL_START]
    print(f"Reference split: train before {VAL_START.date()} ({len(spring_train):,} rows), "
          f"evaluate after ({len(spring_eval):,} rows)")
    rows += evaluate_split(
        spring_train, spring_eval, feature_cols, season_label(spring_eval, "reference split")
    )

    out = pd.DataFrame(rows)[
        ["season", "predictor", "n", "event_rate", "MAE", "RMSE", "MAE_events", "max_pred"]
    ]

    # Skill against the always-zero reference, per season — the comparison the report
    # actually makes. Positive means the predictor beats always-zero on that metric.
    for season, block in out.groupby("season", sort=False):
        zero = block[block["predictor"] == "always zero"].iloc[0]
        for col in ("RMSE", "MAE_events"):
            out.loc[block.index, f"skill_{col}_%"] = (
                100 * (zero[col] - block[col]) / zero[col]
            ).round(2)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(
        "# Seasonal holdout\n\n"
        f"Autumn probe trains on 2025 and evaluates on {AUTUMN_START.date()} .. "
        f"{(AUTUMN_END - pd.Timedelta(days=1)).date()}; the reference split is the one "
        f"every other table in this project uses. The twelve climatology features are "
        f"dropped from both, so neither side sees quantiles fitted on its own evaluation "
        f"window. `skill_*_%` is the improvement over always-zero within that season.\n\n"
        + to_markdown(out) + "\n",
        encoding="utf8",
    )
    print(f"\n{to_markdown(out)}\n")
    print(f"Wrote {OUT_MD} and {OUT_CSV}")


if __name__ == "__main__":
    main()
