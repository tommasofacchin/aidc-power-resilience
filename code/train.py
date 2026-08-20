"""
Train the Task A/B risk model: LightGBM with a Tweedie objective (PLAN.md section 3.2),
predicting target_x from weather + autoregressive + temporal + static county features.

Split is temporal, never random (PLAN.md section 3.7) — a random split would leak
adjacent-in-time rows (the same storm, 15 minutes apart) across train/val and make
validation numbers meaningless.

VAL_START was reviewed against the real table on 20 Aug 2026, as the NOTE that used to
sit here said it should be. It had been 2025-07-01, chosen when the plan still expected
runs through August; the IFS download actually stopped at 2025-06-14 against the
Open-Meteo daily quota, so nothing in the table ever reached that cutoff, the validation
split came out empty every time, and the fallback below quietly substituted a RANDOM
80/20 split — precisely the leaky split the paragraph above rules out. Every validation
number produced before that date came from the leaky path and must not be reported.
2025-05-01 puts the boundary inside the data: ~80/20 by rows, with 32.7% positive rows
in validation against 28.4% overall, so the held-out period is not an unrepresentatively
calm stretch. Re-check this if the table's span changes again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_acquisition.eagle_i import PROCESSED_DIR
from model_bundle import save_bundle

VAL_START = pd.Timestamp("2025-05-01")  # see NOTE above — must stay inside the table's span
BUNDLE_DIR = PROCESSED_DIR / "model_bundle"

NON_FEATURE_COLS = {"issue_time", "target_time", "target_x"}
CATEGORICAL_COLS = ["fips_code"]

TWEEDIE_VARIANCE_POWER = 1.5
LGBM_PARAMS = dict(
    objective="tweedie",
    tweedie_variance_power=TWEEDIE_VARIANCE_POWER,
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["fips_code"] = df["fips_code"].astype("category")
    return df


def temporal_split(df: pd.DataFrame, val_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["issue_time"] < val_start]
    val = df[df["issue_time"] >= val_start]
    return train, val


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nonzero = y_true > 0.001
    mae_nonzero = mean_absolute_error(y_true[nonzero], y_pred[nonzero]) if nonzero.any() else float("nan")
    print(f"  {label}: MAE={mae:.6f}  RMSE={rmse:.6f}  "
          f"MAE(x_true>0.001, n={nonzero.sum()})={mae_nonzero:.6f}")


def main():
    table_path = PROCESSED_DIR / "training_table_partial.parquet"
    df = load_table(table_path)
    print(f"Loaded {len(df)} rows, issue_time range {df['issue_time'].min()} .. {df['issue_time'].max()}")

    train_df, val_df = temporal_split(df, VAL_START)
    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")
    if len(val_df) == 0:
        # Deliberately fatal. This used to fall back to a random 80/20 split with a
        # printed warning, and that is how leaky validation numbers got produced and
        # believed for a whole day: the warning scrolled past, the run exited 0, and a
        # model bundle appeared on disk looking exactly like a good one. A split that
        # violates the file's own stated rule is not a degraded result to warn about,
        # it is a broken configuration to stop on.
        raise ValueError(
            f"Empty validation set: no row has issue_time >= VAL_START ({VAL_START.date()}), "
            f"but the table spans {df['issue_time'].min()} .. {df['issue_time'].max()}. "
            f"Move VAL_START inside that range — and note the climatology in "
            f"build_training_table.py is fitted with the same cutoff, so the table must "
            f"be rebuilt after changing it, not just retrained."
        )

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X_train, y_train = train_df[feature_cols], train_df["target_x"]
    X_val, y_val = val_df[feature_cols], val_df["target_x"]

    # No early stopping, and no eval_set. The previous version passed `eval_X=`/`eval_y=`
    # — not parameters of LightGBM's sklearn fit(), which takes `eval_set`. They were
    # swallowed as unknown booster params instead of registering a validation set, and
    # the early_stopping callback then fired after 25 of the configured 500 trees. The
    # deployed model was that 25-tree stump: RMSE 0.016447 against 0.016281 for the same
    # params trained properly, and — the part that actually mattered — a maximum
    # prediction of 0.0118 against 0.0861, i.e. it had almost no dynamic range and could
    # never signal a severe outage.
    #
    # Fixed by removing early stopping rather than by repairing eval_set: stopping on the
    # validation set and then reporting metrics on that same set would make every number
    # in the report optimistic. With no early stopping the split stays a genuine holdout,
    # and n_estimators is simply a hyperparameter like any other.
    model = lgb.LGBMRegressor(**LGBM_PARAMS, verbose=-1)
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL_COLS)

    pred_val = np.clip(model.predict(X_val), 0, 1)

    print("\n=== Validation metrics ===")
    evaluate(y_val.values, pred_val, "LightGBM Tweedie")

    zero_baseline = np.zeros_like(y_val.values)
    evaluate(y_val.values, zero_baseline, "Baseline: always 0")

    # x_at_issue is deliberately left NaN where the source EAGLE-I reading itself was
    # missing (see build_training_table's missing-value section) — fine for LightGBM's
    # native handling, but the persistence baseline has no defined value to fall back
    # on for those rows, so it's scored only where it actually has an opinion.
    has_issue_x = val_df["x_at_issue"].notna()
    evaluate(
        y_val.values[has_issue_x], val_df.loc[has_issue_x, "x_at_issue"].values,
        f"Baseline: persistence (x_at_issue, n={has_issue_x.sum()}/{len(val_df)})",
    )

    print("\n=== Top 15 feature importances (gain) ===")
    importances = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=feature_cols)
    print(importances.sort_values(ascending=False).head(15).to_string())

    # Save the booster together with the fitted state its features depend on — the
    # fips_code category ordering and the climatology thresholds. Saving the booster
    # alone is what allowed prediction-time code to re-derive both and silently get
    # different answers; see code/model_bundle.py for the two concrete bugs.
    # (booster_.save_model() itself is avoided inside save_bundle because LightGBM's
    # C++ file writer fails on this project's OneDrive-synced path.)
    climatology = pd.read_parquet(PROCESSED_DIR / "climatology.parquet")
    save_bundle(
        BUNDLE_DIR,
        booster=model.booster_,
        fips_categories=list(df["fips_code"].cat.categories),
        climatology=climatology,
    )
    print(f"\nModel bundle saved to {BUNDLE_DIR} "
          f"({len(df['fips_code'].cat.categories)} fips categories, "
          f"{len(climatology)} counties of climatology)")


if __name__ == "__main__":
    main()
