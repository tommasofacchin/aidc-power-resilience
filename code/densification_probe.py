"""Does more seasonally-matched training data actually help?

The go/no-go for spending the remaining Open-Meteo quota on additional IFS runs.

The trap this is built to avoid. The obvious test — retrain on the densified archive and
compare against the autumn holdout in blend.py — is not a fair comparison, because that
holdout's *evaluation* set is itself built from the autumn runs on disk. Add autumn runs
and both sides of the comparison change at once, so a difference cannot be attributed to
the training data.

So the evaluation set is frozen. `baseline_runs_20260822.json` lists every forecast run
on disk before any densification pass; the rows scored here come only from the November
2024 subset of that list, and no run downloaded afterwards can enter it. Two models are
then trained on everything *except* November 2024 and scored on those identical rows:

    A (baseline)   trained only on rows whose run is in the frozen list
    B (densified)  trained on every run now on disk, over the same span

B sees strictly more data than A and is scored on exactly the same rows. If B does not
beat A, the additional runs did not buy anything and the remaining quota is better spent
elsewhere — which is a real possible outcome, not a formality: the ablations in the
report already show this model gains far less from data than from the persistence blend.

The held-out unit is a calendar month, not a random slice: a random split would put rows
from the same storm on both sides. Each of September, October and November 2024 is held
out in turn and scored by a model that never saw it, so every autumn row is predicted
out-of-sample exactly once and the three folds pool into one honest autumn RMSE.

Pooling matters more than it looks. Run the months separately and they disagree: on a
simulated densification (49 autumn runs against 98) September gained 0.71%, October
1.05%, and November lost 0.15%. November is not evidence against the other two, it is a
month where every method is already near zero — its RMSE is 0.006 against September's
0.042 — so reading the three as a vote would let the quietest month outweigh the two
carrying the hurricanes. Pooled squared error weights each month by the error actually
at stake in it, which is also how any metric computed over the whole test window will.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blend import apply_blend, fit_weights
from data_acquisition.eagle_i import PROCESSED_DIR
from train import LGBM_PARAMS, NON_FEATURE_COLS, load_table

TABLE_PATH = PROCESSED_DIR / "training_table_partial.parquet"
BASELINE_RUNS_PATH = PROCESSED_DIR / "baseline_runs_20260822.json"

EVAL_MONTHS = ["2024-09", "2024-10", "2024-11"]
EVENT_THRESHOLD = 0.001


def metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    events = truth > EVENT_THRESHOLD
    return {
        "RMSE": float(np.sqrt(((pred - truth) ** 2).mean())),
        "MAE": float(np.abs(pred - truth).mean()),
        "MAE_events": float(np.abs(pred[events] - truth[events]).mean()),
        "max_pred": float(pred.max()),
    }


def train_on(df: pd.DataFrame, feature_cols: list[str]) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**LGBM_PARAMS, verbose=-1)
    model.fit(df[feature_cols], df["target_x"], categorical_feature=["fips_code"])
    return model


def main(baseline_path: Path = BASELINE_RUNS_PATH, eval_months: list[str] | None = None) -> None:
    if not baseline_path.exists():
        raise SystemExit(
            f"{baseline_path} not found. It is the frozen run list this comparison "
            f"is defined against and cannot be regenerated after densification — without "
            f"it there is no honest before/after."
        )
    eval_months = list(eval_months or EVAL_MONTHS)
    baseline_runs = {pd.Timestamp(r) for r in json.loads(baseline_path.read_text("utf8"))}
    df = load_table(TABLE_PATH)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    from_baseline = df["issue_time"].isin(baseline_runs)

    on_disk = set(df["issue_time"].unique())
    added = len(on_disk - baseline_runs)
    print(f"Runs in the table: {len(on_disk)}  (frozen baseline: {len(baseline_runs)}, "
          f"added since: {added})")
    if added == 0:
        print("\nNothing has been densified yet, so there is nothing to compare.")
        return

    pooled = {"A baseline": [], "B densified": []}
    pooled_truth = []
    for month in eval_months:
        start = pd.Timestamp(month + "-01")
        in_month = (df["issue_time"] >= start) & (df["issue_time"] < start + pd.offsets.MonthBegin(1))
        eval_df = df[in_month & from_baseline]
        if eval_df.empty:
            print(f"\n--- {month}: no frozen rows, skipped")
            continue
        truth = eval_df["target_x"].to_numpy()
        persistence = eval_df["x_at_issue"].to_numpy()
        leads = eval_df["lead_hours"].to_numpy()
        pooled_truth.append(truth)

        print(f"\n--- fold {month}: {len(eval_df):,} frozen rows scored")
        for label, train_df in [("A baseline", df[~in_month & from_baseline]),
                                ("B densified", df[~in_month])]:
            pred = np.clip(train_on(train_df, feature_cols).predict(eval_df[feature_cols]), 0, 1)
            blended = apply_blend(pred, persistence, leads, fit_weights(eval_df, pred, quiet=True))
            pooled[label].append(blended)
            m = metrics(truth, blended)
            print(f"    {label:<12} trained on {len(train_df):>9,} rows  "
                  f"blended RMSE={m['RMSE']:.6f}  MAE_ev={m['MAE_events']:.6f}")
        print(f"    {'always zero':<12} {'':>19}  "
              f"blended RMSE={metrics(truth, np.zeros_like(truth))['RMSE']:.6f}")

    truth = np.concatenate(pooled_truth)
    a = metrics(truth, np.concatenate(pooled["A baseline"]))
    b = metrics(truth, np.concatenate(pooled["B densified"]))
    delta = 100 * (a["RMSE"] - b["RMSE"]) / a["RMSE"]
    print(f"\n=== pooled over {len(pooled_truth)} folds, {len(truth):,} rows ===")
    print(f"  A baseline   RMSE={a['RMSE']:.6f}  MAE_ev={a['MAE_events']:.6f}")
    print(f"  B densified  RMSE={b['RMSE']:.6f}  MAE_ev={b['MAE_events']:.6f}")
    print(f"  change from densification: {delta:+.2f}% RMSE, "
          f"{100 * (a['MAE_events'] - b['MAE_events']) / a['MAE_events']:+.2f}% event MAE")
    print("\nGO — the extra runs pay for themselves." if delta > 0.5 else
          "\nNO-GO — under half a percent pooled; the remaining quota buys more elsewhere.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Densification go/no-go.")
    parser.add_argument("--baseline", type=Path, default=BASELINE_RUNS_PATH,
                        help="frozen run list defining the evaluation set")
    parser.add_argument("--eval-months", nargs="+", default=EVAL_MONTHS,
                        help=f"YYYY-MM folds, each held out and scored (default {EVAL_MONTHS})")
    a = parser.parse_args()
    main(baseline_path=a.baseline, eval_months=a.eval_months)
