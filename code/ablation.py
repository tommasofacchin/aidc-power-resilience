"""
Ablation study and lead-time skill curve — the two experimental sections the
submission guidelines require ("Experimental results and ablation analysis") and that
nothing else in this codebase produces.

Everything here runs on the SAME temporal split train.py uses (VAL_START), never a
random one, for the reason stated in train.py's docstring: adjacent 15-minute rows from
one storm would otherwise appear on both sides and inflate every number reported below.

Three families of comparison, all on the held-out period:

1. Feature-group ablations — retrain the full pipeline with one group removed, so the
   reported contribution of a group is its marginal value given everything else, not
   its univariate correlation with the target.
2. Non-learned baselines — always-zero, persistence, and per-county climatology. The
   first is not a strawman on this target: with 71.6% exact zeros it is genuinely hard
   to beat on MAE, and saying so with numbers is more useful than omitting it.
3. Hurdle model — P(outage) x E[x | outage] as an alternative head, addressing the
   "class-imbalance handling scheme" section directly. It was introduced when the
   deployed head was a Tweedie regressor on the level, which predicted a conditional
   mean, never exceeded ~1.2% of customers out, and so could not commit to a severe
   event; the hurdle formulation is the standard fix for that. The deployed head is now
   a residual-from-persistence model whose dynamic range is no longer the binding
   constraint, so this comparison is kept as evidence rather than as a live candidate.

Outputs a markdown table and a CSV under data/processed/, both meant to be pasted into
the report rather than retyped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_acquisition.eagle_i import PROCESSED_DIR
from model_bundle import DELTA, to_level
from train import LGBM_PARAMS, NON_FEATURE_COLS, VAL_START, load_table, temporal_split

TABLE_PATH = PROCESSED_DIR / "training_table_partial.parquet"
OUT_CSV = PROCESSED_DIR / "ablation_results.csv"
OUT_MD = PROCESSED_DIR / "ablation_results.md"
LEAD_CURVE_CSV = PROCESSED_DIR / "lead_time_skill.csv"

# A row counts as a real event above this ratio. Same threshold select_counties.py uses
# for "significant", kept identical so the report never has to explain two of them.
EVENT_THRESHOLD = 0.001

# Feature groups, matched by exact name or prefix. Every feature column must land in
# exactly one group — assert_groups_cover() enforces that, so adding a feature to the
# pipeline without classifying it here fails loudly instead of silently becoming
# un-ablatable.
FEATURE_GROUPS = {
    "weather_raw": [
        "cape", "cloud_cover", "dew_point_2m", "precipitation", "rain", "snowfall",
        "surface_pressure", "temperature_2m", "wind_direction_10m", "wind_gusts_10m",
        "wind_speed_10m",
    ],
    "weather_derived": [
        "gust_cubed", "wind_rain_interaction", "gust_roll_max_3h", "gust_roll_max_6h",
        "gust_roll_max_12h", "precip_cum_6h", "precip_cum_24h", "pressure_change_3h",
        "gust_jump_1h",
    ],
    "climatology": [
        "gust_clim_p95", "gust_clim_p99", "gust_clim_p999",
        "precip_clim_p95", "precip_clim_p99", "precip_clim_p999",
        "gust_exceeds_p95", "gust_exceeds_p99", "gust_exceeds_p999",
        "precip_exceeds_p95", "precip_exceeds_p99", "precip_exceeds_p999",
    ],
    "autoregressive": [
        "x_at_issue", "x_lag_15m", "x_lag_30m", "x_lag_1h", "x_lag_2h", "x_lag_6h",
        "x_trend_1h", "x_trend_2h", "x_max_24h", "outage_duration_15m_periods",
        "ongoing_outage",
    ],
    "temporal": ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "lead_hours"],
    "static": ["fips_code", "total_customers"],
}


def assert_groups_cover(feature_cols: list[str]) -> None:
    grouped = {c for cols in FEATURE_GROUPS.values() for c in cols}
    unclassified = sorted(set(feature_cols) - grouped)
    phantom = sorted(grouped - set(feature_cols))
    if unclassified:
        raise ValueError(
            f"Feature(s) {unclassified} are in the training table but not in any "
            f"FEATURE_GROUPS entry, so no ablation row would ever remove them and the "
            f"study would silently understate what the model depends on. Classify them."
        )
    if phantom:
        print(f"  note: FEATURE_GROUPS lists {phantom}, absent from this table — ignored.")


def to_markdown(df: pd.DataFrame) -> str:
    """Render a markdown table without pulling in `tabulate`.

    pandas' own .to_markdown() needs that optional dependency, and adding one to
    requirements.txt purely to format a table would make the reviewer's clean-environment
    reproduction install more than the pipeline actually needs.
    """
    def fmt(v):
        return f"{v:.6f}" if isinstance(v, float) else str(v)
    head = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    event = y_true > EVENT_THRESHOLD
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err**2).mean())),
        "MAE_events": float(np.abs(err)[event].mean()) if event.any() else float("nan"),
        "max_pred": float(y_pred.max()),
    }


def fit_predict(train_df, val_df, feature_cols, params=None,
                with_persistence: bool = True) -> np.ndarray:
    """Fit the deployed formulation — the residual from persistence — and return levels.

    `with_persistence=False` is for the autoregressive ablation only. x_at_issue enters
    the deployed system twice: as a feature, and as the base the predicted residual is
    added to. Removing only the feature would leave the ablated model still carrying the
    signal the row exists to measure the absence of, and would report the group as nearly
    free. With no current state available anywhere there is no base to take a residual
    from either, so that variant trains on the level instead — the same architecture the
    model had before 27 Aug 2026, which is the honest counterfactual.
    """
    model = lgb.LGBMRegressor(**(params or LGBM_PARAMS), verbose=-1)
    if not with_persistence:
        model.fit(train_df[feature_cols], train_df["target_x"])
        return np.clip(model.predict(val_df[feature_cols]), 0, 1)
    usable = train_df[train_df["x_at_issue"].notna()]
    model.fit(usable[feature_cols], usable["target_x"] - usable["x_at_issue"])
    return to_level(model.predict(val_df[feature_cols]),
                    val_df["x_at_issue"].to_numpy(), DELTA)


def hurdle_predict(train_df, val_df, feature_cols, threshold: float = 0.0) -> np.ndarray:
    """P(x > threshold) from a classifier, times exp(E[log x | x > threshold]).

    Splitting the two questions is the point: a single head on the level has to satisfy
    both with one number and resolves the tension by staying small everywhere, which is
    why the Tweedie version's predictions never approached the severity the target
    actually reaches. Predicting the residual from persistence resolves the same tension
    differently — the level comes from the observed state and the head only has to supply
    the change — which is why it replaced Tweedie rather than this did.

    The severity head regresses log x under squared loss, not x under a gamma
    objective. The gamma version was tried first and produced predictions around
    3.4e4 — five orders of magnitude above a target whose positive mean is 0.0057 —
    because gamma's log link over labels spanning 1e-7 to 1 puts almost no curvature
    on the tiny ones and lets leaf values run away unbounded. Clipping to [0,1] then
    turned that into a near-constant 0.24 everywhere, i.e. an MAE of 0.24 on a target
    averaging 0.0015. Working in log space bounds the same quantity by construction.
    Note exp(mean(log x)) is a geometric mean and therefore biased low against E[x];
    that is the accepted cost of the formulation, not an oversight.
    """
    is_pos = train_df["target_x"] > threshold
    head_params = {k: v for k, v in LGBM_PARAMS.items() if k != "objective"}
    clf = lgb.LGBMClassifier(**head_params, objective="binary", verbose=-1)
    clf.fit(train_df[feature_cols], is_pos.astype(int))
    p = clf.predict_proba(val_df[feature_cols])[:, 1]

    pos_train = train_df[is_pos]
    reg = lgb.LGBMRegressor(**head_params, objective="regression", verbose=-1)
    reg.fit(pos_train[feature_cols], np.log(pos_train["target_x"]))
    severity = np.exp(reg.predict(val_df[feature_cols]))
    return np.clip(p * severity, 0, 1)


def climatology_baseline(train_df, val_df) -> np.ndarray:
    """Per-county mean of target_x over the training period — fitted, never peeking."""
    means = train_df.groupby("fips_code", observed=True)["target_x"].mean()
    return val_df["fips_code"].map(means).fillna(train_df["target_x"].mean()).to_numpy()


def lead_time_curve(val_df, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    bins = [0, 6, 12, 24, 36, 48, 72]
    bucket = pd.cut(val_df["lead_hours"], bins=bins, right=True)
    rows = []
    for label, pred in preds.items():
        for b, idx in val_df.groupby(bucket, observed=True).groups.items():
            pos = val_df.index.get_indexer(idx)
            m = metrics(val_df.loc[idx, "target_x"].to_numpy(), pred[pos])
            rows.append({"model": label, "lead_bucket": str(b), "n": len(idx), **m})
    return pd.DataFrame(rows)


def main() -> None:
    df = load_table(TABLE_PATH)
    train_df, val_df = temporal_split(df, VAL_START)
    if len(val_df) == 0:
        raise ValueError(
            f"Empty validation set at VAL_START={VAL_START.date()} — see train.py. "
            f"An ablation on a leaky random split would be worse than none."
        )
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    assert_groups_cover(feature_cols)
    y_val = val_df["target_x"].to_numpy()
    print(f"Train {len(train_df):,} / Val {len(val_df):,} rows, split at {VAL_START.date()}")
    print(f"Validation events (x>{EVENT_THRESHOLD}): {(y_val > EVENT_THRESHOLD).sum():,}\n")

    results, preds = [], {}

    def record(label: str, kind: str, pred: np.ndarray) -> None:
        results.append({"model": label, "kind": kind, **metrics(y_val, pred)})
        preds[label] = pred
        print(f"  {label:34s} MAE={results[-1]['MAE']:.6f} "
              f"RMSE={results[-1]['RMSE']:.6f} max_pred={results[-1]['max_pred']:.4f}")

    print("Baselines (no learning):")
    record("always zero", "baseline", np.zeros_like(y_val))
    record("persistence (x_at_issue)", "baseline",
           np.nan_to_num(val_df["x_at_issue"].to_numpy(), nan=0.0))
    record("county climatology", "baseline", climatology_baseline(train_df, val_df))

    print("\nFull model and feature-group ablations:")
    record("full model (delta)", "model", fit_predict(train_df, val_df, feature_cols))
    for group, cols in FEATURE_GROUPS.items():
        kept = [c for c in feature_cols if c not in cols]
        if not kept or len(kept) == len(feature_cols):
            continue
        record(f"without {group}", "ablation",
               fit_predict(train_df, val_df, kept,
                           with_persistence=(group != "autoregressive")))

    print("\nAlternative head:")
    # Two thresholds, because the choice is not obvious and it matters: x > 0 counts a
    # single customer off supply as a positive (27.4% of training rows), while
    # x > 0.001 asks about events of the size the task is really about (5.4%).
    record("hurdle (tau=0)", "model", hurdle_predict(train_df, val_df, feature_cols))
    record(f"hurdle (tau={EVENT_THRESHOLD})", "model",
           hurdle_predict(train_df, val_df, feature_cols, threshold=EVENT_THRESHOLD))

    res = pd.DataFrame(results)
    full = res.loc[res.model == "full model (delta)"].iloc[0]
    res["dRMSE_vs_full_%"] = (100 * (res["RMSE"] - full["RMSE"]) / full["RMSE"]).round(2)
    res.to_csv(OUT_CSV, index=False)

    md = to_markdown(res)
    OUT_MD.write_text(
        f"# Ablation results\n\nTemporal split at {VAL_START.date()}: "
        f"{len(train_df):,} train / {len(val_df):,} validation rows. "
        f"Events (x > {EVENT_THRESHOLD}): {(y_val > EVENT_THRESHOLD).sum():,}.\n\n"
        f"{md}\n\n"
        f"`dRMSE_vs_full_%` is positive when removing the group makes RMSE worse, "
        f"i.e. the group was contributing.\n",
        encoding="utf8",
    )

    curve = lead_time_curve(
        val_df,
        {k: v for k, v in preds.items()
         if k in ("full model (delta)", "hurdle (P(event) x severity)",
                  "always zero", "persistence (x_at_issue)")},
    )
    curve.to_csv(LEAD_CURVE_CSV, index=False)

    print(f"\nWrote {OUT_CSV.name}, {OUT_MD.name}, {LEAD_CURVE_CSV.name} to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
