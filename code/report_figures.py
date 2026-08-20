"""
Figures for the technical report. Regenerates every plot the PDF embeds, from the same
model bundle and held-out split the numbers in the text come from — so a figure can
never quietly disagree with the table beside it.

Output goes to report/figures/ as PNG at 200 dpi (the report is built by printing HTML
to PDF, so raster at print resolution is simpler than juggling SVG fonts).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blend import LEAD_EDGES, apply_blend, bucket_of
from data_acquisition.eagle_i import PROCESSED_DIR
from model_bundle import load_bundle
from train import VAL_START, load_table, temporal_split

import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "report" / "figures"
TABLE_PATH = PROCESSED_DIR / "training_table_partial.parquet"
BUNDLE_DIR = PROCESSED_DIR / "model_bundle"

EVENT_THRESHOLD = 0.001

# Categorical slots 1-3 of the reference palette, validated for all-pairs CVD
# separation on a light surface. "always zero" is drawn as a neutral reference line
# rather than a fourth hue: it is a floor to measure against, not a competing method.
COLORS = {
    "persistence": "#2a78d6",
    "model": "#eb6834",
    "blended": "#1baf7a",
}
REFERENCE_INK = "#8a8985"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def skill_by_lead(val_df: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """RMSE and event-restricted MAE per lead bucket, for each named prediction."""
    buckets = bucket_of(val_df["lead_hours"])
    truth = val_df["target_x"].to_numpy()
    events = truth > EVENT_THRESHOLD

    rows = []
    for name, p in preds.items():
        for b in buckets.cat.categories:
            sel = (buckets == b).to_numpy()
            rows.append({
                "model": name,
                "bucket": str(b),
                "n": int(sel.sum()),
                "rmse": float(np.sqrt(((p[sel] - truth[sel]) ** 2).mean())),
                "mae_events": float(np.abs(p[sel & events] - truth[sel & events]).mean()),
            })
    return pd.DataFrame(rows)


def plot_skill_curve(curve: pd.DataFrame, out_path: Path) -> None:
    """Skill relative to the always-zero floor, per lead bucket.

    Plotted as a skill score rather than as raw RMSE per bucket on purpose. Raw error
    is not comparable across buckets here: outage events are rare and clustered, so
    each bucket catches a different set of storms and the raw curve zigzags by up to
    50% for reasons that have nothing to do with lead time. Normalising each bucket by
    its own always-zero error cancels that composition effect and leaves the question
    the guidelines actually ask — how much skill survives as the horizon lengthens.
    """
    labels = [str(b) for b in bucket_of(pd.Series(LEAD_EDGES)).cat.categories]
    x = np.arange(len(labels))
    tick_labels = ["0-6h", "6-12h", "12-24h", "24-48h", "48-72h"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.5), dpi=200)
    panels = [
        ("rmse", "RMSE, all held-out rows"),
        ("mae_events", f"MAE, rows with observed x > {EVENT_THRESHOLD}"),
    ]

    for ax, (metric, title) in zip(axes, panels):
        floor = curve[curve.model == "always zero"].set_index("bucket").loc[labels, metric]
        ax.axhline(0, color=REFERENCE_INK, lw=1.1, ls=(0, (4, 3)), zorder=2)
        ax.annotate(
            "always-zero baseline", (x[-1], 0), textcoords="offset points", xytext=(0, 4),
            fontsize=7, color=REFERENCE_INK, ha="right", va="bottom",
        )

        ends = []
        for name, color in COLORS.items():
            series = curve[curve.model == name].set_index("bucket").loc[labels, metric]
            skill = 100 * (1 - series.to_numpy() / floor.to_numpy())
            ax.plot(x, skill, color=color, lw=1.8, marker="o", ms=4.5,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3,
                    label=name, clip_on=False)
            ends.append((skill[-1], name, color))

        # The model and blended curves converge at long lead, so their end labels land
        # on top of each other. Push them apart vertically, keeping the reading order.
        span = np.ptp([e[0] for e in ends] + [0])
        min_gap = max(span * 0.16, 0.35)
        placed_y = None
        for y, name, color in sorted(ends, reverse=True):
            placed_y = y if placed_y is None else min(y, placed_y - min_gap)
            ax.annotate(
                name, (x[-1], placed_y), textcoords="offset points", xytext=(6, 0),
                fontsize=7.5, color=color, va="center", annotation_clip=False,
            )

        ax.set_title(title, fontsize=8.5, color=TEXT_PRIMARY, pad=6, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=7.5, color=TEXT_SECONDARY)
        ax.set_xlim(-0.2, len(labels) - 0.8)
        ax.tick_params(axis="y", labelsize=7.5, colors=TEXT_SECONDARY, length=0)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%" if v else "0")
        ax.grid(axis="y", color="#e6e5e1", lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#d9d8d3")
        ax.set_xlabel("lead time (target_time - issue_time)", fontsize=7.5, color=TEXT_SECONDARY)

    axes[0].set_ylabel("error reduction vs always zero", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout(pad=0.6, w_pad=3.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    df = load_table(TABLE_PATH)
    _, val_df = temporal_split(df, VAL_START)

    bundle = load_bundle(BUNDLE_DIR)
    X = val_df[bundle.feature_names].copy()
    X["fips_code"] = bundle.encode_fips(X["fips_code"])
    model_pred = np.clip(bundle.booster.predict(X), 0, 1)

    weights = json.loads((BUNDLE_DIR / "blend_weights.json").read_text(encoding="utf8"))
    persistence = val_df["x_at_issue"].to_numpy()
    blended = apply_blend(model_pred, persistence, val_df["lead_hours"].to_numpy(), weights)

    # Persistence is undefined where EAGLE-I had no reading at issue_time (0.3% of
    # held-out rows). Scoring every series on the same subset keeps the comparison
    # honest; scoring persistence on fewer rows than the model would not.
    usable = ~np.isnan(persistence)
    val_df = val_df.loc[usable].reset_index(drop=True)
    preds = {
        "always zero": np.zeros(usable.sum()),
        "persistence": persistence[usable],
        "model": model_pred[usable],
        "blended": blended[usable],
    }
    print(f"Scoring {usable.sum():,} of {len(usable):,} held-out rows "
          f"({100 * (~usable).mean():.2f}% dropped: no EAGLE-I reading at issue_time)")

    curve = skill_by_lead(val_df, preds)
    curve.to_csv(PROCESSED_DIR / "report_skill_by_lead.csv", index=False)
    print(curve.pivot(index="bucket", columns="model", values="rmse").to_string())
    print()
    print(curve.pivot(index="bucket", columns="model", values="mae_events").to_string())

    plot_skill_curve(curve, FIG_DIR / "skill_by_lead.png")


if __name__ == "__main__":
    main()
