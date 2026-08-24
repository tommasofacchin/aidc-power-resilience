"""
Validate a predictions CSV against the submission guidelines (see PLAN.md section 6
for the checklist this implements). Run before every submission attempt:

    python code/validate_submission.py submission/predictions.csv

Checks, each tied to a specific guideline requirement:
- Schema: exact columns, fips_code is a 5-digit zero-padded string, timestamps are
  ISO 8601 UTC with a 'Z' suffix, predicted_x in [0, 1] with no NaN and written in
  fixed-point rather than scientific notation (a warning, not an error).
- Row count per batch: Task A = 48 rows (issue_time, target_time) covering +1h..+48h;
  Task B = 24 rows covering +15m..+6h. Partial/truncated batches are non-compliant
  per the guidelines' explicit wording.
- County coverage: every batch contains all 5 reporting counties, nothing more,
  nothing less.
- Issuance frequency: >=1 Task A batch per calendar day, >=1 Task B batch per 6h,
  across the announced test window.
- Test window coverage: every day of 2025-09-01..2025-11-30 has to be reachable by at
  least one batch's target_time range.

Deliberately does NOT flag overlapping target_time rows across different issue_times as
an error — the guidelines say this is expected and should not be deduplicated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

# The announced window is 1 Sep - 30 Nov 2025 INCLUSIVE of 30 Nov, so the exclusive
# upper bound is the start of 1 Dec. Using 30 Nov as the exclusive bound (as this
# originally did) silently drops the last day from both the coverage check and the
# in-window filter, letting a submission missing all of 30 Nov pass validation.
TEST_WINDOW_START = pd.Timestamp("2025-09-01", tz="UTC")
TEST_WINDOW_END_EXCLUSIVE = pd.Timestamp("2025-12-01", tz="UTC")

TASK_SPEC = {
    "A": {"n_rows": 48, "min_lead": pd.Timedelta(hours=1), "max_lead": pd.Timedelta(hours=48),
          "step": pd.Timedelta(hours=1), "min_issuance_gap": pd.Timedelta(days=1)},
    "B": {"n_rows": 24, "min_lead": pd.Timedelta(minutes=15), "max_lead": pd.Timedelta(hours=6),
          "step": pd.Timedelta(minutes=15), "min_issuance_gap": pd.Timedelta(hours=6)},
}

REQUIRED_COLUMNS = ["task_id", "fips_code", "county", "state", "issue_time", "target_time", "predicted_x"]


class ValidationResult:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors

    def report(self) -> None:
        if self.errors:
            print(f"FAIL — {len(self.errors)} error(s):")
            for e in self.errors[:50]:
                print(f"  [ERROR] {e}")
            if len(self.errors) > 50:
                print(f"  ... and {len(self.errors) - 50} more")
        else:
            print("PASS — no errors.")
        if self.warnings:
            print(f"\n{len(self.warnings)} warning(s):")
            for w in self.warnings[:20]:
                print(f"  [WARN] {w}")


def load_raw(path: Path) -> pd.DataFrame:
    # dtype=str everywhere at first: we want to catch formatting problems (e.g. a
    # zero-padded fips_code silently stripped to an int by pandas' default inference)
    # rather than have pandas paper over them before we get a chance to check.
    return pd.read_csv(path, dtype=str)


def check_schema(df: pd.DataFrame, result: ValidationResult) -> pd.DataFrame | None:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(REQUIRED_COLUMNS)
    if missing:
        result.error(f"Missing required columns: {sorted(missing)}")
    if extra:
        result.warn(f"Unexpected extra columns (not rejected, just noted): {sorted(extra)}")
    if missing:
        return None  # can't meaningfully continue

    bad_task = df[~df["task_id"].isin(["A", "B"])]
    if len(bad_task):
        result.error(f"{len(bad_task)} row(s) with task_id not in {{A, B}}: "
                     f"{bad_task['task_id'].unique().tolist()}")

    bad_fips = df[~df["fips_code"].str.match(r"^\d{5}$", na=True)]
    if len(bad_fips):
        result.error(f"{len(bad_fips)} row(s) with fips_code not a 5-digit zero-padded "
                      f"string (check the file wasn't opened/resaved in Excel, which "
                      f"strips leading zeros): sample={bad_fips['fips_code'].head(5).tolist()}")

    for col in ["issue_time", "target_time"]:
        bad_fmt = df[~df[col].str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", na=True)]
        if len(bad_fmt):
            result.error(f"{len(bad_fmt)} row(s) with {col} not ISO 8601 UTC with 'Z' "
                          f"suffix: sample={bad_fmt[col].head(3).tolist()}")

    px = pd.to_numeric(df["predicted_x"], errors="coerce")
    n_nan = px.isna().sum()
    if n_nan:
        result.error(f"{n_nan} row(s) with predicted_x missing or non-numeric")
    out_of_range = px[(px < 0) | (px > 1)]
    if len(out_of_range.dropna()):
        result.error(f"{len(out_of_range.dropna())} row(s) with predicted_x outside [0, 1] "
                      f"(min={px.min()}, max={px.max()})")

    # A warning, not an error: the guidelines only require a float in [0, 1], and every
    # parser reads 9.95e-05 correctly. But PLAN.md's checklist asks for no scientific
    # notation, and pandas' default float repr produces it below 1e-4 unless the writer
    # passes float_format — which is easy to lose in a refactor and invisible afterwards.
    sci = df["predicted_x"].str.contains(r"[eE]", na=False)
    if sci.any():
        result.warn(f"{int(sci.sum())} row(s) render predicted_x in scientific notation "
                    f"(e.g. {df.loc[sci, 'predicted_x'].iloc[0]}). Readable by any parser, "
                    f"but predict.py is meant to write fixed-point — check float_format.")

    return df


def parse(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True, errors="coerce")
    df["target_time"] = pd.to_datetime(df["target_time"], utc=True, errors="coerce")
    df["predicted_x"] = pd.to_numeric(df["predicted_x"], errors="coerce")
    return df


def check_batches(df: pd.DataFrame, expected_counties: set[str] | None, result: ValidationResult) -> None:
    for task_id, spec in TASK_SPEC.items():
        task_df = df[df["task_id"] == task_id]
        if task_df.empty:
            result.error(f"Task {task_id}: no rows at all")
            continue

        counties_per_batch = task_df.groupby("issue_time")["fips_code"].nunique()
        rows_per_county_batch = task_df.groupby(["issue_time", "fips_code"]).size()

        bad_row_count = rows_per_county_batch[rows_per_county_batch != spec["n_rows"]]
        if len(bad_row_count):
            result.error(
                f"Task {task_id}: {len(bad_row_count)} (issue_time, county) batch(es) "
                f"don't have exactly {spec['n_rows']} rows (partial/truncated batch — "
                f"non-compliant per the guidelines). Examples: "
                f"{bad_row_count.head(3).to_dict()}"
            )

        if expected_counties is not None:
            n_expected = len(expected_counties)
            bad_coverage = counties_per_batch[counties_per_batch != n_expected]
            if len(bad_coverage):
                result.error(
                    f"Task {task_id}: {len(bad_coverage)} issue_time(s) don't cover all "
                    f"{n_expected} reporting counties. Examples: {bad_coverage.head(3).to_dict()}"
                )
            actual_counties = set(task_df["fips_code"].unique())
            unexpected = actual_counties - expected_counties
            if unexpected:
                result.error(f"Task {task_id}: predictions for counties outside the "
                             f"declared 5-county selection: {sorted(unexpected)}")

        # The guidelines require the *complete, exact* set of lead times per batch, so
        # check set equality rather than just a count and a range — 48 rows all sitting
        # at +1h would otherwise pass both of those.
        expected_leads = set(
            pd.timedelta_range(spec["min_lead"], spec["max_lead"], freq=spec["step"])
        )
        actual_leads = (task_df["target_time"] - task_df["issue_time"])
        bad_batches = 0
        for _, group in task_df.groupby(["issue_time", "fips_code"]):
            got = set(group["target_time"] - group["issue_time"])
            if got != expected_leads:
                bad_batches += 1
        if bad_batches:
            result.error(
                f"Task {task_id}: {bad_batches} (issue_time, county) batch(es) don't contain "
                f"exactly the required lead times "
                f"({spec['min_lead']}..{spec['max_lead']} every {spec['step']})"
            )

        out_of_range = task_df[(actual_leads < spec["min_lead"]) | (actual_leads > spec["max_lead"])]
        if len(out_of_range):
            result.error(f"Task {task_id}: {len(out_of_range)} row(s) with lead time outside "
                         f"[{spec['min_lead']}, {spec['max_lead']}]")

        # Minimum issuance frequency is a hard requirement in the guidelines, not a
        # nice-to-have, so a gap that exceeds it is an error rather than a warning.
        issue_times = sorted(task_df["issue_time"].unique())
        gaps = pd.Series(issue_times).diff().dropna()
        too_sparse = gaps[gaps > spec["min_issuance_gap"]]
        if len(too_sparse):
            result.error(
                f"Task {task_id}: {len(too_sparse)} gap(s) between consecutive issue_times "
                f"exceed the required minimum frequency ({spec['min_issuance_gap']}). "
                f"Largest gap: {gaps.max()}"
            )


def check_test_window_coverage(df: pd.DataFrame, result: ValidationResult) -> None:
    in_window = df[
        (df["target_time"] >= TEST_WINDOW_START) & (df["target_time"] < TEST_WINDOW_END_EXCLUSIVE)
    ]
    if in_window.empty:
        result.error("No target_time rows fall inside the test window "
                      f"[{TEST_WINDOW_START}, {TEST_WINDOW_END_EXCLUSIVE})")
        return
    covered_days = set(in_window["target_time"].dt.date)
    all_days = set(
        pd.date_range(TEST_WINDOW_START, TEST_WINDOW_END_EXCLUSIVE, freq="D", inclusive="left").date
    )
    missing_days = sorted(all_days - covered_days)
    if missing_days:
        result.error(f"{len(missing_days)} day(s) of the test window have no target_time "
                      f"predictions at all: {missing_days[:5]}"
                      f"{'...' if len(missing_days) > 5 else ''}")


def validate(path: Path, expected_counties: set[str] | None = None) -> ValidationResult:
    result = ValidationResult()
    raw = load_raw(path)
    df = check_schema(raw, result)
    if df is None:
        return result
    df = parse(df)
    check_batches(df, expected_counties, result)
    check_test_window_coverage(df, result)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python code/validate_submission.py <path-to-predictions.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    selected_path = PROJECT_ROOT / "data" / "processed" / "selected_counties.csv"
    expected = None
    if selected_path.exists():
        expected = set(pd.read_csv(selected_path, dtype={"fips_code": str})["fips_code"])

    result = validate(csv_path, expected_counties=expected)
    result.report()
    sys.exit(0 if result.ok else 1)
