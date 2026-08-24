"""
Assemble the submission package in the structure the guidelines suggest.

Why this exists rather than a submission/ folder that already holds everything. The
guidelines require three components — report, predictions CSV, and the complete codebase
with its reproducibility guide — and warn that a submission missing any one of them is
not scored. Their suggested layout puts `code/` and `README.md` *inside* `submission/`,
but this repository keeps them at the root, where they are the working copy. Copying them
into `submission/` permanently would leave two versions of every file in one repository,
free to drift, with no way for a reviewer to tell which one produced the results.

So the package is derived, never edited: this script builds `dist/submission/` (and a zip
beside it) from the working copy each time, and refuses to build if the inputs look stale.
Delete `dist/` and rebuild whenever anything changes; nothing else reads it.

    python code/make_submission_package.py            # -> dist/submission/ + dist/submission.zip
    python code/make_submission_package.py --no-zip   # directory only

What it checks before packing, because these are the failures that would cost the whole
submission rather than a few points:
  - predictions.csv passes validate_submission.py (the same check, not a re-implementation)
  - report.pdf is between 3 and 8 pages
  - report.pdf in submission/ matches report/report.pdf byte for byte
  - predictions.csv is not older than the model bundle that should have produced it
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from validate_submission import validate  # noqa: E402

DIST_DIR = PROJECT_ROOT / "dist"
PACKAGE_DIR = DIST_DIR / "submission"
ZIP_PATH = DIST_DIR / "submission.zip"

PREDICTIONS = PROJECT_ROOT / "submission" / "predictions.csv"
SUBMITTED_PDF = PROJECT_ROOT / "submission" / "report.pdf"
SOURCE_PDF = PROJECT_ROOT / "report" / "report.pdf"
CODE_DIR = PROJECT_ROOT / "code"
README = PROJECT_ROOT / "README.md"
MODEL_BUNDLE = PROJECT_ROOT / "data" / "processed" / "model_bundle" / "model.txt"

# Working files that are not part of the deliverable. `data/` is git-ignored and holds
# gigabytes; it is not under code/ anyway, but the exclusions below are what keeps the
# package from carrying caches and editor droppings into a reviewer's checkout.
EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log"}
EXCLUDE_NAMES = {".gitkeep"}  # placeholders that only exist to keep empty dirs in git

MIN_PAGES, MAX_PAGES = 3, 8


def show(path: Path) -> str:
    """Repo-relative where possible, absolute otherwise.

    relative_to() raises on a path outside the project, and the only caller is the code
    that reports what is missing — so the naive version turns "tell the user which file
    is absent" into a traceback exactly when the user needs the message.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def pdf_page_count(path: Path) -> int:
    """Count pages without a PDF library — /Type /Page objects, excluding /Pages."""
    return len(re.findall(rb"/Type\s*/Page[^s]", path.read_bytes()))


def preflight() -> list[str]:
    problems: list[str] = []

    for path in (PREDICTIONS, SUBMITTED_PDF, SOURCE_PDF, README):
        if not path.exists():
            problems.append(f"missing: {show(path)}")
    if problems:
        return problems

    result = validate(PREDICTIONS)
    if not result.ok:
        problems.append(f"predictions.csv fails validation ({len(result.errors)} error(s)); "
                        f"run `python code/validate_submission.py {PREDICTIONS}` to see them")
    for warning in result.warnings:
        print(f"  note: {warning}")

    pages = pdf_page_count(SUBMITTED_PDF)
    if not MIN_PAGES <= pages <= MAX_PAGES:
        problems.append(f"report.pdf is {pages} pages, outside the guidelines' "
                        f"{MIN_PAGES}-{MAX_PAGES}")

    if SUBMITTED_PDF.read_bytes() != SOURCE_PDF.read_bytes():
        problems.append("submission/report.pdf differs from report/report.pdf — the built "
                        "report was not copied across; re-print the PDF and copy it")

    # A predictions file older than the model that should have produced it means the last
    # training run never reached predict.py, which is silent and easy to miss.
    if MODEL_BUNDLE.exists() and PREDICTIONS.stat().st_mtime < MODEL_BUNDLE.stat().st_mtime:
        problems.append("submission/predictions.csv is older than the model bundle — "
                        "re-run `python code/predict.py` before packaging")

    return problems


def copy_code(destination: Path) -> int:
    def ignore(directory: str, names: list[str]) -> set[str]:
        skip = {n for n in names if n in EXCLUDE_DIRS or n in EXCLUDE_NAMES}
        skip |= {n for n in names if Path(n).suffix in EXCLUDE_SUFFIXES}
        return skip

    shutil.copytree(CODE_DIR, destination, ignore=ignore)
    return sum(1 for _ in destination.rglob("*") if _.is_file())


def build(make_zip: bool = True) -> None:
    problems = preflight()
    if problems:
        print("Refusing to package:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)

    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    PACKAGE_DIR.mkdir(parents=True)

    shutil.copy2(SUBMITTED_PDF, PACKAGE_DIR / "report.pdf")
    shutil.copy2(PREDICTIONS, PACKAGE_DIR / "predictions.csv")
    shutil.copy2(README, PACKAGE_DIR / "README.md")
    n_code = copy_code(PACKAGE_DIR / "code")

    files = sorted(p for p in PACKAGE_DIR.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"Built {show(PACKAGE_DIR)}: {len(files)} files "
          f"({n_code} under code/), {total / 1e6:.1f} MB")
    print(f"  report.pdf       {pdf_page_count(PACKAGE_DIR / 'report.pdf')} pages")
    print(f"  predictions.csv  {sum(1 for _ in (PACKAGE_DIR / 'predictions.csv').open(encoding='utf8')) - 1:,} rows")

    if make_zip:
        with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in files:
                zf.write(path, path.relative_to(DIST_DIR))
        print(f"Wrote {show(ZIP_PATH)} "
              f"({ZIP_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble the submission package.")
    parser.add_argument("--no-zip", action="store_true", help="build the directory only")
    build(make_zip=not parser.parse_args().no_zip)
