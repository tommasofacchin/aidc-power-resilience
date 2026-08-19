"""
The one centralized causality check (PLAN.md section 2.4): every feature must be a
function exclusively of data with timestamp <= issue_time. Scattered ad-hoc checks are
how this kind of leakage slips through in a multi-week project; one assert everything
routes through is how it doesn't.
"""

from __future__ import annotations

import pandas as pd


class CausalityViolation(Exception):
    pass


def assert_causal(timestamps: pd.Series, issue_time: pd.Timestamp, context: str) -> None:
    """Raise if any timestamp in `timestamps` is later than issue_time.

    `context` should name what's being checked (e.g. "autoregressive EAGLE-I lookback",
    "IFS run_time") so a violation is immediately actionable, not just a bare assert.
    """
    future = timestamps[timestamps > issue_time]
    if len(future):
        raise CausalityViolation(
            f"{context}: {len(future)} timestamp(s) after issue_time {issue_time} "
            f"(latest: {future.max()}). A prediction cannot condition on the future."
        )
