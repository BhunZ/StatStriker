"""
quality.py
----------
Data quality gate for the weekly pipeline.

The pipeline has run unattended every Monday since April and commits whatever it scrapes.
Nothing has ever inspected that data before it lands. The failure mode that matters is not
a scraper that dies — that shows up as a red run — but one that keeps returning 200s after
a site changes its markup, so the parse yields fewer matches, or nulls, or team names that
no longer resolve. Those commit cleanly and are only noticed later, if at all.

So: run these checks between processing and saving. Anything that fails stops the run
before models are retrained on it and before the result is committed.

Two kinds of check:

* **Structural** — things that are wrong on their own terms: duplicate fixtures, missing
  scores, negative goals, dates in the future.
* **Comparative** — things only wrong relative to last week: the match count falling, xG
  coverage collapsing. These need the previous run's numbers, which live in
  ``pipeline_state.json``.

A comparative check with no baseline (first ever run) passes rather than guessing.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["date", "home_team", "away_team", "home_goals", "away_goals"]

#: A Premier League scoreline above this is a parsing artefact, not a football result.
MAX_PLAUSIBLE_GOALS = 15
#: The match count may not fall at all — fixtures are never un-played. A small tolerance
#: absorbs a source correcting a duplicate, but a real parse failure drops far more.
MATCH_COUNT_TOLERANCE = 2
#: xG comes from a second source; if that source breaks, coverage collapses rather than
#: dipping. Anything below this against a previous run that had xG is a broken join.
MIN_XG_COVERAGE = 0.80
#: Matches dated beyond today are fixtures that leaked into the results table.
FUTURE_DATE_GRACE_DAYS = 1


class DataQualityError(RuntimeError):
    """Raised when processed data fails a check. Stops the run before retraining."""


def _structural_problems(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"required columns absent: {missing}"]

    if df.empty:
        return ["processed dataset is empty"]

    for col in ("home_goals", "away_goals", "date", "home_team", "away_team"):
        n_null = int(df[col].isna().sum())
        if n_null:
            problems.append(f"{col} is null on {n_null} of {len(df)} matches")

    key = ["date", "home_team", "away_team"]
    n_dup = int(df.duplicated(key).sum())
    if n_dup:
        problems.append(f"{n_dup} duplicate fixtures on {key} — the same match counted twice")

    goals = df[["home_goals", "away_goals"]].apply(pd.to_numeric, errors="coerce")
    n_neg = int((goals < 0).any(axis=1).sum())
    if n_neg:
        problems.append(f"{n_neg} matches have a negative score")
    n_absurd = int((goals > MAX_PLAUSIBLE_GOALS).any(axis=1).sum())
    if n_absurd:
        problems.append(f"{n_absurd} matches score above {MAX_PLAUSIBLE_GOALS} — parsed wrong")

    n_self = int((df["home_team"] == df["away_team"]).sum())
    if n_self:
        problems.append(f"{n_self} matches have a team playing itself")

    cutoff = pd.Timestamp(date.today() + timedelta(days=FUTURE_DATE_GRACE_DAYS))
    n_future = int((pd.to_datetime(df["date"], errors="coerce") > cutoff).sum())
    if n_future:
        problems.append(f"{n_future} matches are dated after today — unplayed fixtures "
                        f"have leaked into the results")

    return problems


def _comparative_problems(df: pd.DataFrame, previous_matches: int | None,
                          previous_had_xg: bool) -> list[str]:
    problems: list[str] = []

    if previous_matches:
        if len(df) < previous_matches - MATCH_COUNT_TOLERANCE:
            problems.append(
                f"match count fell from {previous_matches} to {len(df)} — fixtures are "
                f"never un-played, so a fall means the parse lost rows")

    if previous_had_xg and "home_xg" in df.columns:
        coverage = float(df["home_xg"].notna().mean())
        if coverage < MIN_XG_COVERAGE:
            problems.append(
                f"xG present on only {coverage:.0%} of matches (was above "
                f"{MIN_XG_COVERAGE:.0%}) — the second source or its join has broken")

    return problems


def check(df: pd.DataFrame, previous_matches: int | None = None,
          previous_had_xg: bool = True) -> list[str]:
    """Return every problem found. An empty list means the data is fit to train on."""
    problems = _structural_problems(df)
    if not problems:                     # comparative checks assume the frame is sane
        problems += _comparative_problems(df, previous_matches, previous_had_xg)
    return problems


def assert_ok(df: pd.DataFrame, previous_matches: int | None = None,
              previous_had_xg: bool = True) -> None:
    """Raise `DataQualityError` if the data is not fit to train on."""
    problems = check(df, previous_matches, previous_had_xg)
    if not problems:
        logger.info("Data quality: %d matches, all checks passed", len(df))
        return

    detail = "\n".join(f"  - {p}" for p in problems)
    raise DataQualityError(
        f"processed data failed {len(problems)} quality check(s):\n{detail}\n"
        f"Nothing has been retrained or committed. Inspect data/processed/ before rerunning."
    )
