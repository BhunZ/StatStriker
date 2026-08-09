"""The data quality gate.

The gate exists for scrapers that keep returning 200 after a site changes its markup: the
run stays green, the parse quietly yields fewer matches or nulls or self-fixtures, and the
result is committed and trained on. Every test below is a way that has of happening.

A gate that cannot fail is decoration, so each check is tested by breaking exactly one
thing and asserting the gate names it.
"""

import numpy as np
import pandas as pd
import pytest

import quality


@pytest.fixture
def good(matches):
    return matches


def test_clean_data_passes(good):
    assert quality.check(good) == []
    quality.assert_ok(good)      # must not raise


# --- structural -------------------------------------------------------------------

def test_an_empty_frame_is_caught(good):
    assert any("empty" in p for p in quality.check(good.iloc[0:0]))


def test_missing_columns_are_caught(good):
    assert any("required columns absent" in p
               for p in quality.check(good.drop(columns=["home_goals"])))


def test_a_null_score_is_caught(good):
    broken = good.copy()
    broken.loc[3, "home_goals"] = np.nan
    assert any("home_goals is null" in p for p in quality.check(broken))


def test_a_duplicated_fixture_is_caught(good):
    broken = pd.concat([good, good.iloc[[2]]], ignore_index=True)
    assert any("duplicate fixtures" in p for p in quality.check(broken))


def test_a_negative_score_is_caught(good):
    broken = good.copy()
    broken.loc[1, "away_goals"] = -1
    assert any("negative score" in p for p in quality.check(broken))


def test_an_absurd_score_is_caught(good):
    """A 40-0 is a parse artefact — usually a table column read as a scoreline."""
    broken = good.copy()
    broken.loc[1, "home_goals"] = 40
    assert any("score above" in p for p in quality.check(broken))


def test_a_team_playing_itself_is_caught(good):
    broken = good.copy()
    broken.loc[0, "away_team"] = broken.loc[0, "home_team"]
    assert any("playing itself" in p for p in quality.check(broken))


def test_an_unplayed_fixture_leaking_in_is_caught(good):
    broken = good.copy()
    broken.loc[0, "date"] = pd.Timestamp.today() + pd.Timedelta(days=30)
    assert any("dated after today" in p for p in quality.check(broken))


# --- comparative ------------------------------------------------------------------

def test_a_falling_match_count_is_caught(good):
    """The signature of a parser that broke while still returning 200."""
    problems = quality.check(good, previous_matches=len(good) + 50)
    assert any("match count fell" in p for p in problems)


def test_growth_is_not_a_problem(good):
    assert quality.check(good, previous_matches=len(good) - 10) == []


def test_a_tiny_dip_is_tolerated(good):
    """One source correcting a duplicate should not stop the pipeline."""
    assert quality.check(good, previous_matches=len(good) + quality.MATCH_COUNT_TOLERANCE) == []


def test_collapsed_xg_coverage_is_caught(good):
    broken = good.copy()
    broken.loc[broken.index[2:], "home_xg"] = np.nan
    assert any("xG present on only" in p for p in quality.check(broken))


def test_xg_is_not_checked_when_the_previous_run_had_none(good):
    """A first run, or a period where the xG source was known-absent, must not fail here."""
    broken = good.copy()
    broken["home_xg"] = np.nan
    assert quality.check(broken, previous_had_xg=False) == []


def test_the_first_ever_run_has_nothing_to_compare_against(good):
    assert quality.check(good, previous_matches=None) == []


# --- how it reports ---------------------------------------------------------------

def test_assert_ok_raises_and_names_every_problem(good):
    broken = good.copy()
    broken.loc[1, "away_goals"] = -1
    broken.loc[0, "away_team"] = broken.loc[0, "home_team"]

    with pytest.raises(quality.DataQualityError) as exc:
        quality.assert_ok(broken)

    message = str(exc.value)
    assert "negative score" in message and "playing itself" in message
    assert "Nothing has been retrained or committed" in message


def test_structural_failures_short_circuit_the_comparative_ones(good):
    """No point complaining that the count fell when the columns are not even there."""
    problems = quality.check(good.drop(columns=["home_goals"]), previous_matches=10_000)
    assert len(problems) == 1
