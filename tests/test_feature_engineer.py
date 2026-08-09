"""Rolling-feature construction, and the one property everything else rests on.

If a match-day feature contains any part of that match's own result, every metric this
project reports is inflated and none of them mean anything. The guard is a `.shift(1)`
before each `.rolling()` and `.ewm()`; these tests exist so removing it fails loudly
rather than quietly improving the numbers.

The fixture makes each team score a fixed number of goals, so the expected value of a
rolling mean is known by hand rather than recomputed with the same code under test.
"""

import numpy as np
import pandas as pd
import pytest

from processors.feature_engineer import FeatureEngineer


@pytest.fixture
def engineered(matches):
    return FeatureEngineer(save_parquet=False).engineer(matches)


def test_engineering_preserves_every_match(matches, engineered):
    assert len(engineered) == len(matches)


def test_a_teams_first_match_has_no_history(engineered):
    """Nothing is known before the first game, so the feature must be missing — not zero,
    which would read downstream as 'scored nothing'."""
    first = engineered.sort_values("date").groupby("home_team").head(1)
    assert first["home_goals_scored_5"].isna().all()


def test_a_rolling_feature_uses_only_earlier_matches(engineered):
    """The direct leakage check, done per home side because the rolling views are built
    separately for home and away."""
    checked = 0
    for team, side in [(t, "home") for t in engineered["home_team"].unique()]:
        sub = engineered[engineered[f"{side}_team"] == team].sort_values("date")
        goals = sub[f"{side}_goals"].tolist()
        feat = sub[f"{side}_goals_scored_5"].tolist()
        for i in range(1, len(sub)):
            expected_past = np.mean(goals[max(0, i - 5):i])
            assert feat[i] == pytest.approx(expected_past), (
                f"{team} match {i}: feature {feat[i]} != mean of earlier matches "
                f"{expected_past} — the current match has leaked in")
            checked += 1
    assert checked > 0, "fixture produced nothing to check"


def test_changing_a_result_cannot_change_that_match_own_feature(matches):
    """The sharpest form of the same question: rewrite one match's score and its own
    features must not move. Anything that moves has read the future."""
    base = FeatureEngineer(save_parquet=False).engineer(matches)

    tampered = matches.copy()
    target = len(tampered) // 2
    tampered.loc[target, "home_goals"] = 99
    after = FeatureEngineer(save_parquet=False).engineer(tampered)

    rolling_cols = [c for c in base.columns
                    if any(k in c for k in ("_scored_", "_conceded_", "_ewm", "form"))]
    assert rolling_cols, "no rolling columns found — has the naming changed?"

    row_before = base.loc[target, rolling_cols]
    row_after = after.loc[target, rolling_cols]
    pd.testing.assert_series_equal(row_before, row_after, check_names=False)


def test_a_later_match_does_pick_the_change_up(matches):
    """The mirror of the test above: if nothing ever changed, the features would be
    constants and the test above would pass for the wrong reason."""
    base = FeatureEngineer(save_parquet=False).engineer(matches)
    tampered = matches.copy()
    tampered.loc[0, "home_goals"] = 99
    after = FeatureEngineer(save_parquet=False).engineer(tampered)

    team = matches.loc[0, "home_team"]
    later = base[(base["home_team"] == team) & (base.index > 0)].index
    assert len(later) > 0, "fixture gives the team no later home match"
    idx = later[0]
    assert base.loc[idx, "home_goals_scored_5"] != after.loc[idx, "home_goals_scored_5"]


def test_rows_stay_in_date_order(engineered):
    assert engineered["date"].is_monotonic_increasing
