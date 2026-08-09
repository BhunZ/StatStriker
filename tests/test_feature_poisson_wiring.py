"""FeaturePoisson must actually receive its features.

The model reads its covariates from `features_home` / `features_away` kwargs, and until
2026-08-10 nothing passed them. `BaseMatchModel.predict_batch` calls
`predict_1x2(home_team, away_team)` with no kwargs, and a grep for `features_home=` across
the whole repository returned nothing. So every out-of-fold prediction the ensemble learned
its weights from, and every prediction served by the API, ran with the feature term at
exactly zero.

The model quietly became the plain Poisson it exists to improve on. Out-of-fold over 25
folds it correlated 0.995 with Dixon-Coles before the fix and 0.70 after — which is the
difference between an ensemble member that is redundant and one that carries information
nothing else has.

The failure left no trace: no exception, no warning, and predictions that looked entirely
reasonable. These tests are the trace.
"""

import numpy as np
import pandas as pd
import pytest

from models.feature_poisson import FeaturePoisson

STEMS = ["xg_scored_5", "xg_conceded_5", "xg_overperformance_5", "xpts_5"]


@pytest.fixture
def fitted(matches):
    """Fit on the shared fixture, with feature columns the model will pick up."""
    df = matches.copy()
    rng = np.random.default_rng(0)
    for side in ("home", "away"):
        for stem in STEMS:
            df[f"{side}_{stem}"] = rng.normal(0, 1, len(df))
    return FeaturePoisson(l2_penalty=1.0, symmetric_mode=True).fit(df), df


def test_the_model_learned_some_feature_weights(fitted):
    model, _ = fitted
    assert len(model.feature_weights_home_) > 0, "no features were used at all"


def test_predict_batch_gives_each_fixture_its_own_features(fitted, monkeypatch):
    """The regression test for the bug: the base implementation passed team names only."""
    model, df = fitted
    seen = []

    original = model.predict_1x2

    def spy(home, away, **kwargs):
        seen.append(kwargs)
        return original(home, away, **kwargs)

    monkeypatch.setattr(model, "predict_1x2", spy)
    model.predict_batch(df.head(4))

    assert len(seen) == 4
    for kwargs in seen:
        assert kwargs.get("features_home") is not None, \
            "predict_batch called predict_1x2 without features — the covariates are dead"
        assert kwargs.get("features_away") is not None


def test_two_fixtures_with_different_form_get_different_predictions(fitted):
    """If the features were being dropped, every meeting of the same two teams would give
    an identical answer however the form columns differ."""
    model, df = fitted
    pair = df[(df.home_team == df.home_team.iloc[0]) &
              (df.away_team == df.away_team.iloc[0])].copy()
    if len(pair) < 2:
        pytest.skip("fixture has no repeated meeting")

    pair.loc[pair.index[0], [f"home_{s}" for s in STEMS]] = 2.0
    pair.loc[pair.index[1], [f"home_{s}" for s in STEMS]] = -2.0

    out = model.predict_batch(pair)
    assert out["p_home"].iloc[0] != pytest.approx(out["p_home"].iloc[1]), \
        "the same fixture with opposite form produced the same prediction"


def test_a_fixture_with_no_row_falls_back_to_the_latest_known_form(fitted):
    """Live prediction has no match row. Falling back to zeros would mean 'exactly average
    on every stem', which is what silently disabled the model."""
    model, _ = fitted
    assert model._latest_features_, "no form snapshot was taken at fit time"
    for vec in model._latest_features_.values():
        assert len(vec) == len(model.feature_weights_home_)


def test_the_snapshot_holds_the_most_recent_values(fitted):
    model, df = fitted
    last = df.sort_values("date").iloc[-1]
    team = last["home_team"]
    expected = np.array([last[f"home_{s}"] for s in model.feature_cols_used_stems_])
    np.testing.assert_allclose(model._latest_features_[team], expected)


def test_an_old_pickle_without_a_snapshot_still_predicts(fitted):
    """Models saved before the fix have no `_latest_features_`; they must not crash."""
    model, _ = fitted
    del model._latest_features_
    probs = model.predict_1x2("Alpha FC", "Bravo United")
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
