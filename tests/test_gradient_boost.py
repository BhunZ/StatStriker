"""The one model that is not a Poisson variant.

It exists because the other four are the same thing in different clothes — generative
Poisson models that estimate a scoring rate per side and read 1X2 off the grid — and out
of fold they correlate 0.94-0.997, so blending them bought almost nothing.

Two things about it need guarding. It must never see the match's own result. And its
probabilities must stay believable:
boosted trees are sharp and uncalibrated, and this one reached 0.97 on a market where the
sharpest bookmakers rarely pass 0.90.
"""

import numpy as np
import pandas as pd
import pytest

from models.gradient_boost import GradientBoostModel, safe_feature_columns

STEMS = ["goals_scored_5", "goals_conceded_5", "xg_scored_5", "xg_conceded_5"]


@pytest.fixture
def frame(matches):
    df = matches.copy()
    rng = np.random.default_rng(0)
    for side in ("home", "away"):
        for stem in STEMS:
            df[f"{side}_{stem}"] = rng.normal(1.5, 0.5, len(df))
        df[f"{side}_ctx_stats_performance_gls"] = 86.0     # a season total
    return pd.concat([df] * 12, ignore_index=True).assign(
        date=pd.date_range("2024-08-01", periods=len(df) * 12, freq="D"))


# --- what the model is allowed to see ---------------------------------------------

def test_the_matchs_own_result_is_never_a_feature(frame):
    cols = safe_feature_columns(frame)
    for banned in ("home_goals", "away_goals", "home_xg", "away_xg"):
        assert banned not in cols


def test_season_aggregates_are_off_by_default_but_not_forbidden(frame):
    """`ctx_*` is constant within a (team, season), which looks like leakage and is not:
    FeatureEngineer shifts the season key so a match carries the PRIOR season's figures
    (correlation 0.995 with prior-season totals, 0.589 with the current season's).

    They are excluded by default for a different reason — 210 features on a thousand
    matches measured worse than 42 — so this is a modelling choice, not a safety rule.
    """
    assert not [c for c in safe_feature_columns(frame) if "_ctx_" in c]
    assert [c for c in safe_feature_columns(frame, form_only=False) if "_ctx_" in c]


def test_rolling_form_columns_are_kept(frame):
    cols = safe_feature_columns(frame)
    assert "home_goals_scored_5" in cols and "away_xg_conceded_5" in cols


def test_the_fitted_model_only_holds_safe_columns(frame):
    model = GradientBoostModel(max_iter=20).fit(frame)
    assert model.feature_cols_ == safe_feature_columns(frame)


# --- output shape ------------------------------------------------------------------

@pytest.fixture
def fitted(frame):
    return GradientBoostModel(max_iter=30).fit(frame), frame


def test_probabilities_sum_to_one(fitted):
    model, _ = fitted
    probs = model.predict_1x2("Alpha FC", "Bravo United")
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(v > 0 for v in probs.values())


def test_predict_batch_scores_every_fixture(fitted):
    model, frame = fitted
    out = model.predict_batch(frame.head(20))
    assert len(out) == 20
    np.testing.assert_allclose(
        out[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0, atol=1e-9)


def test_the_scoreline_grid_matches_the_models_own_1x2(fitted):
    """There is no goal model here, so the grid is synthesised — but it must agree with
    the probabilities the model actually reports, or the dashboard shows two answers."""
    model, _ = fitted
    probs = model.predict_1x2("Alpha FC", "Bravo United")
    grid = model.predict_scoreline_probs("Alpha FC", "Bravo United")

    assert grid.sum() == pytest.approx(1.0, abs=1e-9)
    idx = np.arange(grid.shape[0])
    home = grid[idx[:, None] > idx[None, :]].sum()
    draw = float(np.trace(grid))
    away = grid[idx[:, None] < idx[None, :]].sum()
    assert (home, draw, away) == pytest.approx(
        (probs["home"], probs["draw"], probs["away"]), abs=1e-9)


# --- calibration --------------------------------------------------------------------

def test_shrinkage_is_fitted_rather_than_assumed(fitted):
    model, _ = fitted
    assert 0.0 <= model._alpha_ <= 0.9


def test_predictions_stay_inside_believable_bounds(fitted):
    """Raw, this model reached 0.97 on a football 1X2 market. That is not a probability,
    it is an artefact of sharp trees on a thousand matches."""
    model, frame = fitted
    p = model.predict_batch(frame)[["p_home", "p_draw", "p_away"]].to_numpy()
    assert p.max() < 0.95, f"most confident prediction was {p.max():.3f}"


def test_shrinkage_moves_predictions_towards_the_base_rate(fitted):
    model, _ = fitted
    model._alpha_ = 0.0
    raw = model.predict_1x2("Alpha FC", "Bravo United")
    model._alpha_ = 0.8
    shrunk = model.predict_1x2("Alpha FC", "Bravo United")

    spread_raw = max(raw.values()) - min(raw.values())
    spread_shrunk = max(shrunk.values()) - min(shrunk.values())
    assert spread_shrunk < spread_raw


# --- robustness ---------------------------------------------------------------------

def test_a_team_never_seen_before_does_not_crash(fitted):
    """No team identity is used as a feature, so a promoted club is just missing form."""
    model, _ = fitted
    probs = model.predict_1x2("Newly Promoted FC", "Alpha FC")
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)


def test_predicting_before_fitting_is_refused():
    with pytest.raises(RuntimeError):
        GradientBoostModel().predict_1x2("Alpha FC", "Bravo United")


def test_a_frame_with_no_usable_columns_is_refused(matches):
    with pytest.raises(ValueError, match="no usable feature columns"):
        GradientBoostModel().fit(matches[["date", "home_team", "away_team",
                                          "home_goals", "away_goals"]])
