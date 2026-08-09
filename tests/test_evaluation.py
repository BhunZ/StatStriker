"""Scoring functions and the temporal split.

Every headline number this project reports comes out of here, so the metrics are checked
against values worked out by hand rather than against the code that produces them. The
split tests guard the other half of the claim: that training never sees a match played
after the one being predicted.
"""

import numpy as np
import pandas as pd
import pytest

from models.evaluation import ModelEvaluator


@pytest.fixture
def ev():
    return ModelEvaluator()


# --- outcome encoding -------------------------------------------------------------

@pytest.mark.parametrize("hg, ag, expected", [
    (2, 0, [1, 0, 0]),
    (1, 1, [0, 1, 0]),
    (0, 3, [0, 0, 1]),
])
def test_the_result_is_encoded_the_way_the_metrics_expect(ev, hg, ag, expected):
    assert list(ev._outcome_vector(hg, ag)) == expected


# --- metrics ----------------------------------------------------------------------

def test_log_loss_of_a_certain_correct_prediction_is_zero(ev):
    y = np.array([[1.0, 0, 0]])
    assert ev.log_loss(y, np.array([[1.0, 0, 0]])) == pytest.approx(0, abs=1e-6)


def test_log_loss_of_a_uniform_guess_is_ln3(ev):
    y = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    p = np.full((3, 3), 1 / 3)
    assert ev.log_loss(y, p) == pytest.approx(np.log(3), abs=1e-6)


def test_a_confident_wrong_prediction_scores_worse_than_a_hedged_one(ev):
    y = np.array([[1.0, 0, 0]])
    confident_wrong = np.array([[0.01, 0.01, 0.98]])
    hedged = np.array([[0.30, 0.35, 0.35]])
    assert ev.log_loss(y, confident_wrong) > ev.log_loss(y, hedged)


def test_brier_of_a_perfect_prediction_is_zero(ev):
    y = np.array([[0, 1.0, 0]])
    assert ev.brier_score(y, np.array([[0, 1.0, 0]])) == pytest.approx(0, abs=1e-9)


def test_rps_matches_the_textbook_formula(ev):
    """RPS = mean over matches of sum of squared cumulative errors, divided by (r-1)=2."""
    y = np.array([[1.0, 0, 0]])
    p = np.array([[0.5, 0.3, 0.2]])
    cum_p, cum_y = np.cumsum(p[0]), np.cumsum(y[0])
    expected = float(np.sum((cum_p - cum_y) ** 2) / 2)
    assert ev.ranked_probability_score(y, p) == pytest.approx(expected, abs=1e-9)


def test_rps_punishes_being_wrong_by_two_places_more_than_by_one(ev):
    """This is the whole reason RPS is used instead of log-loss: the outcomes are ordered,
    and calling a home win when it was an away win is worse than calling it a draw."""
    y = np.array([[1.0, 0, 0]])                       # home win
    near = np.array([[0.0, 1.0, 0.0]])                # said draw
    far = np.array([[0.0, 0.0, 1.0]])                 # said away win
    assert ev.ranked_probability_score(y, far) > ev.ranked_probability_score(y, near)


def test_accuracy_counts_the_most_likely_outcome(ev, outcomes):
    y_true, y_pred = outcomes
    # rows 0,1,2 predict home and are home wins; 3,4 predict draw and are draws; 5 predicts
    # away and is an away win -> everything correct
    assert ev.accuracy(y_true, y_pred) == pytest.approx(1.0)


# --- temporal splits --------------------------------------------------------------

@pytest.fixture
def ordered_matches():
    return pd.DataFrame({
        "date": pd.date_range("2024-08-01", periods=400, freq="D"),
        "home_goals": np.arange(400) % 4,
        "away_goals": np.arange(400) % 3,
    })


def test_training_never_contains_a_match_played_after_the_test_set(ev, ordered_matches):
    """The claim the whole evaluation rests on."""
    for train, test in ev.temporal_cv_splits(ordered_matches):
        assert train["date"].max() < test["date"].min()


def test_the_training_window_only_ever_grows(ev, ordered_matches):
    sizes = [len(train) for train, _ in ev.temporal_cv_splits(ordered_matches)]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes), "a fold repeated the same training window"


def test_no_match_appears_in_both_sides_of_a_split(ev, ordered_matches):
    for train, test in ev.temporal_cv_splits(ordered_matches):
        assert set(train.index).isdisjoint(set(test.index))


def test_splits_are_produced_even_when_the_input_arrives_shuffled(ev, ordered_matches):
    """`temporal_cv_splits` sorts internally; a caller passing unsorted data must still get
    chronological folds rather than silently leaking."""
    shuffled = ordered_matches.sample(frac=1, random_state=0)
    for train, test in ev.temporal_cv_splits(shuffled):
        assert train["date"].max() < test["date"].min()


def test_too_little_data_yields_no_folds_rather_than_a_bad_one(ev):
    tiny = pd.DataFrame({"date": pd.date_range("2024-08-01", periods=3),
                         "home_goals": [1, 2, 0], "away_goals": [0, 1, 1]})
    assert ev.temporal_cv_splits(tiny) == []
