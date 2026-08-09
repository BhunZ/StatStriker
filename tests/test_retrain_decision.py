"""When the pipeline decides to retrain.

This is the whole of the automation's judgement, and until 2026-08-10 two of its three
branches could not fire. The decay branch read `state["recent_predictions"]`, needed
twenty entries with known results, and nothing in the codebase ever appended to that list;
it also needed `last_train_log_loss`, which nothing ever wrote. So the pipeline ran
unattended for four months advertising decay detection that was unreachable code.

It now scores the saved model on matches played since the last training — unseen by
definition — which needs no ledger. These tests exist so it stays reachable.
"""

import math

import pandas as pd
import pytest

import main


def _state(**over):
    base = {
        "last_train_matches": 1000,
        "last_train_date": "2026-01-01",
        "model_version": 2,
        "last_train_log_loss": None,
    }
    base.update(over)
    return base


# --- condition 1: enough new matches ----------------------------------------------

def test_enough_new_matches_triggers_a_retrain():
    assert main._should_train(_state(), 1000 + main.RETRAIN_MATCH_THRESHOLD) is True


def test_one_match_short_of_the_threshold_does_not():
    assert main._should_train(_state(), 1000 + main.RETRAIN_MATCH_THRESHOLD - 1) is False


# --- condition 3: never trained ---------------------------------------------------

def test_a_model_that_has_never_been_trained_is_always_retrained():
    assert main._should_train(_state(model_version=0), 1001) is True


# --- condition 2: decay ------------------------------------------------------------

def test_the_decay_branch_is_skipped_without_a_baseline(caplog):
    """No baseline is a reason to skip, not to retrain — but it must not be the permanent
    state it used to be."""
    assert main._should_train(_state(last_train_log_loss=None), 1001) is False


def test_decay_is_detected_when_the_model_has_got_worse(monkeypatch):
    monkeypatch.setattr(main, "_log_loss_since_last_train",
                        lambda state, df: (1.20, 40))
    # baseline 1.00 -> ratio 1.20, over the 1.05 threshold
    assert main._should_train(_state(last_train_log_loss=1.00), 1001,
                              features_df=pd.DataFrame()) is True


def test_a_model_holding_up_is_left_alone(monkeypatch):
    monkeypatch.setattr(main, "_log_loss_since_last_train",
                        lambda state, df: (1.02, 40))
    assert main._should_train(_state(last_train_log_loss=1.00), 1001,
                              features_df=pd.DataFrame()) is False


def test_an_improving_model_is_left_alone(monkeypatch):
    monkeypatch.setattr(main, "_log_loss_since_last_train",
                        lambda state, df: (0.90, 40))
    assert main._should_train(_state(last_train_log_loss=1.00), 1001,
                              features_df=pd.DataFrame()) is False


def test_too_few_unseen_matches_is_not_read_as_decay(monkeypatch):
    """Three matches in a fortnight say nothing about the model."""
    monkeypatch.setattr(main, "_log_loss_since_last_train", lambda state, df: (None, 3))
    assert main._should_train(_state(last_train_log_loss=1.00), 1001,
                              features_df=pd.DataFrame()) is False


# --- the scorer itself -------------------------------------------------------------

def _fixtures(n, start="2026-02-01"):
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq="D"),
        "home_team": ["Arsenal"] * n,
        "away_team": ["Chelsea"] * n,
        "home_goals": [2] * n,
        "away_goals": [1] * n,
    })


def test_scoring_needs_a_training_date_to_know_what_is_unseen():
    assert main._log_loss_since_last_train({}, _fixtures(50)) == (None, 0)


def test_scoring_refuses_when_barely_any_matches_are_unseen():
    state = {"last_train_date": "2026-01-01"}
    ll, n = main._log_loss_since_last_train(state, _fixtures(5))
    assert ll is None and n == 5


def test_only_matches_after_the_training_date_are_scored(monkeypatch):
    """A match the model was trained on would make this an in-sample reading."""
    seen = {}

    class FakeModel:
        def predict_1x2(self, home, away):
            seen["calls"] = seen.get("calls", 0) + 1
            return {"home": 0.5, "draw": 0.3, "away": 0.2}

    class FakePredictor:
        _models = {"ensemble": FakeModel()}

        @classmethod
        def load_models(cls, directory, df):
            return cls()

    import models
    monkeypatch.setattr(models, "MatchPredictor", FakePredictor)

    df = pd.concat([_fixtures(30, "2025-06-01"), _fixtures(30, "2026-06-01")],
                   ignore_index=True)
    ll, n = main._log_loss_since_last_train({"last_train_date": "2026-01-01"}, df)

    assert n == 30, "matches from before the training date were scored"
    # every fixture is a 2-1 home win and the fake model says home with p=0.5
    assert ll == pytest.approx(-math.log(0.5), abs=1e-6)


def test_a_broken_model_file_skips_the_check_instead_of_killing_the_run(monkeypatch):
    """A decay check is a nice-to-have; it must never be why the weekly run fails."""
    import models

    class Exploding:
        @classmethod
        def load_models(cls, directory, df):
            raise RuntimeError("pickle from an older version")

    monkeypatch.setattr(models, "MatchPredictor", Exploding)
    ll, n = main._log_loss_since_last_train({"last_train_date": "2026-01-01"}, _fixtures(50))
    assert ll is None
