"""
models/gradient_boost.py
------------------------
Tier 5 — gradient-boosted classifier over form features.

Every other model here is the same thing wearing different clothes: a generative Poisson
model that estimates a scoring rate per side and reads 1X2 off the resulting grid. Out of
fold they correlate 0.94-0.997 with each other, which is why blending them bought almost
nothing — an ensemble of near-copies is an expensive way to compute an average.

This one is discriminative. It never models goals at all; it maps form features straight
to P(home), P(draw), P(away) with gradient-boosted trees, so it can use interactions and
non-linearities the Poisson likelihood cannot express. Being wrong in a different way is
the entire point — a member that is individually weaker but decorrelated is worth more to
a blend than a fourth variation on Dixon-Coles.

**Feature safety.** The processed frame carries 220 numeric columns and most of them
cannot be used:

* 10 are the result of the match being predicted (``home_goals``, ``home_xg``, …).
* 168 are ``ctx_*`` season aggregates scraped from FBRef's team tables. These are
  *end-of-season* totals written onto every match of that season — verified constant
  within each (team, season) group, so Arsenal's first match of 2024-25 carries the fact
  that Arsenal finished the season on 86 goals. Using them would be reading the future.

That leaves the 42 rolling and EWMA columns, which are built with ``.shift(1)`` and were
checked match by match. Only those are fed here.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .base import BaseMatchModel
from .config_models import DC_MAX_GOALS

logger = logging.getLogger(__name__)

#: Columns holding the outcome of the match being predicted.
_OUTCOME = re.compile(r"^(home|away)_(goals|xg|shots|poss|ppda|deep|xpts)$")

#: Fallback scoring rates for the synthesised scoreline grid, refined at fit time.
_DEFAULT_LAMBDA = (1.5, 1.2)


def safe_feature_columns(df: pd.DataFrame) -> list[str]:
    """The columns a model may legitimately see before kick-off.

    Excludes the match's own result and every ``ctx_*`` season aggregate — see the module
    docstring for why the latter are future information rather than context.
    """
    numeric = df.select_dtypes("number").columns
    return sorted(c for c in numeric if not _OUTCOME.match(c) and "_ctx_" not in c)


class GradientBoostModel(BaseMatchModel):
    """Predicts 1X2 directly from form features, with no goal model underneath."""

    def __init__(
        self,
        max_goals: int = DC_MAX_GOALS,
        max_iter: int = 200,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 8,
        min_samples_leaf: int = 30,
        l2_regularization: float = 1.0,
        random_state: int = 0,
    ) -> None:
        super().__init__(name="gradient_boost", max_goals=max_goals)
        # Deliberately small trees and a slow rate: roughly a thousand matches against
        # forty features is not much data, and an unconstrained booster memorises it.
        self._params = dict(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )
        self._clf: HistGradientBoostingClassifier | None = None
        self.feature_cols_: list[str] = []
        self._alpha_: float = 0.0                    # shrinkage towards the base rate
        self._base_rate_: np.ndarray = np.array([1 / 3, 1 / 3, 1 / 3])
        self._latest_features_: dict[str, dict[str, float]] = {}
        self._lambda_: tuple[float, float] = _DEFAULT_LAMBDA

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    @staticmethod
    def _labels(df: pd.DataFrame) -> np.ndarray:
        """0 = home win, 1 = draw, 2 = away win — the order the rest of the code uses."""
        home, away = df["home_goals"].to_numpy(), df["away_goals"].to_numpy()
        return np.where(home > away, 0, np.where(home == away, 1, 2))

    def _snapshot_latest_form(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Each team's most recent form, for fixtures that have no row yet."""
        latest: dict[str, dict[str, float]] = {}
        for _, row in df.sort_values("date").iterrows():
            for side in ("home", "away"):
                team = row.get(f"{side}_team")
                if team is None:
                    continue
                stems = {}
                for col in self.feature_cols_:
                    if col.startswith(f"{side}_"):
                        stems[col[len(side) + 1:]] = row.get(col)
                if stems:
                    latest[team] = stems
        return latest

    def _fit_shrinkage(self, df: pd.DataFrame) -> float:
        """How far to pull raw predictions towards the base rate.

        Boosted trees produce sharp probabilities that are not calibrated. Left raw this
        model reached 0.97 on a football 1X2 market where the sharpest bookmakers rarely
        pass 0.90, and it paid for that in log-loss. Shrinking towards the league's base
        rate is the cheapest correction that preserves the ranking:

            p' = (1 - a) * p + a * base_rate

        `a` is chosen on the last fifth of the training window, fitted on the first four
        fifths — a split in time, never a random one, so nothing later informs anything
        earlier.
        """
        cut = int(len(df) * 0.8)
        inner_train, inner_val = df.iloc[:cut], df.iloc[cut:]
        if len(inner_val) < 30:
            return 0.3        # too little to measure; a mild default beats none

        probe = HistGradientBoostingClassifier(**self._params).fit(
            inner_train[self.feature_cols_].to_numpy(dtype=float), self._labels(inner_train))

        raw = probe.predict_proba(inner_val[self.feature_cols_].to_numpy(dtype=float))
        if raw.shape[1] != 3:
            full = np.zeros((len(raw), 3))
            for i, cls in enumerate(probe.classes_):
                full[:, int(cls)] = raw[:, i]
            raw = full

        y_val = np.eye(3)[self._labels(inner_val)]
        base = np.eye(3)[self._labels(inner_train)].mean(axis=0)

        best_alpha, best_ll = 0.0, np.inf
        for alpha in np.linspace(0.0, 0.9, 19):
            blended = np.clip((1 - alpha) * raw + alpha * base, 1e-10, 1.0)
            ll = -np.mean(np.sum(y_val * np.log(blended), axis=1))
            if ll < best_ll:
                best_alpha, best_ll = float(alpha), ll

        logger.info("Gradient boost shrinkage: a=%.2f (val log-loss %.4f)",
                    best_alpha, best_ll)
        return best_alpha

    def fit(self, df: pd.DataFrame, **kwargs) -> "GradientBoostModel":
        self.feature_cols_ = safe_feature_columns(df)
        if not self.feature_cols_:
            raise ValueError("no usable feature columns — is this the processed frame?")

        df = df.sort_values("date") if "date" in df.columns else df
        X = df[self.feature_cols_].to_numpy(dtype=float)
        y = self._labels(df)

        self._base_rate_ = np.eye(3)[y].mean(axis=0)
        self._alpha_ = self._fit_shrinkage(df)
        self._clf = HistGradientBoostingClassifier(**self._params).fit(X, y)
        self._latest_features_ = self._snapshot_latest_form(df)
        self._lambda_ = (
            float(df["home_goals"].mean()) if "home_goals" in df else _DEFAULT_LAMBDA[0],
            float(df["away_goals"].mean()) if "away_goals" in df else _DEFAULT_LAMBDA[1],
        )
        self._fitted = True

        logger.info(
            "Gradient boost fitted: %d matches, %d features, %d trees/class",
            len(df), len(self.feature_cols_), self._params["max_iter"],
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _row_vector(self, row) -> np.ndarray:
        return np.array([row.get(c, np.nan) for c in self.feature_cols_], dtype=float)

    def _team_vector(self, home_team: str, away_team: str) -> np.ndarray:
        """Build a feature row for a fixture with no row of its own, from latest form."""
        values = []
        for col in self.feature_cols_:
            side, stem = ("home", col[5:]) if col.startswith("home_") else ("away", col[5:])
            team = home_team if side == "home" else away_team
            values.append(self._latest_features_.get(team, {}).get(stem, np.nan))
        return np.array(values, dtype=float)

    def _probs(self, X: np.ndarray) -> np.ndarray:
        proba = self._clf.predict_proba(X)
        # A fold may not contain every outcome; align to the fixed 0/1/2 ordering.
        if proba.shape[1] != 3:
            full = np.zeros((len(proba), 3))
            for i, cls in enumerate(self._clf.classes_):
                full[:, int(cls)] = proba[:, i]
            proba = full
        # Pull towards the base rate — see _fit_shrinkage for why raw boosted
        # probabilities cannot be used directly.
        proba = (1 - self._alpha_) * proba + self._alpha_ * self._base_rate_
        proba = np.clip(proba, 1e-6, 1.0)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict_1x2(self, home_team: str, away_team: str, **kwargs) -> dict[str, float]:
        self._check_fitted()
        row = kwargs.get("feature_row")
        X = (self._row_vector(row) if row is not None
             else self._team_vector(home_team, away_team)).reshape(1, -1)
        p = self._probs(X)[0]
        return {"home": float(p[0]), "draw": float(p[1]), "away": float(p[2])}

    def predict_batch(self, fixtures: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Score a set of fixtures, each from its own row rather than from latest form."""
        self._check_fitted()
        X = fixtures.reindex(columns=self.feature_cols_).to_numpy(dtype=float)
        p = self._probs(X)
        out = fixtures.copy()
        out["p_home"], out["p_draw"], out["p_away"] = p[:, 0], p[:, 1], p[:, 2]
        return out

    def predict_scoreline_probs(
        self, home_team: str, away_team: str, **kwargs
    ) -> np.ndarray:
        """A scoreline grid consistent with this model's 1X2.

        There is no goal model here, so the grid is synthesised: start from a Poisson grid
        at the league's average scoring rates, then rescale its home / draw / away regions
        until each carries the mass this model assigns. The marginals are league-shaped
        while the three outcome probabilities are exactly the classifier's — honest about
        what the model does and does not claim to know.
        """
        self._check_fitted()
        n = self._max_goals + 1
        grid = np.outer(self._poisson_pmf(self._lambda_[0], self._max_goals),
                        self._poisson_pmf(self._lambda_[1], self._max_goals))

        idx = np.arange(n)
        region = np.where(idx[:, None] > idx[None, :], 0,
                          np.where(idx[:, None] == idx[None, :], 1, 2))

        target = self.predict_1x2(home_team, away_team, **kwargs)
        wanted = [target["home"], target["draw"], target["away"]]
        for r in range(3):
            mass = grid[region == r].sum()
            grid[region == r] *= (wanted[r] / mass) if mass > 0 else 0.0
        return grid / grid.sum()

    def _check_fitted(self) -> None:
        if self._clf is None:
            raise RuntimeError("GradientBoostModel.fit must be called first")
