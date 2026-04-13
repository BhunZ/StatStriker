"""
models/base.py
--------------
Abstract base class for all match prediction models.

Every tier inherits from ``BaseMatchModel`` and must implement:
- ``fit()``  — learn parameters from historical match data
- ``predict_scoreline_probs()`` — produce a joint (home_goals, away_goals) PMF

All derived quantities (1X2 probabilities, expected scorelines, most likely
scoreline) are computed from the joint PMF, making tiers fully interchangeable.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .config_models import DC_MAX_GOALS, DC_PROMOTED_ATTACK, DC_PROMOTED_DEFENSE

logger = logging.getLogger(__name__)


class BaseMatchModel(ABC):
    """
    Abstract base for all Poisson-family match prediction models.

    Parameters
    ----------
    name : str
        Human-readable model name (e.g. ``"dixon_coles"``).
    max_goals : int
        Truncate Poisson PMF at this value (default 8).
    """

    def __init__(self, name: str, max_goals: int = DC_MAX_GOALS) -> None:
        self.name = name
        self._max_goals = max_goals
        self._fitted = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs) -> "BaseMatchModel":
        """Fit model parameters from historical match data."""

    @abstractmethod
    def predict_scoreline_probs(
        self, home_team: str, away_team: str, **kwargs
    ) -> np.ndarray:
        """
        Return the joint PMF matrix of shape ``(max_goals+1, max_goals+1)``.

        Entry ``[i, j]`` is ``P(home_goals=i, away_goals=j)``.
        """

    # ------------------------------------------------------------------
    # Derived predictions (from joint PMF)
    # ------------------------------------------------------------------

    def predict_1x2(
        self, home_team: str, away_team: str, **kwargs
    ) -> dict[str, float]:
        """
        Derive 1X2 probabilities from the joint PMF.

        Returns
        -------
        dict
            ``{"home": P(H), "draw": P(D), "away": P(A)}``
        """
        mat = self.predict_scoreline_probs(home_team, away_team, **kwargs)
        n = mat.shape[0]
        p_home = sum(mat[i, j] for i in range(n) for j in range(n) if i > j)
        p_draw = sum(mat[i, i] for i in range(n))
        p_away = sum(mat[i, j] for i in range(n) for j in range(n) if i < j)
        total = p_home + p_draw + p_away
        if total > 0:
            p_home /= total
            p_draw /= total
            p_away /= total

        # Enforce a minimum probability floor per outcome. No EPL match
        # should ever be assigned < 2% for any 1X2 outcome — even the
        # strongest favourites lose or draw occasionally. Without this
        # floor, overconfident predictions (p < 1e-6) blow up log-loss
        # via -log(p) ≈ 14-34 per match. See Task 3b diagnostics.
        from .config_models import PRED_MIN_PROB
        probs = [p_home, p_draw, p_away]
        probs = [max(p, PRED_MIN_PROB) for p in probs]
        s = sum(probs)
        p_home, p_draw, p_away = probs[0] / s, probs[1] / s, probs[2] / s

        return {"home": float(p_home), "draw": float(p_draw), "away": float(p_away)}

    def predict_expected_goals(
        self, home_team: str, away_team: str, **kwargs
    ) -> tuple[float, float]:
        """Return ``(E[home_goals], E[away_goals])`` from the joint PMF marginals."""
        mat = self.predict_scoreline_probs(home_team, away_team, **kwargs)
        goals = np.arange(mat.shape[0])
        e_home = float(mat.sum(axis=1) @ goals)
        e_away = float(mat.sum(axis=0) @ goals)
        return e_home, e_away

    def predict_batch(self, fixtures: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Predict 1X2 for a DataFrame of fixtures.

        Parameters
        ----------
        fixtures : pd.DataFrame
            Must contain ``home_team`` and ``away_team`` columns.

        Returns
        -------
        pd.DataFrame
            Original columns plus ``p_home``, ``p_draw``, ``p_away``.
        """
        results = []
        for _, row in fixtures.iterrows():
            probs = self.predict_1x2(row["home_team"], row["away_team"], **kwargs)
            results.append(probs)
        out = fixtures.copy()
        probs_df = pd.DataFrame(results)
        out["p_home"] = probs_df["home"].values
        out["p_draw"] = probs_df["draw"].values
        out["p_away"] = probs_df["away"].values
        return out

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _poisson_pmf(lam: float, max_k: int) -> np.ndarray:
        """Poisson PMF vector ``[P(0), P(1), ..., P(max_k)]``."""
        return poisson.pmf(np.arange(max_k + 1), mu=max(lam, 1e-10))

    @staticmethod
    def _time_weights(
        dates: pd.Series,
        reference_date: pd.Timestamp,
        half_life_years: float,
    ) -> np.ndarray:
        """
        Exponential decay weights.

        Parameters
        ----------
        dates : pd.Series
            Match dates.
        reference_date : pd.Timestamp
            The date from which to measure time elapsed (typically the
            most recent match date in the training set).
        half_life_years : float
            Half-life in years. After this period, a match receives
            weight 0.5.

        Returns
        -------
        np.ndarray
            Weights in [0, 1], one per match.
        """
        xi = np.log(2) / (half_life_years * 365.25)
        days_ago = (reference_date - pd.to_datetime(dates)).dt.days.values.astype(float)
        days_ago = np.maximum(days_ago, 0.0)
        return np.exp(-xi * days_ago)

    def _handle_cold_start(self, team: str) -> tuple[float, float]:
        """
        Return ``(attack, defense)`` defaults for teams absent from
        the training data (e.g. newly promoted clubs).
        """
        logger.debug(
            "Cold-start for '%s': using promoted defaults (att=%.2f, def=%.2f)",
            team, DC_PROMOTED_ATTACK, DC_PROMOTED_DEFENSE,
        )
        return DC_PROMOTED_ATTACK, DC_PROMOTED_DEFENSE

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist the fitted model to disk via pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved %s model -> %s", self.name, path)

    @classmethod
    def load(cls, path: Path) -> "BaseMatchModel":
        """Load a previously saved model."""
        with open(path, "rb") as f:
            model = pickle.load(f)  # noqa: S301
        logger.info("Loaded %s model <- %s", model.name, path)
        return model

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"<{self.__class__.__name__}(name={self.name!r}, {status})>"
