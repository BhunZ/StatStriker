"""
models/xg_dixon_coles.py
------------------------
Tier 3: xG-driven Dixon-Coles.

Fits the Poisson model on a blend of xG and actual goals rather than
raw goals alone. xG has lower variance than actual goals (var/mean
~0.7 vs ~1.1), producing smoother attack/defense parameters that
generalize better to unseen matches.

The blend factor alpha is optimized via grid search:
    y_target = alpha * xG + (1 - alpha) * goals

With non-integer targets, the Poisson log-likelihood uses
``gammaln(y + 1)`` instead of ``log(y!)`` — a standard quasi-likelihood
estimator.
"""

import logging

import numpy as np
import pandas as pd

from .dixon_coles import DixonColesModel
from .config_models import XG_BLEND_GRID, DC_MAX_GOALS

logger = logging.getLogger(__name__)


class XGDixonColes(DixonColesModel):
    """
    xG-blended Dixon-Coles model (subclass of Tier 1).

    Parameters
    ----------
    xg_blend : float or None
        Blend factor in [0, 1]. ``1.0`` = pure xG, ``0.0`` = pure goals.
        If None, grid-searched via ``XG_BLEND_GRID``.
    half_life_years : float or None
        Passed to parent ``DixonColesModel``.
    """

    def __init__(
        self,
        xg_blend: float | None = None,
        half_life_years: float | None = None,
        max_goals: int = DC_MAX_GOALS,
    ) -> None:
        super().__init__(half_life_years=half_life_years, max_goals=max_goals)
        self.name = "xg_dixon_coles"
        self._xg_blend = xg_blend
        self.xg_blend_used_: float = 0.0

    def fit(self, df: pd.DataFrame, **kwargs) -> "XGDixonColes":
        """
        Fit on blended xG/goals targets.

        Requires ``home_xg`` and ``away_xg`` columns in ``df``.
        Falls back to pure goals if xG columns are missing.
        """
        df = df.copy()

        if "home_xg" not in df.columns or "away_xg" not in df.columns:
            logger.warning("xG columns missing — falling back to pure goals")
            self.xg_blend_used_ = 0.0
            return super().fit(df, **kwargs)

        if self._xg_blend is not None:
            blend = self._xg_blend
        else:
            blend = self._optimal_blend(df)

        self.xg_blend_used_ = blend

        # Blend the targets
        blended = df.copy()
        blended["home_goals"] = (
            blend * df["home_xg"].astype(float)
            + (1 - blend) * df["home_goals"].astype(float)
        )
        blended["away_goals"] = (
            blend * df["away_xg"].astype(float)
            + (1 - blend) * df["away_goals"].astype(float)
        )

        logger.info("xG-Dixon-Coles: blend=%.2f (xG weight)", blend)
        result = super().fit(blended, **kwargs)
        self.name = "xg_dixon_coles"  # restore after parent sets it
        return result

    def _optimal_blend(self, df: pd.DataFrame) -> float:
        """Grid search for the xG blend factor that minimizes validation log-loss."""
        from .evaluation import ModelEvaluator

        n = len(df)
        split = int(n * 0.8)
        train = df.iloc[:split]
        val = df.iloc[split:]

        best_blend = 0.5
        best_ll = float("inf")

        for blend in XG_BLEND_GRID:
            temp = XGDixonColes(
                xg_blend=blend,
                half_life_years=self._half_life or 1.5,
                max_goals=self._max_goals,
            )
            temp.fit(train)

            preds = temp.predict_batch(val)
            y_true = np.array([
                ModelEvaluator._outcome_vector(int(r["home_goals"]), int(r["away_goals"]))
                for _, r in val.iterrows()
            ])
            y_pred = preds[["p_home", "p_draw", "p_away"]].values
            ll = ModelEvaluator.log_loss(y_true, y_pred)

            logger.debug("  xG blend %.2f -> log-loss %.4f", blend, ll)
            if ll < best_ll:
                best_ll = ll
                best_blend = blend

        logger.info("Optimal xG blend: %.2f (log-loss=%.4f)", best_blend, best_ll)
        return best_blend
