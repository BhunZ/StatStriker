"""
models/ensemble.py
------------------
Tier 5: Stacked ensemble meta-model.

Combines Tiers 1-4 using optimised per-model weights learned on
out-of-fold predictions from temporal cross-validation. Weights are
constrained to be positive and sum to 1 (via softmax reparametrisation),
giving an interpretable convex combination of base-model probabilities.

Architecture
~~~~~~~~~~~~
    Level 0 (base learners): Tiers 1-4 each output [P(H), P(D), P(A)]
    Level 1 (meta-learner):  4 weights -> weighted average of base probs

Only 4 parameters are learned (one per base model), making the
meta-learner robust even with small OOF sample sizes.
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize as sp_minimize

from .base import BaseMatchModel
from .config_models import DC_MAX_GOALS, EVAL_MIN_TRAIN_MATCHES, EVAL_STEP_SIZE

logger = logging.getLogger(__name__)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / e.sum()


class EnsembleModel(BaseMatchModel):
    """
    Stacked ensemble over multiple base models.

    Parameters
    ----------
    base_models : list[BaseMatchModel]
        Instantiated (unfitted) base models.
    """

    def __init__(
        self,
        base_models: list[BaseMatchModel] | None = None,
        max_goals: int = DC_MAX_GOALS,
    ) -> None:
        super().__init__(name="ensemble", max_goals=max_goals)
        self.base_models_: list[BaseMatchModel] = base_models or []
        self._weights: np.ndarray = np.array([])  # learned model weights
        self._fast_mode: bool = False
        self.oof_log_loss_: float | None = None    # OOF log-loss from weight optimisation
        self.oof_brier_: float | None = None       # OOF Brier score
        self.oof_rps_: float | None = None         # OOF ranked probability score
        self.oof_accuracy_: float | None = None    # OOF accuracy
        self.oof_n_matches_: int = 0               # number of OOF matches used

    def fit(self, df: pd.DataFrame, **kwargs) -> "EnsembleModel":
        """
        Train the stacked ensemble:
        1. Temporal CV to generate out-of-fold base predictions
        2. Optimise per-model weights on OOF predictions (log-loss)
        3. Refit all base models on full data
        """
        from .evaluation import ModelEvaluator

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        n = len(df)
        all_oof_preds = []   # list of (n_test, n_models, 3)
        all_oof_true = []    # list of (n_test, 3)

        def _collect_fold(train_fold, test_fold):
            fold_preds = []
            for model in self.base_models_:
                model.fit(train_fold)
                preds = model.predict_batch(test_fold)
                fold_preds.append(preds[["p_home", "p_draw", "p_away"]].values)
            # Shape: (n_test, n_models, 3)
            stacked = np.stack(fold_preds, axis=1)
            all_oof_preds.append(stacked)
            y_true = np.array([
                ModelEvaluator._outcome_vector(
                    int(r["home_goals"]), int(r["away_goals"])
                )
                for _, r in test_fold.iterrows()
            ])
            all_oof_true.append(y_true)

        if self._fast_mode:
            split = max(int(n * 0.8), n - 20)
            train_fb, test_fb = df.iloc[:split], df.iloc[split:]
            if len(test_fb) >= 2:
                _collect_fold(train_fb, test_fb)
        else:
            step = EVAL_STEP_SIZE
            cursor = EVAL_MIN_TRAIN_MATCHES
            fold_idx = 0
            while cursor + step <= n:
                _collect_fold(df.iloc[:cursor], df.iloc[cursor:cursor + step])
                fold_idx += 1
                logger.debug(
                    "  Ensemble fold %d: train=%d, test=%d",
                    fold_idx, cursor, step,
                )
                cursor += step
            if cursor < n and (n - cursor) >= 5:
                _collect_fold(df.iloc[:cursor], df.iloc[cursor:])

        # Fallback for tiny datasets
        if not all_oof_preds:
            logger.warning(
                "Ensemble: dataset too small (%d matches); "
                "using single 80/20 split.", n,
            )
            split = max(int(n * 0.8), n - 5)
            train_fb, test_fb = df.iloc[:split], df.iloc[split:]
            if len(test_fb) >= 1:
                _collect_fold(train_fb, test_fb)

        # X_oof: (N, n_models, 3),  y_oof: (N, 3)
        X_oof = np.vstack(all_oof_preds)
        y_oof = np.vstack(all_oof_true)

        n_models = len(self.base_models_)
        logger.info(
            "Ensemble: %d OOF predictions, %d base models — optimising weights",
            len(X_oof), n_models,
        )

        # Optimise model weights via log-loss minimisation
        # w_raw are unconstrained; softmax maps them to positive weights summing to 1
        def _neg_log_lik(w_raw):
            w = _softmax(w_raw)
            # Weighted average of base predictions: (N, 3)
            blended = np.tensordot(w, X_oof, axes=([0], [1]))  # (3, N) if wrong
            # Actually: X_oof is (N, n_models, 3), w is (n_models,)
            # einsum is clearest here
            blended = np.einsum("m,nmc->nc", w, X_oof)
            blended = np.clip(blended, 1e-10, 1.0)
            blended = blended / blended.sum(axis=1, keepdims=True)
            # Log-loss: -sum(y_true * log(y_pred))
            return -np.mean(np.sum(y_oof * np.log(blended), axis=1))

        result = sp_minimize(
            _neg_log_lik,
            x0=np.zeros(n_models),      # equal weights initially
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-6},
        )
        self._weights = _softmax(result.x)
        self.oof_log_loss_ = float(result.fun)
        self.oof_n_matches_ = len(X_oof)

        # Compute additional OOF metrics (Brier, RPS, accuracy)
        blended_final = np.einsum("m,nmc->nc", self._weights, X_oof)
        blended_final = np.clip(blended_final, 1e-10, 1.0)
        blended_final = blended_final / blended_final.sum(axis=1, keepdims=True)
        self.oof_brier_ = float(np.mean(np.sum((blended_final - y_oof) ** 2, axis=1)))
        # RPS (ranked probability score)
        cum_pred = np.cumsum(blended_final[:, :2], axis=1)
        cum_true = np.cumsum(y_oof[:, :2], axis=1)
        self.oof_rps_ = float(np.mean(np.sum((cum_pred - cum_true) ** 2, axis=1)) / 2)
        self.oof_accuracy_ = float(np.mean(blended_final.argmax(axis=1) == y_oof.argmax(axis=1)))

        logger.info(
            "Ensemble weights: %s (log-loss=%.4f)",
            {m.name: f"{w:.3f}" for m, w in zip(self.base_models_, self._weights)},
            result.fun,
        )

        # Refit all base models on full data for production predictions
        for model in self.base_models_:
            model.fit(df)

        self._fitted = True
        logger.info("Ensemble fitted: %d base models", len(self.base_models_))
        return self

    def _blend(self, base_probs: np.ndarray) -> np.ndarray:
        """
        Weighted average of base-model probabilities.

        Parameters
        ----------
        base_probs : (n_models, 3) array

        Returns
        -------
        (3,) array of [P(H), P(D), P(A)]
        """
        blended = self._weights @ base_probs       # (3,)
        blended = np.clip(blended, 1e-10, 1.0)
        return blended / blended.sum()

    def predict_scoreline_probs(
        self, home_team: str, away_team: str, **kwargs
    ) -> np.ndarray:
        """
        Produce a joint PMF by:
        1. Getting 1X2 from weighted average
        2. Using the best-weighted base model's joint PMF shape, rescaled
           to match the ensemble's 1X2 margins.
        """
        base_probs = np.array([
            [p["home"], p["draw"], p["away"]]
            for model in self.base_models_
            for p in [model.predict_1x2(home_team, away_team, **kwargs)]
        ])
        meta_1x2 = self._blend(base_probs)

        # Use the highest-weighted base model for the scoreline shape
        best_idx = int(np.argmax(self._weights))
        best_model = self.base_models_[best_idx]
        base_mat = best_model.predict_scoreline_probs(
            home_team, away_team, **kwargs
        )
        base_1x2 = best_model.predict_1x2(home_team, away_team, **kwargs)

        # Rescale each cell by the ratio of ensemble vs base 1X2
        mg = base_mat.shape[0]
        mat = base_mat.copy()
        for i in range(mg):
            for j in range(mg):
                if i > j and base_1x2["home"] > 1e-10:
                    mat[i, j] *= meta_1x2[0] / base_1x2["home"]
                elif i == j and base_1x2["draw"] > 1e-10:
                    mat[i, j] *= meta_1x2[1] / base_1x2["draw"]
                elif i < j and base_1x2["away"] > 1e-10:
                    mat[i, j] *= meta_1x2[2] / base_1x2["away"]

        total = mat.sum()
        if total > 0:
            mat /= total
        return mat

    def predict_1x2(
        self, home_team: str, away_team: str, **kwargs
    ) -> dict[str, float]:
        """Direct 1X2 from weighted average (faster than joint PMF)."""
        base_probs = np.array([
            [p["home"], p["draw"], p["away"]]
            for model in self.base_models_
            for p in [model.predict_1x2(home_team, away_team, **kwargs)]
        ])
        meta = self._blend(base_probs)
        return {"home": float(meta[0]), "draw": float(meta[1]), "away": float(meta[2])}
