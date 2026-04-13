"""
models/evaluation.py
--------------------
Temporal cross-validation engine and proper scoring rules for soccer
match prediction models.

Key design choices
------------------
- **Expanding-window CV**: training set grows monotonically in time.
  Never shuffles, never leaks future data into training.
- **RPS (Ranked Probability Score)**: the gold-standard metric in the
  soccer prediction literature (Ley et al. 2019). Rewards probability
  mass *near* the correct outcome, not just on it.
- **Automated sanity checks**: flags suspiciously good results that
  likely indicate evaluation bugs or data leakage.
"""

import logging

import numpy as np
import pandas as pd

from .config_models import EVAL_MIN_TRAIN_MATCHES, EVAL_STEP_SIZE

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Temporal expanding-window cross-validation for match prediction models.

    Parameters
    ----------
    min_train_matches : int
        Minimum training set size before the first test fold.
    step_size : int
        Number of matches per test fold (~38 = one full round).
    """

    def __init__(
        self,
        min_train_matches: int = EVAL_MIN_TRAIN_MATCHES,
        step_size: int = EVAL_STEP_SIZE,
    ) -> None:
        self._min_train = min_train_matches
        self._step = step_size

    # ------------------------------------------------------------------
    # Temporal splits
    # ------------------------------------------------------------------

    def temporal_cv_splits(
        self, df: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate expanding-window ``(train, test)`` pairs.

        The DataFrame must be sorted by date. Each split expands the
        training window by ``step_size`` matches.
        """
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)
        splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []

        start = self._min_train
        while start + self._step <= n:
            train = df.iloc[:start].copy()
            test = df.iloc[start : start + self._step].copy()
            splits.append((train, test))
            start += self._step

        # Final partial fold (if any remaining matches)
        if start < n:
            train = df.iloc[:start].copy()
            test = df.iloc[start:].copy()
            if len(test) >= 5:  # skip trivially small folds
                splits.append((train, test))

        logger.info(
            "Temporal CV: %d folds (train min=%d, step=%d, total=%d)",
            len(splits), self._min_train, self._step, n,
        )
        return splits

    # ------------------------------------------------------------------
    # Scoring functions
    # ------------------------------------------------------------------

    @staticmethod
    def _outcome_vector(home_goals: int, away_goals: int) -> np.ndarray:
        """One-hot encode the 1X2 outcome: [home_win, draw, away_win]."""
        if home_goals > away_goals:
            return np.array([1.0, 0.0, 0.0])
        elif home_goals == away_goals:
            return np.array([0.0, 1.0, 0.0])
        else:
            return np.array([0.0, 0.0, 1.0])

    @staticmethod
    def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Multi-class log-loss (cross-entropy).

        Parameters
        ----------
        y_true : (n, 3)
            One-hot encoded outcomes.
        y_pred : (n, 3)
            Predicted probabilities [P(H), P(D), P(A)].
        """
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return -float(np.mean(np.sum(y_true * np.log(y_pred), axis=1)))

    @staticmethod
    def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Multi-class Brier score: mean squared error on probability vectors."""
        return float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1)))

    @staticmethod
    def ranked_probability_score(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """
        Ranked Probability Score for ordered outcomes [H, D, A].

        RPS = (1/2) * sum_{r=1}^{2} (CDF_pred(r) - CDF_actual(r))^2

        Lower is better. Standard in soccer prediction literature.
        """
        cdf_pred = np.cumsum(y_pred, axis=1)
        cdf_true = np.cumsum(y_true, axis=1)
        # Only use first 2 CDF values (the 3rd is always 1.0 for both)
        rps_per_match = 0.5 * np.sum(
            (cdf_pred[:, :2] - cdf_true[:, :2]) ** 2, axis=1
        )
        return float(np.mean(rps_per_match))

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Fraction of matches where the highest-probability outcome was correct."""
        pred_class = np.argmax(y_pred, axis=1)
        true_class = np.argmax(y_true, axis=1)
        return float(np.mean(pred_class == true_class))

    @staticmethod
    def calibration_data(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Reliability diagram data: bin predictions by predicted probability,
        compare to observed frequency.

        Returns
        -------
        pd.DataFrame
            Columns: bin_mid, observed_freq, predicted_mean, count.
        """
        bins = np.linspace(0, 1, n_bins + 1)
        records = []
        # Pool all 3 outcome classes together
        for col in range(3):
            preds = y_pred[:, col]
            actuals = y_true[:, col]
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (preds >= lo) & (preds < hi)
                if mask.sum() == 0:
                    continue
                records.append({
                    "bin_mid": (lo + hi) / 2,
                    "observed_freq": float(actuals[mask].mean()),
                    "predicted_mean": float(preds[mask].mean()),
                    "count": int(mask.sum()),
                })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------

    @staticmethod
    def _sanity_check(metrics: dict[str, float]) -> None:
        """Flag suspiciously good results that likely indicate bugs."""
        if metrics.get("log_loss", 999) < 0.90:
            logger.warning(
                "SUSPICIOUSLY LOW log-loss (%.3f) — possible data leakage. "
                "Verify temporal splits and feature computation.",
                metrics["log_loss"],
            )
        if metrics.get("accuracy", 0) > 0.60:
            logger.warning(
                "SUSPICIOUSLY HIGH accuracy (%.1f%%) — possible overfitting or "
                "evaluation bug. Check that no future data enters features.",
                metrics["accuracy"] * 100,
            )

    # ------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model: "BaseMatchModel",
        df: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Run temporal CV on a single model.

        Returns
        -------
        dict
            Mean metrics across all folds: log_loss, brier_score, rps, accuracy.
        """
        from .base import BaseMatchModel  # avoid circular at module level

        splits = self.temporal_cv_splits(df)
        if not splits:
            logger.error("No CV splits generated — dataset too small?")
            return {}

        all_y_true: list[np.ndarray] = []
        all_y_pred: list[np.ndarray] = []
        fold_metrics: list[dict[str, float]] = []

        for fold_idx, (train, test) in enumerate(splits):
            logger.info(
                "  Fold %d/%d: train=%d, test=%d",
                fold_idx + 1, len(splits), len(train), len(test),
            )
            model.fit(train)
            preds = model.predict_batch(test)

            y_true = np.array([
                self._outcome_vector(int(r["home_goals"]), int(r["away_goals"]))
                for _, r in test.iterrows()
            ])
            y_pred = preds[["p_home", "p_draw", "p_away"]].values

            all_y_true.append(y_true)
            all_y_pred.append(y_pred)

            fold_m = {
                "log_loss": self.log_loss(y_true, y_pred),
                "brier_score": self.brier_score(y_true, y_pred),
                "rps": self.ranked_probability_score(y_true, y_pred),
                "accuracy": self.accuracy(y_true, y_pred),
            }
            fold_metrics.append(fold_m)
            logger.debug(
                "    Fold %d metrics: LL=%.4f BS=%.4f RPS=%.4f Acc=%.3f",
                fold_idx + 1, fold_m["log_loss"], fold_m["brier_score"],
                fold_m["rps"], fold_m["accuracy"],
            )

        # Aggregate: compute on ALL out-of-fold predictions pooled
        y_true_all = np.concatenate(all_y_true)
        y_pred_all = np.concatenate(all_y_pred)

        mean_metrics = {
            "log_loss": self.log_loss(y_true_all, y_pred_all),
            "brier_score": self.brier_score(y_true_all, y_pred_all),
            "rps": self.ranked_probability_score(y_true_all, y_pred_all),
            "accuracy": self.accuracy(y_true_all, y_pred_all),
            "n_test_matches": len(y_true_all),
            "n_folds": len(splits),
        }

        self._sanity_check(mean_metrics)
        return mean_metrics

    def compare_models(
        self,
        models: list["BaseMatchModel"],
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Evaluate multiple models and return a side-by-side comparison.

        Returns
        -------
        pd.DataFrame
            One row per model with all metrics.
        """
        rows = []
        for model in models:
            logger.info("Evaluating %s ...", model.name)
            metrics = self.evaluate_model(model, df)
            metrics["model"] = model.name
            rows.append(metrics)

        result = pd.DataFrame(rows)
        if "model" in result.columns:
            result = result.set_index("model")
        result = result.sort_values("rps")
        return result

    def naive_baseline(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Compute metrics for the naive baseline: always predict league-average
        home/draw/away rates.
        """
        df = df.sort_values("date").reset_index(drop=True)
        outcomes = np.array([
            self._outcome_vector(int(r["home_goals"]), int(r["away_goals"]))
            for _, r in df.iterrows()
        ])
        rates = outcomes.mean(axis=0)  # [P(H), P(D), P(A)]
        y_pred = np.tile(rates, (len(df), 1))

        metrics = {
            "log_loss": self.log_loss(outcomes, y_pred),
            "brier_score": self.brier_score(outcomes, y_pred),
            "rps": self.ranked_probability_score(outcomes, y_pred),
            "accuracy": self.accuracy(outcomes, y_pred),
            "rates": f"H={rates[0]:.3f} D={rates[1]:.3f} A={rates[2]:.3f}",
        }
        logger.info(
            "Naive baseline: LL=%.4f BS=%.4f RPS=%.4f Acc=%.3f (%s)",
            metrics["log_loss"], metrics["brier_score"],
            metrics["rps"], metrics["accuracy"], metrics["rates"],
        )
        return metrics
