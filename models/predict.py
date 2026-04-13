"""
models/predict.py
-----------------
Unified prediction interface across all model tiers.

Provides a single entry point for fitting, predicting, evaluating,
and serializing models — used by main.py and the future Streamlit/FastAPI
web interface (Phase 3).
"""

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .base import BaseMatchModel
from .dixon_coles import DixonColesModel
from .bivariate_poisson import BivariatePoisson
from .xg_dixon_coles import XGDixonColes
from .feature_poisson import FeaturePoisson
from .ensemble import EnsembleModel
from .evaluation import ModelEvaluator

logger = logging.getLogger(__name__)

# Tier number -> model factory (zero-arg callable returning a fitted-ready model).
# Tier 4 uses symmetric_mode=True as the production default since 2026-04-11
# pending the log-loss calibration fix tracked in Task 3b of
# plans/piped-greeting-pretzel.md. Pass symmetric_mode=False for experiments.
_TIER_FACTORIES: dict[int, Callable[[], BaseMatchModel]] = {
    1: DixonColesModel,
    2: BivariatePoisson,
    3: XGDixonColes,
    4: lambda: FeaturePoisson(symmetric_mode=True),
}


class MatchPredictor:
    """
    Unified interface for match prediction across all tiers.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered match data (from ``features.parquet``).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()
        self._df["date"] = pd.to_datetime(self._df["date"])
        self._models: dict[str, BaseMatchModel] = {}
        self._evaluator = ModelEvaluator()

    @classmethod
    def from_parquet(cls, path: str | Path) -> "MatchPredictor":
        """Load features from Parquet and create a predictor."""
        df = pd.read_parquet(path)
        logger.info("Loaded %d matches from %s", len(df), path)
        return cls(df)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_all(self, tiers: list[int] | None = None) -> None:
        """
        Fit specified tiers (default: all 1-5).

        Tier 5 (ensemble) automatically wraps Tiers 1-4 as base models.
        """
        tiers = tiers or [1, 2, 3, 4, 5]

        for tier in tiers:
            if tier == 5:
                # Ensemble needs Tiers 1-4 fitted first
                self._fit_ensemble()
                continue

            if tier not in _TIER_FACTORIES:
                logger.warning("Unknown tier %d — skipping", tier)
                continue

            model = _TIER_FACTORIES[tier]()
            logger.info("=== Fitting Tier %d: %s ===", tier, model.name)
            model.fit(self._df)
            self._models[model.name] = model

    def _fit_ensemble(self) -> None:
        """Fit Tier 5 ensemble using fresh instances of Tiers 1-4."""
        logger.info("=== Fitting Tier 5: ensemble ===")
        base_models = [
            DixonColesModel(half_life_years=1.5),
            BivariatePoisson(half_life_years=1.5),
            XGDixonColes(half_life_years=1.5),
            FeaturePoisson(half_life_years=1.5, symmetric_mode=True),
        ]
        ensemble = EnsembleModel(base_models=base_models)
        ensemble.fit(self._df)
        self._models["ensemble"] = ensemble

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        home_team: str,
        away_team: str,
        tier: int | str | None = None,
    ) -> dict:
        """
        Predict match outcome.

        Parameters
        ----------
        home_team, away_team : str
            Canonical team names.
        tier : int, str, or None
            Specific tier/model name, or None for all fitted models.

        Returns
        -------
        dict with keys:
            - ``tiers``: {model_name: {"home": ..., "draw": ..., "away": ...}}
            - ``scoreline_matrix``: from the best model
            - ``most_likely_scoreline``: "X-Y"
            - ``expected_goals``: (home_xg, away_xg)
        """
        if tier is not None:
            # Single model prediction
            if isinstance(tier, int):
                name = _TIER_FACTORIES.get(tier, lambda: None)
                if name is None:
                    name = "ensemble" if tier == 5 else str(tier)
                else:
                    name = name().name
            else:
                name = tier

            model = self._models.get(name)
            if model is None:
                raise ValueError(f"Model '{name}' not fitted. Available: {list(self._models)}")

            probs = model.predict_1x2(home_team, away_team)
            mat = model.predict_scoreline_probs(home_team, away_team)
            eg = model.predict_expected_goals(home_team, away_team)
            idx = np.unravel_index(mat.argmax(), mat.shape)

            return {
                "tiers": {name: probs},
                "scoreline_matrix": mat,
                "most_likely_scoreline": f"{idx[0]}-{idx[1]}",
                "expected_goals": {"home": eg[0], "away": eg[1]},
            }

        # All fitted models
        tier_results = {}
        best_model = None
        for name, model in self._models.items():
            tier_results[name] = model.predict_1x2(home_team, away_team)
            if best_model is None:
                best_model = model

        # Use ensemble if available, else first model
        primary = self._models.get("ensemble", best_model)
        mat = primary.predict_scoreline_probs(home_team, away_team)
        eg = primary.predict_expected_goals(home_team, away_team)
        idx = np.unravel_index(mat.argmax(), mat.shape)

        return {
            "tiers": tier_results,
            "scoreline_matrix": mat,
            "most_likely_scoreline": f"{idx[0]}-{idx[1]}",
            "expected_goals": {"home": eg[0], "away": eg[1]},
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> pd.DataFrame:
        """Run temporal CV on all fitted models and return comparison table."""
        models = list(self._models.values())
        if not models:
            logger.error("No models fitted. Call fit_all() first.")
            return pd.DataFrame()

        # Tiers 1-4 do full nested CV (re-tune hyperparameters per fold)
        # for unbiased evaluation. Only the Ensemble (Tier 5) uses fast mode
        # to avoid nested CV (CV-inside-CV), which is prohibitively slow.
        for model in models:
            if hasattr(model, "_fast_mode"):
                model._fast_mode = True

        logger.info("=== Model Evaluation ===")

        # Add naive baseline
        baseline = self._evaluator.naive_baseline(self._df)
        results = [{"model": "naive_baseline", **baseline}]

        comparison = self._evaluator.compare_models(models, self._df)
        comparison = comparison.reset_index()

        # Combine baseline + model results
        combined = pd.concat(
            [pd.DataFrame(results), comparison], ignore_index=True,
        )
        combined = combined.set_index("model")
        return combined

    # ------------------------------------------------------------------
    # Metrics (for pipeline_state.json)
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """
        Return ensemble performance metrics for pipeline_state.json.

        Pulls OOF metrics computed during ensemble weight optimisation,
        plus dataset-level stats.  Returns an empty dict if no ensemble
        is fitted.
        """
        ens = self._models.get("ensemble")
        if ens is None or not getattr(ens, "oof_log_loss_", None):
            return {}

        teams = set(self._df["home_team"].unique()) | set(self._df["away_team"].unique())
        return {
            "ensemble_log_loss": round(ens.oof_log_loss_, 4),
            "ensemble_brier": round(getattr(ens, "oof_brier_", 0), 4),
            "ensemble_rps": round(getattr(ens, "oof_rps_", 0), 4),
            "ensemble_accuracy": round(getattr(ens, "oof_accuracy_", 0), 4),
            "ensemble_oof_matches": ens.oof_n_matches_,
            "ensemble_weights": {
                m.name: round(float(w), 4)
                for m, w in zip(ens.base_models_, ens._weights)
            },
            "total_matches": len(self._df),
            "n_teams": len(teams),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_models(self, directory: Path) -> None:
        """Save all fitted models to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, model in self._models.items():
            model.save(directory / f"{name}.pkl")

    @classmethod
    def load_models(cls, directory: Path, df: pd.DataFrame) -> "MatchPredictor":
        """
        Load all saved models from disk.

        As of v2.0 the FeaturePoisson model has an asymmetric weight
        architecture (``feature_weights_home_`` / ``feature_weights_away_``).
        Pickles trained before the refactor are missing those attributes
        and would crash at inference time. We detect them via
        ``fp_version_`` and skip the stale file — downstream callers
        (the API) will then see a predictor with no ``feature_poisson``
        key and fall back to the remaining tiers / ensemble without
        500-ing.

        Symmetric-mode v2 pickles (w_home == w_away, averaged post-fit)
        are fully accepted; the version check only rejects pickles
        strictly older than v2.
        """
        from .feature_poisson import FP_ARCHITECTURE_VERSION

        predictor = cls(df)
        directory = Path(directory)
        for pkl_file in sorted(directory.glob("*.pkl")):
            model = BaseMatchModel.load(pkl_file)
            # Version guard for FeaturePoisson pickles
            if isinstance(model, FeaturePoisson):
                fp_version = getattr(model, "fp_version_", 1)
                symmetric_mode = getattr(model, "symmetric_mode_", False)
                if fp_version < FP_ARCHITECTURE_VERSION and not symmetric_mode:
                    logger.warning(
                        "Stale FeaturePoisson v%d pickle at %s — v%d is "
                        "required (or symmetric_mode=True on a v%d pickle). "
                        "Skipping; please retrain with "
                        "`python main.py --force-train`.",
                        fp_version, pkl_file, FP_ARCHITECTURE_VERSION,
                        FP_ARCHITECTURE_VERSION,
                    )
                    continue
            predictor._models[model.name] = model
        logger.info("Loaded %d models from %s", len(predictor._models), directory)
        return predictor
