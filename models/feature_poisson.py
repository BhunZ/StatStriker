"""
models/feature_poisson.py
-------------------------
Tier 4: Feature-augmented Poisson GLM with team effects.

Extends Tier 1 by adding rolling form features (PPDA, xG, xG
overperformance) and team context (prior-season stats) as log-linear
covariates:

    log(lambda_home) = mu + alpha_i - beta_j + gamma + sum_k(w_k * x_k_home)
    log(lambda_away) = mu + alpha_j - beta_i         + sum_k(w_k * x_k_away)

L2 regularization is applied ONLY to the feature weights w_k, not to
team parameters — those are structural and must be free.

References
----------
- Groll, A. & Schauberger, G. (2019). Variable selection and model
  choice for sports data. *AStA Advances in Statistical Analysis*.
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from .base import BaseMatchModel
from .config_models import (
    DC_INITIAL_RHO,
    DC_MAX_GOALS,
    FP_FEATURE_COLS,
    FP_L2_PENALTY_GRID,
)

logger = logging.getLogger(__name__)

_CONSTRAINT_WEIGHT: float = 50.0


class FeaturePoisson(BaseMatchModel):
    """
    Feature-augmented Poisson GLM with team-specific attack/defense
    parameters and covariate weights with L2 regularization.

    Parameters
    ----------
    feature_cols : list[str] or None
        Feature column names from the DataFrame. Defaults to ``FP_FEATURE_COLS``.
    l2_penalty : float or None
        L2 penalty strength on feature weights. If None, grid-searched.
    half_life_years : float
        Time decay half-life.
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        l2_penalty: float | None = None,
        half_life_years: float = 1.5,
        max_goals: int = DC_MAX_GOALS,
    ) -> None:
        super().__init__(name="feature_poisson", max_goals=max_goals)
        self._feature_cols = feature_cols or FP_FEATURE_COLS
        self._l2_penalty = l2_penalty
        self._half_life = half_life_years

        # Fitted state
        self.teams_: list[str] = []
        self.attack_: dict[str, float] = {}
        self.defense_: dict[str, float] = {}
        self.home_adv_: float = 0.0
        self.rho_: float = 0.0
        self.intercept_: float = 0.0
        self.feature_weights_: np.ndarray = np.array([])
        self.feature_cols_used_: list[str] = []
        self.feature_mean_: np.ndarray = np.array([])
        self.feature_std_: np.ndarray = np.array([])
        self.l2_used_: float = 0.0

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    def _prepare_features(
        self, df: pd.DataFrame, fit_scaler: bool = False
    ) -> np.ndarray:
        """
        Extract and standardize feature matrix.

        Returns (n_matches, n_features) array. NaN imputed with 0 after
        standardization (= league average).
        """
        # Only use columns that exist in df
        available = [c for c in self._feature_cols if c in df.columns]
        if not available:
            logger.warning("No feature columns found in DataFrame")
            return np.zeros((len(df), 0))

        self.feature_cols_used_ = available
        X = df[available].values.astype(float)

        if fit_scaler:
            all_nan = np.all(np.isnan(X), axis=0)
            with np.errstate(all="ignore"):
                self.feature_mean_ = np.where(all_nan, 0.0, np.nanmean(X, axis=0))
                std = np.where(all_nan, 1.0, np.nanstd(X, axis=0))
            self.feature_std_ = np.where(std < 1e-8, 1.0, std)

        X = (X - self.feature_mean_) / self.feature_std_
        X = np.nan_to_num(X, nan=0.0)  # NaN -> league average
        return X

    # ------------------------------------------------------------------
    # Negative log-likelihood
    # ------------------------------------------------------------------

    def _neg_log_likelihood(
        self,
        params: np.ndarray,
        home_idx: np.ndarray,
        away_idx: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        X_home: np.ndarray,
        X_away: np.ndarray,
        weights: np.ndarray,
        n_teams: int,
        n_features: int,
        l2_penalty: float,
    ) -> float:
        """
        Parameter vector:
        [mu, log(alpha_0..n-1), log(beta_0..n-1), gamma, rho, w_0..w_p-1]
        Total: 2*n_teams + 3 + n_features
        """
        mu = params[0]
        log_alpha = params[1 : n_teams + 1]
        log_beta = params[n_teams + 1 : 2 * n_teams + 1]
        gamma = params[2 * n_teams + 1]
        rho = params[2 * n_teams + 2]
        w = params[2 * n_teams + 3:]

        alpha = np.exp(log_alpha)
        beta = np.exp(log_beta)

        # Log-linear model for expected goals
        log_lam_h = (
            mu + log_alpha[home_idx] - log_beta[away_idx] + gamma
            + X_home @ w
        )
        log_lam_a = (
            mu + log_alpha[away_idx] - log_beta[home_idx]
            + X_away @ w
        )

        lam_h = np.exp(np.clip(log_lam_h, -10, 5))
        lam_a = np.exp(np.clip(log_lam_a, -10, 5))

        # Poisson log-likelihood
        log_lik = (
            home_goals * np.log(np.maximum(lam_h, 1e-10)) - lam_h
            + away_goals * np.log(np.maximum(lam_a, 1e-10)) - lam_a
            - gammaln(home_goals + 1) - gammaln(away_goals + 1)
        )

        # Tau correction (Dixon-Coles)
        hg = home_goals.astype(int)
        ag = away_goals.astype(int)
        tau = np.ones_like(lam_h)
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)
        tau[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho
        tau[m01] = 1.0 + lam_h[m01] * rho
        tau[m10] = 1.0 + lam_a[m10] * rho
        tau[m11] = 1.0 - rho
        log_lik += np.log(np.maximum(tau, 1e-10))

        nll = -np.sum(weights * log_lik)

        # L2 penalty on feature weights ONLY
        nll += l2_penalty * np.sum(w ** 2)

        # Sum-to-one constraint
        nll += _CONSTRAINT_WEIGHT * (np.sum(alpha) - n_teams) ** 2

        return nll

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, **kwargs) -> "FeaturePoisson":
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        team_to_idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)

        home_idx = df["home_team"].map(team_to_idx).values.astype(int)
        away_idx = df["away_team"].map(team_to_idx).values.astype(int)
        home_goals = df["home_goals"].values.astype(float)
        away_goals = df["away_goals"].values.astype(float)

        ref_date = pd.to_datetime(df["date"]).max()
        weights = self._time_weights(df["date"], ref_date, self._half_life)

        # Prepare features
        X = self._prepare_features(df, fit_scaler=True)
        n_features = X.shape[1]

        # Split features for home and away observations
        # Home features are used for home lambda, away features for away lambda
        X_home = X
        X_away = X

        # Determine L2 penalty
        if self._l2_penalty is not None:
            l2 = self._l2_penalty
        else:
            l2 = self._optimal_l2(
                df, teams, team_to_idx, home_idx, away_idx,
                home_goals, away_goals, weights, n_teams, n_features,
            )
        self.l2_used_ = l2

        logger.info(
            "Fitting Feature Poisson on %d matches, %d features, L2=%.4f ...",
            len(df), n_features, l2,
        )

        # Initial parameters
        n_params = 2 * n_teams + 3 + n_features
        x0 = np.zeros(n_params)
        x0[0] = np.log(1.5)  # mu (intercept)
        x0[2 * n_teams + 1] = 0.25  # gamma (home advantage)
        x0[2 * n_teams + 2] = DC_INITIAL_RHO

        bounds = (
            [(None, None)]                     # mu
            + [(None, None)] * n_teams         # log(alpha)
            + [(None, None)] * n_teams         # log(beta)
            + [(None, None)]                   # gamma
            + [(-0.5, 0.5)]                    # rho
            + [(None, None)] * n_features      # w
        )

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(home_idx, away_idx, home_goals, away_goals,
                  X_home, X_away, weights, n_teams, n_features, l2),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "maxfun": 50000, "disp": False},
        )

        if not result.success:
            logger.warning("FP optimizer did not converge: %s", result.message)

        # Extract parameters
        p = result.x
        self.intercept_ = float(p[0])
        alpha = np.exp(p[1 : n_teams + 1])
        beta = np.exp(p[n_teams + 1 : 2 * n_teams + 1])

        self.teams_ = teams
        self.attack_ = {t: float(alpha[i]) for i, t in enumerate(teams)}
        self.defense_ = {t: float(beta[i]) for i, t in enumerate(teams)}
        self.home_adv_ = float(p[2 * n_teams + 1])
        self.rho_ = float(p[2 * n_teams + 2])
        self.feature_weights_ = p[2 * n_teams + 3:]
        self._fitted = True

        logger.info(
            "Feature Poisson fitted: %d teams, %d features, gamma=%.3f, "
            "rho=%.4f, L2=%.4f",
            len(teams), n_features, self.home_adv_, self.rho_, l2,
        )
        return self

    def _optimal_l2(
        self, df, teams, team_to_idx, home_idx, away_idx,
        home_goals, away_goals, weights, n_teams, n_features,
    ) -> float:
        """Nested CV to select optimal L2 penalty."""
        from .evaluation import ModelEvaluator

        n = len(df)
        split = int(n * 0.8)
        train = df.iloc[:split]
        val = df.iloc[split:]

        best_l2 = 0.1
        best_ll = float("inf")

        for l2 in FP_L2_PENALTY_GRID:
            temp = FeaturePoisson(
                feature_cols=self._feature_cols,
                l2_penalty=l2,
                half_life_years=self._half_life,
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

            logger.debug("  L2=%.4f -> log-loss %.4f", l2, ll)
            if ll < best_ll:
                best_ll = ll
                best_l2 = l2

        logger.info("Optimal L2 penalty: %.4f (log-loss=%.4f)", best_l2, best_ll)
        return best_l2

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _get_team_params(self, team: str) -> tuple[float, float]:
        if team in self.attack_:
            return self.attack_[team], self.defense_[team]
        return self._handle_cold_start(team)

    def predict_scoreline_probs(
        self, home_team: str, away_team: str, **kwargs
    ) -> np.ndarray:
        alpha_h, beta_h = self._get_team_params(home_team)
        alpha_a, beta_a = self._get_team_params(away_team)

        # Without match-specific features at prediction time, fall back
        # to team strength only (features contribute 0 when unavailable)
        log_lam_h = (
            self.intercept_ + np.log(max(alpha_h, 1e-10))
            - np.log(max(beta_a, 1e-10)) + self.home_adv_
        )
        log_lam_a = (
            self.intercept_ + np.log(max(alpha_a, 1e-10))
            - np.log(max(beta_h, 1e-10))
        )

        # If features are provided via kwargs, apply them
        features = kwargs.get("features")
        if features is not None and len(self.feature_weights_) > 0:
            f = (np.array(features) - self.feature_mean_) / self.feature_std_
            f = np.nan_to_num(f, nan=0.0)
            contribution = f @ self.feature_weights_
            log_lam_h += contribution
            log_lam_a += contribution

        lam_h = np.exp(np.clip(log_lam_h, -10, 5))
        lam_a = np.exp(np.clip(log_lam_a, -10, 5))

        mg = self._max_goals + 1
        pmf_h = self._poisson_pmf(lam_h, self._max_goals)
        pmf_a = self._poisson_pmf(lam_a, self._max_goals)
        mat = np.outer(pmf_h, pmf_a)

        # Tau correction
        rho = self.rho_
        if mat.shape[0] > 1 and mat.shape[1] > 1:
            mat[0, 0] *= max(1.0 - lam_h * lam_a * rho, 1e-10)
            mat[0, 1] *= max(1.0 + lam_h * rho, 1e-10)
            mat[1, 0] *= max(1.0 + lam_a * rho, 1e-10)
            mat[1, 1] *= max(1.0 - rho, 1e-10)

        total = mat.sum()
        if total > 0:
            mat /= total
        return mat

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        """Return feature names and weights sorted by absolute magnitude."""
        return (
            pd.DataFrame({
                "feature": self.feature_cols_used_,
                "weight": self.feature_weights_[:len(self.feature_cols_used_)],
                "abs_weight": np.abs(
                    self.feature_weights_[:len(self.feature_cols_used_)]
                ),
            })
            .sort_values("abs_weight", ascending=False)
            .reset_index(drop=True)
        )
