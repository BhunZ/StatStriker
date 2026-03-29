"""
models/dixon_coles.py
---------------------
Tier 1: Enhanced Dixon-Coles model (Dixon & Coles, 1997).

Independent Poisson with:
- **Rho correction (tau)**: inflates P(0-0) draws, deflates P(1-0)/P(0-1).
  Addresses the systematic under-prediction of low-scoring draws by
  independent Poisson.
- **Optimized exponential time decay**: recent matches weighted more
  heavily. The half-life is grid-searched rather than fixed at the
  original paper's xi = 0.0065.
- **Sum-to-one constraint**: attack parameters are anchored via a
  quadratic penalty to ensure identifiability.

MLE is solved via ``scipy.optimize.minimize`` (L-BFGS-B).

Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~
    lambda_home = alpha_i * beta_j * gamma
    lambda_away = alpha_j * beta_i

    P(X=x, Y=y) = tau(x, y, lam_h, lam_a, rho)
                 * Poisson(x | lam_h) * Poisson(y | lam_a)

    tau(0,0) = 1 - lam_h * lam_a * rho
    tau(0,1) = 1 + lam_h * rho
    tau(1,0) = 1 + lam_a * rho
    tau(1,1) = 1 - rho
    tau(else) = 1

References
----------
- Dixon, M. J. & Coles, S. G. (1997). Modelling association football
  scores and inefficiencies in the football betting market. *Journal of
  the Royal Statistical Society: Series C*, 46(2), 265-280.
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson as poisson_dist

from .base import BaseMatchModel
from .config_models import DC_HALF_LIFE_GRID, DC_INITIAL_RHO, DC_MAX_GOALS

logger = logging.getLogger(__name__)

# Constraint weight for sum(alpha) = n_teams
_CONSTRAINT_WEIGHT: float = 50.0


class DixonColesModel(BaseMatchModel):
    """
    Enhanced Dixon-Coles independent Poisson model with tau correction
    and optimized exponential time decay.

    Parameters
    ----------
    half_life_years : float
        Exponential decay half-life in years. If ``None``, automatically
        selected via grid search on ``DC_HALF_LIFE_GRID``.
    max_goals : int
        Truncate PMF at this value (default 8).
    """

    def __init__(
        self,
        half_life_years: float | None = None,
        max_goals: int = DC_MAX_GOALS,
    ) -> None:
        super().__init__(name="dixon_coles", max_goals=max_goals)
        self._half_life = half_life_years

        # Fitted parameters (populated after fit())
        self.teams_: list[str] = []
        self.attack_: dict[str, float] = {}
        self.defense_: dict[str, float] = {}
        self.home_adv_: float = 0.0
        self.rho_: float = 0.0
        self.half_life_used_: float = 0.0

    # ------------------------------------------------------------------
    # Tau correction
    # ------------------------------------------------------------------

    @staticmethod
    def _tau(
        x: np.ndarray,
        y: np.ndarray,
        lam_h: np.ndarray,
        lam_a: np.ndarray,
        rho: float,
    ) -> np.ndarray:
        """
        Vectorized Dixon-Coles tau correction factor.

        Parameters are all 1-D arrays of the same length (one entry per match).
        """
        tau = np.ones_like(lam_h)
        mask_00 = (x == 0) & (y == 0)
        mask_01 = (x == 0) & (y == 1)
        mask_10 = (x == 1) & (y == 0)
        mask_11 = (x == 1) & (y == 1)

        tau[mask_00] = 1.0 - lam_h[mask_00] * lam_a[mask_00] * rho
        tau[mask_01] = 1.0 + lam_h[mask_01] * rho
        tau[mask_10] = 1.0 + lam_a[mask_10] * rho
        tau[mask_11] = 1.0 - rho
        return tau

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
        weights: np.ndarray,
        n_teams: int,
    ) -> float:
        """
        Objective function for scipy.optimize.minimize.

        Parameter vector layout:
            [log(alpha_0..n-1), log(beta_0..n-1), log(gamma), rho]

        Total: 2*n_teams + 2 parameters.
        """
        log_alpha = params[:n_teams]
        log_beta = params[n_teams : 2 * n_teams]
        log_gamma = params[2 * n_teams]
        rho = params[2 * n_teams + 1]

        alpha = np.exp(log_alpha)
        beta = np.exp(log_beta)
        gamma = np.exp(log_gamma)

        # Expected goals per match
        lam_h = alpha[home_idx] * beta[away_idx] * gamma
        lam_a = alpha[away_idx] * beta[home_idx]

        # Clamp to avoid log(0)
        lam_h = np.maximum(lam_h, 1e-10)
        lam_a = np.maximum(lam_a, 1e-10)

        # Poisson log-likelihood (vectorized)
        log_lik = (
            home_goals * np.log(lam_h) - lam_h
            + away_goals * np.log(lam_a) - lam_a
        )
        # gammaln terms for non-integer goals (Tier 3 compatibility)
        from scipy.special import gammaln
        log_lik -= gammaln(home_goals + 1) + gammaln(away_goals + 1)

        # Tau correction
        tau_vals = self._tau(
            home_goals.astype(int), away_goals.astype(int),
            lam_h, lam_a, rho,
        )
        tau_vals = np.maximum(tau_vals, 1e-10)
        log_lik += np.log(tau_vals)

        # Weighted negative log-likelihood
        nll = -np.sum(weights * log_lik)

        # Sum-to-one constraint: sum(alpha) should equal n_teams
        constraint_penalty = _CONSTRAINT_WEIGHT * (np.sum(alpha) - n_teams) ** 2
        nll += constraint_penalty

        return nll

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit_with_half_life(
        self,
        df: pd.DataFrame,
        half_life: float,
    ) -> tuple[dict[str, float], float]:
        """
        Fit the model with a specific half-life. Returns (params_dict, final_nll).
        """
        df = df.sort_values("date").reset_index(drop=True)

        teams = sorted(
            set(df["home_team"].unique()) | set(df["away_team"].unique())
        )
        team_to_idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)

        home_idx = df["home_team"].map(team_to_idx).values.astype(int)
        away_idx = df["away_team"].map(team_to_idx).values.astype(int)
        home_goals = df["home_goals"].values.astype(float)
        away_goals = df["away_goals"].values.astype(float)

        # Time weights
        ref_date = pd.to_datetime(df["date"]).max()
        weights = self._time_weights(df["date"], ref_date, half_life)

        # Initial parameter vector
        n_params = 2 * n_teams + 2
        x0 = np.zeros(n_params)
        # log(alpha) = 0 -> alpha = 1.0
        # log(beta) = 0 -> beta = 1.0
        x0[2 * n_teams] = np.log(1.3)  # home advantage ~1.3
        x0[2 * n_teams + 1] = DC_INITIAL_RHO

        # Bounds: rho in [-0.5, 0.5], everything else unbounded
        bounds = [(None, None)] * (2 * n_teams)  # log(alpha), log(beta)
        bounds.append((None, None))               # log(gamma)
        bounds.append((-0.5, 0.5))                # rho

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(home_idx, away_idx, home_goals, away_goals, weights, n_teams),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "disp": False},
        )

        if not result.success:
            logger.warning(
                "Optimizer did not converge (half_life=%.1f): %s",
                half_life, result.message,
            )

        # Extract fitted parameters
        log_alpha = result.x[:n_teams]
        log_beta = result.x[n_teams : 2 * n_teams]
        log_gamma = result.x[2 * n_teams]
        rho = result.x[2 * n_teams + 1]

        alpha = np.exp(log_alpha)
        beta = np.exp(log_beta)
        gamma = np.exp(log_gamma)

        params = {
            "teams": teams,
            "attack": {t: float(alpha[i]) for i, t in enumerate(teams)},
            "defense": {t: float(beta[i]) for i, t in enumerate(teams)},
            "home_adv": float(gamma),
            "rho": float(rho),
        }
        return params, float(result.fun)

    def fit(self, df: pd.DataFrame, **kwargs) -> "DixonColesModel":
        """
        Fit the Dixon-Coles model via MLE.

        If ``half_life_years`` was not specified at construction, performs
        a grid search over ``DC_HALF_LIFE_GRID`` using the last 20% of
        matches as a validation set.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        if self._half_life is not None:
            half_life = self._half_life
        else:
            half_life = self._optimal_half_life(df)

        logger.info(
            "Fitting Dixon-Coles on %d matches (half_life=%.1f years) ...",
            len(df), half_life,
        )

        params, nll = self._fit_with_half_life(df, half_life)

        self.teams_ = params["teams"]
        self.attack_ = params["attack"]
        self.defense_ = params["defense"]
        self.home_adv_ = params["home_adv"]
        self.rho_ = params["rho"]
        self.half_life_used_ = half_life
        self._fitted = True

        logger.info(
            "Dixon-Coles fitted: %d teams, gamma=%.3f, rho=%.4f, "
            "sum(alpha)=%.2f, NLL=%.1f",
            len(self.teams_), self.home_adv_, self.rho_,
            sum(self.attack_.values()), nll,
        )
        return self

    def _optimal_half_life(self, df: pd.DataFrame) -> float:
        """Grid search for the half-life that minimizes log-loss on a held-out validation set."""
        from .evaluation import ModelEvaluator

        n = len(df)
        split_point = int(n * 0.8)
        train = df.iloc[:split_point]
        val = df.iloc[split_point:]

        best_hl = DC_HALF_LIFE_GRID[len(DC_HALF_LIFE_GRID) // 2]  # default: middle
        best_ll = float("inf")

        for hl in DC_HALF_LIFE_GRID:
            temp_model = DixonColesModel(half_life_years=hl, max_goals=self._max_goals)
            params, _ = temp_model._fit_with_half_life(train, hl)

            # Apply params to temp model for prediction
            temp_model.teams_ = params["teams"]
            temp_model.attack_ = params["attack"]
            temp_model.defense_ = params["defense"]
            temp_model.home_adv_ = params["home_adv"]
            temp_model.rho_ = params["rho"]
            temp_model._fitted = True

            preds = temp_model.predict_batch(val)
            y_true = np.array([
                ModelEvaluator._outcome_vector(int(r["home_goals"]), int(r["away_goals"]))
                for _, r in val.iterrows()
            ])
            y_pred = preds[["p_home", "p_draw", "p_away"]].values
            ll = ModelEvaluator.log_loss(y_true, y_pred)

            logger.debug("  Half-life %.1f yr -> log-loss %.4f", hl, ll)
            if ll < best_ll:
                best_ll = ll
                best_hl = hl

        logger.info(
            "Optimal half-life: %.1f years (log-loss=%.4f)", best_hl, best_ll,
        )
        return best_hl

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _get_team_params(self, team: str) -> tuple[float, float]:
        """Get (attack, defense) for a team, using cold-start if unknown."""
        if team in self.attack_:
            return self.attack_[team], self.defense_[team]
        return self._handle_cold_start(team)

    def predict_scoreline_probs(
        self, home_team: str, away_team: str, **kwargs
    ) -> np.ndarray:
        """
        Build the joint PMF matrix.

        ``P(X=i, Y=j) = tau(i,j,...,rho) * Poisson(i|lam_h) * Poisson(j|lam_a)``
        """
        alpha_h, beta_h = self._get_team_params(home_team)
        alpha_a, beta_a = self._get_team_params(away_team)

        lam_h = alpha_h * beta_a * self.home_adv_
        lam_a = alpha_a * beta_h

        mg = self._max_goals + 1
        pmf_h = self._poisson_pmf(lam_h, self._max_goals)
        pmf_a = self._poisson_pmf(lam_a, self._max_goals)

        # Outer product: independent Poisson joint
        mat = np.outer(pmf_h, pmf_a)

        # Apply tau correction for low-scoring outcomes
        rho = self.rho_
        if mat.shape[0] > 1 and mat.shape[1] > 1:
            mat[0, 0] *= max(1.0 - lam_h * lam_a * rho, 1e-10)
            mat[0, 1] *= max(1.0 + lam_h * rho, 1e-10)
            mat[1, 0] *= max(1.0 + lam_a * rho, 1e-10)
            mat[1, 1] *= max(1.0 - rho, 1e-10)

        # Re-normalize to ensure probabilities sum to 1
        total = mat.sum()
        if total > 0:
            mat /= total

        return mat
