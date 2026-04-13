"""
models/feature_poisson.py
-------------------------
Tier 4: Feature-augmented Poisson GLM with team effects.

Extends Tier 1 by adding rolling form features (PPDA, xG, xG
overperformance) and team context (prior-season stats) as log-linear
covariates. Since v2.0 the model uses ASYMMETRIC weights — each side
of the match gets its own independent weight vector:

    log(lambda_home) = mu + alpha_i - beta_j + gamma + x_home @ w_home
    log(lambda_away) = mu + alpha_j - beta_i         + x_away @ w_away

Before v2.0, ``w_home`` and ``w_away`` were tied to the same vector
(``X_home = X_away = X``), which forced every feature weight to act
symmetrically on the total match goals. The refactor lets the GLM
learn that e.g. ``ctx_keepers_save`` on the HOME side should suppress
AWAY goals — a signal the symmetric model simply could not express.

L2 regularization is applied ONLY to the feature weights (both
``w_home`` and ``w_away``), not to team parameters — those are
structural and must be free.

Symmetric mode (production default since 2026-04-11)
----------------------------------------------------
The NLL still learns independent ``w_home`` / ``w_away`` vectors (the
parameter space is unchanged), but when ``symmetric_mode=True`` the two
vectors are averaged element-wise post-fit so the model behaves
symmetrically at inference. This is a temporary measure: a walk-forward
A/B backtest showed the fully asymmetric architecture regressed on
log-loss (+0.33 vs symmetric) because of a pre-existing calibration bug
further downstream (overconfident 1X2 tails amplified by the evaluator's
eps=1e-15 clamp). Keeping the flag means the full v2 architecture can be
re-enabled instantly via ``FeaturePoisson(symmetric_mode=False)`` once
the calibration fix lands. See the Task 3b section of
``plans/piped-greeting-pretzel.md`` for details.

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


#: Architecture version marker. Bump whenever the fitted-state shape changes
#: so ``app/predict.py`` can refuse stale pickles instead of crashing at
#: inference time. v1 = symmetric (w_home = w_away); v2 = asymmetric.
FP_ARCHITECTURE_VERSION: int = 2


class FeaturePoisson(BaseMatchModel):
    """
    Feature-augmented Poisson GLM with team-specific attack/defense
    parameters and asymmetric covariate weights with L2 regularization.

    Parameters
    ----------
    feature_cols : list[str] or None
        Feature STEMS (without the ``home_``/``away_`` prefix). At fit
        time, ``_prepare_features`` resolves each stem to two columns.
        Defaults to ``FP_FEATURE_COLS``.
    l2_penalty : float or None
        L2 penalty strength on feature weights. If None, grid-searched.
    symmetric_mode : bool, default True
        Production default. When True, ``w_home`` and ``w_away`` are
        averaged element-wise after the optimizer converges so the
        fitted model behaves symmetrically (matches the pre-v2.0
        bottleneck). Set False to preserve the full asymmetric weights
        for experimental work. See the module docstring.
    half_life_years : float
        Time decay half-life.
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        l2_penalty: float | None = None,
        symmetric_mode: bool = True,
        half_life_years: float = 1.5,
        max_goals: int = DC_MAX_GOALS,
    ) -> None:
        super().__init__(name="feature_poisson", max_goals=max_goals)
        # Stems only (no ``home_``/``away_`` prefix). Old callers that pass
        # a fully-prefixed pair list are auto-collapsed to stems below so
        # the A/B backtest harness can still supply FP_FEATURE_COLS_V1_PAIRS.
        self._feature_cols = self._coerce_to_stems(feature_cols or FP_FEATURE_COLS)
        self._l2_penalty = l2_penalty
        self._symmetric_mode = symmetric_mode
        self._half_life = half_life_years

        # Fitted state
        self.teams_: list[str] = []
        self.attack_: dict[str, float] = {}
        self.defense_: dict[str, float] = {}
        self.home_adv_: float = 0.0
        self.rho_: float = 0.0
        self.intercept_: float = 0.0
        # Asymmetric weight vectors (v2.0). Each has length == n_stems.
        self.feature_weights_home_: np.ndarray = np.array([])
        self.feature_weights_away_: np.ndarray = np.array([])
        # Back-compat alias: [w_home_0 ... w_home_k, w_away_0 ... w_away_k]
        # so ``feature_importance()`` and legacy tooling still work.
        self.feature_weights_: np.ndarray = np.array([])
        self.feature_cols_used_: list[str] = []       # full 2*k names
        self.feature_cols_used_stems_: list[str] = []  # k stems
        # Pooled scaler (same mean/std applied to home and away matrices).
        self.feature_mean_: np.ndarray = np.array([])
        self.feature_std_: np.ndarray = np.array([])
        self.l2_used_: float = 0.0
        # Architecture version — pickled alongside the model so the loader
        # in app/predict.py can detect and refuse stale v1 pickles.
        self.fp_version_: int = FP_ARCHITECTURE_VERSION
        # Whether w_home and w_away were averaged post-fit. Pickled so
        # the loader can distinguish a symmetric-mode v2 pickle from a
        # fully asymmetric v2 pickle without inspecting weight arrays.
        self.symmetric_mode_: bool = symmetric_mode

    # ------------------------------------------------------------------
    # Stem coercion helper
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_to_stems(cols: list[str]) -> list[str]:
        """
        Accept either a stem list or a fully-prefixed pair list and return
        a deduplicated stem list preserving first-seen order.

        Pair lists are collapsed by stripping the ``home_`` prefix and
        dropping any ``away_...`` entry whose stem was already added via
        its ``home_...`` partner. This lets legacy callers (including the
        A/B backtest harness passing ``FP_FEATURE_COLS_V1_PAIRS``) work
        without modification.
        """
        stems: list[str] = []
        seen: set[str] = set()
        for c in cols:
            if c.startswith("home_"):
                stem = c[len("home_"):]
            elif c.startswith("away_"):
                stem = c[len("away_"):]
            else:
                stem = c
            if stem not in seen:
                seen.add(stem)
                stems.append(stem)
        return stems

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    def _prepare_features(
        self, df: pd.DataFrame, fit_scaler: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract and standardize the asymmetric feature matrices.

        For each stem ``s`` in ``self._feature_cols``, resolves both
        ``home_{s}`` and ``away_{s}`` columns from ``df``. Returns
        ``(X_home, X_away)`` both shaped ``(n_matches, n_stems)``.

        Only stems where BOTH the home and away column exist in ``df``
        are kept — this way the per-stem weight vectors stay aligned.

        When ``fit_scaler=True``, a pooled mean/std is computed per stem
        from the stacked home+away column so the two weight vectors
        operate on the same scale. NaN values become 0 after
        standardization (= league average).
        """
        home_cols: list[str] = []
        away_cols: list[str] = []
        stems_used: list[str] = []
        missing: list[str] = []

        for stem in self._feature_cols:
            h = f"home_{stem}"
            a = f"away_{stem}"
            if h in df.columns and a in df.columns:
                home_cols.append(h)
                away_cols.append(a)
                stems_used.append(stem)
            else:
                missing.append(stem)

        if missing:
            logger.debug(
                "FeaturePoisson: %d stems missing from DataFrame and "
                "skipped: %s", len(missing), missing,
            )

        if not stems_used:
            logger.warning("No feature stems resolved from DataFrame")
            self.feature_cols_used_ = []
            self.feature_cols_used_stems_ = []
            return np.zeros((len(df), 0)), np.zeros((len(df), 0))

        self.feature_cols_used_stems_ = stems_used
        self.feature_cols_used_ = home_cols + away_cols

        X_home = df[home_cols].values.astype(float)
        X_away = df[away_cols].values.astype(float)

        # Pooled scaler: mean and std computed from the stacked home+away
        # values for each stem. A team's rolling xG looks the same whether
        # they're the home or the away side of a match, so pooling doubles
        # the sample size and prevents drift between the two weight vectors.
        if fit_scaler:
            stacked = np.concatenate([X_home, X_away], axis=0)
            all_nan = np.all(np.isnan(stacked), axis=0)
            with np.errstate(all="ignore"):
                self.feature_mean_ = np.where(
                    all_nan, 0.0, np.nanmean(stacked, axis=0)
                )
                std = np.where(all_nan, 1.0, np.nanstd(stacked, axis=0))
            self.feature_std_ = np.where(std < 1e-8, 1.0, std)

        X_home = (X_home - self.feature_mean_) / self.feature_std_
        X_away = (X_away - self.feature_mean_) / self.feature_std_
        X_home = np.nan_to_num(X_home, nan=0.0)
        X_away = np.nan_to_num(X_away, nan=0.0)
        return X_home, X_away

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
        Parameter vector (v2.0 asymmetric):
        [mu, log(alpha_0..n-1), log(beta_0..n-1), gamma, rho,
         w_home_0..w_home_{k-1}, w_away_0..w_away_{k-1}]
        Total: 2*n_teams + 3 + 2*n_features
        """
        mu = params[0]
        log_alpha = params[1 : n_teams + 1]
        log_beta = params[n_teams + 1 : 2 * n_teams + 1]
        gamma = params[2 * n_teams + 1]
        rho = params[2 * n_teams + 2]
        w_start = 2 * n_teams + 3
        w_home = params[w_start : w_start + n_features]
        w_away = params[w_start + n_features : w_start + 2 * n_features]

        alpha = np.exp(log_alpha)

        # Log-linear model for expected goals — each side uses its own
        # learned weight vector.
        log_lam_h = (
            mu + log_alpha[home_idx] - log_beta[away_idx] + gamma
            + X_home @ w_home
        )
        log_lam_a = (
            mu + log_alpha[away_idx] - log_beta[home_idx]
            + X_away @ w_away
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

        # L2 penalty on feature weight vectors
        nll += l2_penalty * (np.sum(w_home ** 2) + np.sum(w_away ** 2))

        # Sum-to-one constraint on alpha only (matches Dixon-Coles Tier 1).
        # Beta is left unconstrained — bounded via optimizer bounds [-2, 2]
        # on log_beta to prevent the fold-15 explosion (674x) without the
        # over-regularization that FP_TEAM_L2_PENALTY + beta constraint caused.
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

        # Prepare asymmetric features (pooled scaler, one matrix per side)
        X_home, X_away = self._prepare_features(df, fit_scaler=True)
        n_features = X_home.shape[1]  # == n_stems, not 2*n_stems

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
            "Fitting Feature Poisson v%d on %d matches, %d stems "
            "(%d total weights), L2=%.4f ...",
            FP_ARCHITECTURE_VERSION, len(df), n_features,
            2 * n_features, l2,
        )

        # Initial parameters — v2.0 has 2*n_features weights (home + away)
        n_params = 2 * n_teams + 3 + 2 * n_features
        x0 = np.zeros(n_params)
        x0[0] = np.log(1.5)  # mu (intercept)
        x0[2 * n_teams + 1] = 0.25  # gamma (home advantage)
        x0[2 * n_teams + 2] = DC_INITIAL_RHO

        bounds = (
            [(None, None)]                         # mu
            + [(-2.0, 2.0)] * n_teams              # log(alpha) — bounded
            + [(-2.0, 2.0)] * n_teams              # log(beta)  — bounded
            + [(None, None)]                       # gamma
            + [(-0.5, 0.5)]                        # rho
            + [(None, None)] * (2 * n_features)    # w_home || w_away
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

        w_start = 2 * n_teams + 3
        self.feature_weights_home_ = p[w_start : w_start + n_features].copy()
        self.feature_weights_away_ = p[w_start + n_features : w_start + 2 * n_features].copy()
        # Back-compat concatenation so feature_importance() and any
        # external inspector keep working — [home weights || away weights].
        self.feature_weights_ = np.concatenate(
            [self.feature_weights_home_, self.feature_weights_away_]
        )

        # Post-fit symmetrization (production default since 2026-04-11).
        # Average w_home and w_away element-wise so the model behaves
        # symmetrically at inference. Matches the reference pattern in
        # scripts/ab_backtest_fp_v1_vs_v2.py::FeaturePoissonV1Symmetric
        # bit-for-bit, so the walk-forward A/B baseline (LL=4.2945) is
        # reproduced exactly. Set symmetric_mode=False to keep the full
        # asymmetric weights for experimentation.
        if self._symmetric_mode:
            shared = 0.5 * (
                self.feature_weights_home_ + self.feature_weights_away_
            )
            self.feature_weights_home_ = shared.copy()
            self.feature_weights_away_ = shared.copy()
            self.feature_weights_ = np.concatenate([shared, shared])
            logger.info(
                "Feature Poisson: symmetric_mode=True — averaged "
                "w_home/w_away post-fit (||w|| = %.3f)",
                float(np.linalg.norm(shared)),
            )
        self.symmetric_mode_ = self._symmetric_mode
        self.fp_version_ = FP_ARCHITECTURE_VERSION
        self._fitted = True

        logger.info(
            "Feature Poisson v%d fitted: %d teams, %d stems "
            "(%d total weights), symmetric=%s, gamma=%.3f, "
            "rho=%.4f, L2=%.4f",
            FP_ARCHITECTURE_VERSION, len(teams), n_features,
            2 * n_features, self._symmetric_mode, self.home_adv_,
            self.rho_, l2,
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

        # v2.0: accept per-side feature vectors. Each side is standardized
        # with the SAME pooled scaler (self.feature_mean_ / self.feature_std_)
        # and then multiplied by its own learned weight vector.
        features_home = kwargs.get("features_home")
        features_away = kwargs.get("features_away")

        # Legacy path: ``features`` kwarg used to apply symmetrically.
        # Warn once and treat it as both sides for mid-refactor compatibility.
        legacy_features = kwargs.get("features")
        if legacy_features is not None and (
            features_home is None and features_away is None
        ):
            logger.warning(
                "FeaturePoisson.predict_scoreline_probs received the "
                "legacy 'features' kwarg — v2.0 expects 'features_home' "
                "and 'features_away'. Applying symmetrically as a "
                "fallback; please update the caller."
            )
            features_home = legacy_features
            features_away = legacy_features

        if (
            features_home is not None
            and len(self.feature_weights_home_) > 0
        ):
            f_h = (
                np.asarray(features_home, dtype=float) - self.feature_mean_
            ) / self.feature_std_
            f_h = np.nan_to_num(f_h, nan=0.0)
            log_lam_h += float(f_h @ self.feature_weights_home_)

        if (
            features_away is not None
            and len(self.feature_weights_away_) > 0
        ):
            f_a = (
                np.asarray(features_away, dtype=float) - self.feature_mean_
            ) / self.feature_std_
            f_a = np.nan_to_num(f_a, nan=0.0)
            log_lam_a += float(f_a @ self.feature_weights_away_)

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
    # Batch prediction (override — pipes per-match features to each side)
    # ------------------------------------------------------------------

    def predict_batch(
        self, fixtures: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """
        Override ``BaseMatchModel.predict_batch`` so Feature Poisson
        predictions actually USE their learned feature weights at
        inference time.

        The base implementation calls ``predict_scoreline_probs(home, away)``
        with no features kwargs, which would silently zero out the
        feature contribution and make v2 predictions identical to a
        pure Dixon-Coles model. Here we standardize the per-side
        feature matrices from the fixtures DataFrame (with ``fit_scaler=False``
        so the pooled scaler from training is reused) and pass row ``i``
        to each call.

        The output schema mirrors ``BaseMatchModel.predict_batch`` exactly:
        every original fixtures column plus ``p_home`` / ``p_draw`` / ``p_away``.
        """
        # If no stems resolved (e.g. caller passes a goals-only DataFrame
        # without feature columns), gracefully fall back to the base
        # implementation which skips feature contribution entirely.
        have_features = bool(self.feature_cols_used_stems_)
        if have_features:
            X_home, X_away = self._prepare_features(fixtures, fit_scaler=False)

        results: list[dict[str, float]] = []
        for i, (_, row) in enumerate(fixtures.iterrows()):
            call_kwargs = dict(kwargs)
            if have_features:
                call_kwargs["features_home"] = X_home[i]
                call_kwargs["features_away"] = X_away[i]
            probs = self.predict_1x2(
                row["home_team"], row["away_team"], **call_kwargs
            )
            results.append(probs)

        out = fixtures.copy()
        probs_df = pd.DataFrame(results)
        out["p_home"] = probs_df["home"].values
        out["p_draw"] = probs_df["draw"].values
        out["p_away"] = probs_df["away"].values
        return out

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        """
        Return feature names and weights sorted by absolute magnitude.

        With v2.0 each stem has TWO weights (one for the home side, one
        for the away side), so the output contains ``2 * n_stems`` rows.
        A ``side`` column disambiguates which weight vector each row
        comes from.
        """
        if not self.feature_cols_used_stems_:
            return pd.DataFrame(columns=["feature", "side", "stem", "weight", "abs_weight"])

        stems = self.feature_cols_used_stems_
        rows = []
        for stem, w in zip(stems, self.feature_weights_home_):
            rows.append(
                {"feature": f"home_{stem}", "side": "home", "stem": stem,
                 "weight": float(w), "abs_weight": float(abs(w))}
            )
        for stem, w in zip(stems, self.feature_weights_away_):
            rows.append(
                {"feature": f"away_{stem}", "side": "away", "stem": stem,
                 "weight": float(w), "abs_weight": float(abs(w))}
            )
        return (
            pd.DataFrame(rows)
            .sort_values("abs_weight", ascending=False)
            .reset_index(drop=True)
        )
