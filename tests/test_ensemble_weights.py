"""Blend weights, and the two bugs that made them meaningless.

**The weights were not identifiable.** The base models agreed closely, so the log-loss
surface over the weight simplex was nearly flat and the unpenalised optimiser landed on
whatever corner it reached first. Twelve restarts on identical data gave twelve different
answers; logits ran far enough for exp() to underflow, which is how a softmax — which
cannot produce zero — came to report weights of exactly 0.0. One saved ensemble held 0.73
on Dixon-Coles while the metrics recorded for that same run said 0.63 on bivariate Poisson.

**And they agreed closely partly by accident.** `FeaturePoisson` reads its covariates from
`features_home` / `features_away` kwargs, and nothing ever passed them — see
`test_feature_poisson_wiring.py`. With its features switched off it correlated 0.995 with
Dixon-Coles; wired through, 0.70.
"""

import numpy as np
import pytest

from models.ensemble import WEIGHT_L2, _softmax


def _blend_log_loss(w, X, Y):
    b = np.clip(np.einsum("m,nmc->nc", w, X), 1e-10, 1.0)
    b = b / b.sum(axis=1, keepdims=True)
    return -np.mean(np.sum(Y * np.log(b), axis=1))


def _fit(X, Y, lam, seed):
    from scipy.optimize import minimize

    def objective(z):
        return _blend_log_loss(_softmax(z), X, Y) + lam * float(np.sum(np.square(z)))

    start = np.random.default_rng(seed).normal(0, 1, X.shape[1])
    return _softmax(minimize(objective, x0=start, method="Nelder-Mead",
                             options={"maxiter": 3000, "fatol": 1e-10}).x)


@pytest.fixture
def near_collinear():
    """Three models that mostly agree — the situation the real base models are in."""
    rng = np.random.default_rng(7)
    n = 400
    base = rng.dirichlet([4, 3, 3], size=n)
    X = np.stack([
        base,
        np.clip(base + rng.normal(0, 0.01, base.shape), 1e-6, 1),
        np.clip(base + rng.normal(0, 0.015, base.shape), 1e-6, 1),
    ], axis=1)
    X /= X.sum(axis=2, keepdims=True)
    y = np.array([rng.choice(3, p=p) for p in base])
    return X, np.eye(3)[y]


def test_the_penalty_is_small_enough_to_be_a_tiebreak_not_a_constraint():
    assert 0 < WEIGHT_L2 <= 1e-3


def test_without_a_penalty_repeated_fits_disagree(near_collinear):
    """The bug, reproduced: same data, different starting points, different answers."""
    X, Y = near_collinear
    solutions = np.array([_fit(X, Y, 0.0, s) for s in range(8)])
    assert solutions.max(axis=0).max() - solutions.min(axis=0).min() > 0.1


def test_with_the_penalty_repeated_fits_agree(near_collinear):
    X, Y = near_collinear
    solutions = np.array([_fit(X, Y, WEIGHT_L2, s) for s in range(8)])
    spread = (solutions.max(axis=0) - solutions.min(axis=0)).max()
    assert spread < 1e-3, f"weights still vary by {spread:.4f} between restarts"


def test_the_penalty_costs_almost_nothing_in_log_loss(near_collinear):
    X, Y = near_collinear
    free = _blend_log_loss(_fit(X, Y, 0.0, 0), X, Y)
    penalised = _blend_log_loss(_fit(X, Y, WEIGHT_L2, 0), X, Y)
    assert penalised - free < 0.005


def test_weights_are_a_convex_combination(near_collinear):
    X, Y = near_collinear
    w = _fit(X, Y, WEIGHT_L2, 0)
    assert w.min() >= 0
    assert w.sum() == pytest.approx(1.0)


def test_a_softmax_weight_is_never_exactly_zero(near_collinear):
    """Exactly 0.0 out of a softmax means the logits underflowed — a runaway optimiser,
    not a model being ruled out."""
    X, Y = near_collinear
    w = _fit(X, Y, WEIGHT_L2, 0)
    assert (w > 0).all()


def test_a_clearly_better_model_still_gets_most_of_the_weight():
    """The penalty must not flatten a real difference into an average."""
    rng = np.random.default_rng(3)
    n = 600
    y = rng.choice(3, size=n)
    Y = np.eye(3)[y]
    good = np.clip(Y * 0.7 + 0.1 + rng.normal(0, 0.02, Y.shape), 1e-6, 1)
    poor = np.full((n, 3), 1 / 3)
    X = np.stack([good / good.sum(1, keepdims=True), poor], axis=1)

    w = _fit(X, Y, WEIGHT_L2, 0)

    assert w[0] > 0.9, f"the informative model only received {w[0]:.2f}"
