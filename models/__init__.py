"""
models — Premier League match prediction model tiers.

Tier 1: Dixon-Coles (1997) with rho correction + time decay
Tier 2: Feature-augmented Poisson GLM with L2 regularization
Tier 3: Gradient-boosted classifier — the one model that is not a Poisson variant
Tier 4: Stacked ensemble over the three above

Bivariate Poisson and an xG-driven Dixon-Coles were also implemented and were removed on
2026-08-10: out of fold they tracked Dixon-Coles closely enough to be the same model twice.
`models/predict.py` records the measurements that decided it.
"""

from .base import BaseMatchModel
from .dixon_coles import DixonColesModel
from .feature_poisson import FeaturePoisson
from .gradient_boost import GradientBoostModel
from .ensemble import EnsembleModel
from .evaluation import ModelEvaluator
from .predict import MatchPredictor
