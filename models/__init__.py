"""
models — Premier League match prediction model tiers.

Tier 1: Dixon-Coles (1997) with rho correction + time decay
Tier 2: Bivariate Poisson (Karlis & Ntzoufras, 2003)
Tier 3: xG-driven Dixon-Coles (quasi-likelihood)
Tier 4: Feature-augmented Poisson GLM with L2 regularization
Tier 5: Gradient-boosted classifier — the one model that is not a Poisson variant
Tier 6: Stacked ensemble over all of the above
"""

from .base import BaseMatchModel
from .dixon_coles import DixonColesModel
from .bivariate_poisson import BivariatePoisson
from .xg_dixon_coles import XGDixonColes
from .feature_poisson import FeaturePoisson
from .gradient_boost import GradientBoostModel
from .ensemble import EnsembleModel
from .evaluation import ModelEvaluator
from .predict import MatchPredictor
