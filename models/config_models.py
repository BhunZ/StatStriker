"""
models/config_models.py
-----------------------
Hyperparameters for all model tiers, separated from the scraping/processing
config in config.py.
"""

# ---------------------------------------------------------------------------
# Time decay
# ---------------------------------------------------------------------------
DC_HALF_LIFE_GRID: list[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # years

# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------
DC_MAX_GOALS: int = 8          # truncate PMF at this many goals per side
DC_INITIAL_RHO: float = -0.05  # starting rho for Dixon-Coles draw correction

# ---------------------------------------------------------------------------
# Cold start (promoted teams with no PL history)
# ---------------------------------------------------------------------------
DC_PROMOTED_ATTACK: float = 0.85   # 15% below league-average scoring
DC_PROMOTED_DEFENSE: float = 1.15  # 15% above league-average conceding

# ---------------------------------------------------------------------------
# Tier 2: Bivariate Poisson (Karlis & Ntzoufras 2003)
# ---------------------------------------------------------------------------
BP_INITIAL_LAMBDA3: float = 0.1  # correlation parameter starting value

# ---------------------------------------------------------------------------
# Tier 3: xG-Dixon-Coles
# ---------------------------------------------------------------------------
XG_BLEND_GRID: list[float] = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]

# ---------------------------------------------------------------------------
# Tier 4: Feature-augmented Poisson GLM
# ---------------------------------------------------------------------------
FP_L2_PENALTY_GRID: list[float] = [0.001, 0.01, 0.1, 1.0, 10.0]
FP_FEATURE_COLS: list[str] = [
    # Rolling-5 form (8)
    "home_xg_scored_5", "away_xg_scored_5",
    "home_xg_conceded_5", "away_xg_conceded_5",
    "home_ppda_5", "away_ppda_5",
    "home_xg_overperformance_5", "away_xg_overperformance_5",
    # EWMA form (4)
    "home_xg_scored_ewma", "away_xg_scored_ewma",
    "home_ppda_ewma", "away_ppda_ewma",
    # Prior-season context (8)
    "home_ctx_stats_poss", "away_ctx_stats_poss",
    "home_ctx_shooting_standard_sot_90", "away_ctx_shooting_standard_sot_90",
    "home_ctx_stats_per_90_minutes_gls", "away_ctx_stats_per_90_minutes_gls",
    "home_ctx_stats_per_90_minutes_g_pk", "away_ctx_stats_per_90_minutes_g_pk",
]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_MIN_TRAIN_MATCHES: int = 200   # minimum training set size
EVAL_STEP_SIZE: int = 38            # expand by ~1 full matchweek
