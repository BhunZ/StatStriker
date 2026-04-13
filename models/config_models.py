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
# Grid for L2 regularization on the feature weights. The top end (10.0) is
# what keeps ~40 features from overfitting on 1069 training matches.
FP_L2_PENALTY_GRID: list[float] = [0.001, 0.01, 0.1, 1.0, 10.0]

# ---------------------------------------------------------------------------
# Prediction calibration
# ---------------------------------------------------------------------------
# Minimum 1X2 probability floor applied in BaseMatchModel.predict_1x2().
# Prevents overconfident predictions from blowing up log-loss via -log(p)
# when p → 0. Set to 0.5% — enough to prevent degenerate log(0) while
# letting the model express confident predictions. DC only hits this floor
# ~3 times across 869 OOF matches.
PRED_MIN_PROB: float = 0.005

# ---------------------------------------------------------------------------
# FP_FEATURE_COLS  —  v2.0 (X_home / X_away asymmetric architecture)
# ---------------------------------------------------------------------------
#
# Since v2.0, FeaturePoisson learns TWO independent weight vectors
# (``feature_weights_home_`` and ``feature_weights_away_``) rather than
# sharing a single weight vector across both sides of the match. To support
# that cleanly we store **stems** here — the shared portion of the column
# name after the ``home_`` / ``away_`` prefix. ``_prepare_features`` resolves
# each stem to both ``home_{stem}`` and ``away_{stem}`` at fit time, builds
# two standardized matrices pooled on the same scaler, and the GLM learns
# per-side weights.
#
# Why we stopped using the pair list directly:
#   Before v2.0 the model was ``X_home = X_away = X``; both sides shared one
#   weight vector, so the GLM could only learn signals that moved *total*
#   match goals symmetrically. It could not learn that ``ctx_keepers_save``
#   on the *home* side should suppress *away* goals. The refactor breaks
#   that tie and lets each learned weight influence exactly the side it
#   belongs to.
#
# Selection rules applied to produce this list (from scripts/rank_features.py):
#   1. Non-trivial signal (|spearman| > 0.08 on at least one of
#      home_goals / away_goals / total_goals / goal_diff, OR RF gain > 0.002).
#   2. Prefer causal-style metrics (finishing, keeper save%, defensive
#      work rate, discipline) over team-quality metrics that overlap with
#      the model's alpha/beta parameters.
#   3. Avoid redundant duplicates within a stat type (pick sh_90 over sh).
#
# Scaling: handled inside ``_prepare_features`` with a POOLED scaler — mean
# and std are computed from ``np.concatenate([home_col, away_col])`` for
# each stem so the two weight vectors operate on the same scale. NaN values
# become 0 after standardization (= league average).
FP_FEATURE_COLS: list[str] = [
    # Rolling-window form (5-match window, leakage-safe)
    "xg_scored_5",           # recent attacking quality
    "xg_conceded_5",         # recent defensive quality
    "xg_overperformance_5",  # luck / finishing quality
    "xpts_5",                # recent points trajectory (#1 combined rank)
    # Prior-season team context (low-correlation, high-signal stems)
    "ctx_stats_poss",                    # possession style
    "ctx_keepers_performance_save",      # goalkeeper quality (save %)
    "ctx_keepers_performance_ga90",      # goals conceded rate
    "ctx_playingtime_team_success_ppm",  # overall team quality (PPM)
]


# ---------------------------------------------------------------------------
# FP_FEATURE_COLS_V1_PAIRS  —  frozen 44-column pair list from v1.5
# ---------------------------------------------------------------------------
# Preserved for A/B walk-forward backtests (symmetric-v1 vs asymmetric-v2).
# Do NOT use this in production training; it's the pre-refactor snapshot.
# Column order matches the original v1.5 list exactly so the walk-forward
# harness can reproduce v1 results bit-for-bit.
FP_FEATURE_COLS_V1_PAIRS: list[str] = [
    # Rolling-window form
    "home_xg_scored_5", "away_xg_scored_5",
    "home_xg_conceded_5", "away_xg_conceded_5",
    "home_ppda_5", "away_ppda_5",
    "home_xg_overperformance_5", "away_xg_overperformance_5",
    "home_xpts_5", "away_xpts_5",
    "home_deep_5", "away_deep_5",
    # EWMA form
    "home_xg_scored_ewma", "away_xg_scored_ewma",
    "home_ppda_ewma", "away_ppda_ewma",
    # Attacking profile
    "home_ctx_stats_poss", "away_ctx_stats_poss",
    "home_ctx_shooting_standard_sot_90", "away_ctx_shooting_standard_sot_90",
    "home_ctx_stats_per_90_minutes_gls", "away_ctx_stats_per_90_minutes_gls",
    "home_ctx_stats_per_90_minutes_g_pk", "away_ctx_stats_per_90_minutes_g_pk",
    "home_ctx_shooting_standard_sh_90", "away_ctx_shooting_standard_sh_90",
    "home_ctx_shooting_standard_g_sh", "away_ctx_shooting_standard_g_sh",
    # Keeper / goal prevention
    "home_ctx_keepers_performance_save", "away_ctx_keepers_performance_save",
    "home_ctx_keepers_performance_cs_2", "away_ctx_keepers_performance_cs_2",
    "home_ctx_keepers_performance_ga90", "away_ctx_keepers_performance_ga90",
    # Defensive work rate
    "home_ctx_misc_performance_int", "away_ctx_misc_performance_int",
    "home_ctx_misc_performance_tklw", "away_ctx_misc_performance_tklw",
    # Discipline / set pieces
    "home_ctx_misc_performance_fls", "away_ctx_misc_performance_fls",
    "home_ctx_misc_performance_crdy", "away_ctx_misc_performance_crdy",
    # Overall team quality
    "home_ctx_playingtime_team_success_ppm",
    "away_ctx_playingtime_team_success_ppm",
]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_MIN_TRAIN_MATCHES: int = 200   # minimum training set size
EVAL_STEP_SIZE: int = 38            # expand by ~1 full matchweek
