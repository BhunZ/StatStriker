"""
scripts/rank_features.py
------------------------
Rank every numeric column in ``data/processed/features.parquet`` by how
well it correlates with, and predicts, the Poisson targets:

    home_goals,  away_goals,  total_goals,  goal_diff

Two complementary rankings:

1. **Spearman correlation** — rank-based, robust to non-linearity and
   outliers. Signed value shows direction of relationship.
2. **Random Forest feature importance** — gain-based, captures non-linear
   and interaction effects. Fits two small 200-tree regressors (one per
   side) and averages the importance from both.

The script prints the top 20 features per target and writes a full
ranked CSV to ``data/processed/feature_ranking.csv`` so you can open it
in Excel / pandas and slice further.

Usage
-----
    python scripts/rank_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Path bootstrap so we can run as `python scripts/rank_features.py` ----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURES_PARQUET = ROOT / "data" / "processed" / "features.parquet"
OUTPUT_CSV = ROOT / "data" / "processed" / "feature_ranking.csv"

# Columns that are either targets, identifiers, or obviously leaky.
# Anything matching these is excluded from the ranking.
_NON_FEATURE_PREFIXES = (
    "date", "season", "home_team", "away_team",
    "home_goals", "away_goals",
    "home_xg", "away_xg",          # leaky: co-occurs with the match
    "home_shots", "away_shots",
    "home_poss", "away_poss",
    "home_ppda", "away_ppda",       # match-level, co-occurs with target
    "home_ppda_allowed", "away_ppda_allowed",
    "home_deep", "away_deep",
    "home_deep_allowed", "away_deep_allowed",
    "home_xpts", "away_xpts",
)


def _is_feature_col(col: str, df: pd.DataFrame) -> bool:
    """Keep only numeric columns that aren't targets or leaky match-level stats."""
    if col in ("home_goals", "away_goals", "total_goals", "goal_diff"):
        return False
    if col in _NON_FEATURE_PREFIXES:
        return False
    # Allow rolling/EWMA versions of any metric (they use .shift(1), so no leakage)
    # Exclude only the raw match-level columns listed above.
    if not pd.api.types.is_numeric_dtype(df[col]):
        return False
    # Drop columns that are entirely null
    if df[col].isna().all():
        return False
    return True


def _load_features() -> pd.DataFrame:
    if not FEATURES_PARQUET.exists():
        raise SystemExit(f"features.parquet not found at {FEATURES_PARQUET}")
    df = pd.read_parquet(FEATURES_PARQUET)
    df = df.copy()
    df["total_goals"] = df["home_goals"].astype(float) + df["away_goals"].astype(float)
    df["goal_diff"] = df["home_goals"].astype(float) - df["away_goals"].astype(float)
    return df


def _spearman_rank(df: pd.DataFrame, feature_cols: list[str], target: str) -> pd.Series:
    """Signed Spearman correlation of each feature vs target."""
    # Pandas corr with method='spearman' handles NaN pairwise.
    target_series = df[target]
    corrs = {}
    for c in feature_cols:
        s = df[c]
        if s.nunique(dropna=True) < 2:
            corrs[c] = 0.0
            continue
        corrs[c] = s.corr(target_series, method="spearman")
    return pd.Series(corrs, name=f"spearman_{target}").fillna(0.0)


def _rf_importance(
    df: pd.DataFrame, feature_cols: list[str], target: str, n_estimators: int = 200
) -> pd.Series:
    """Gain-based importance from a RandomForestRegressor."""
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as e:
        raise SystemExit(
            "scikit-learn is required. Install with `pip install scikit-learn`."
        ) from e

    X = df[feature_cols].copy()
    # Impute NaN with column median (simple, robust, matches the spirit
    # of the feature_poisson scaler which imputes NaN -> league average).
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    y = df[target].astype(float).values

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=8,       # shallow trees = less variance, faster
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X.values, y)
    return pd.Series(
        rf.feature_importances_, index=feature_cols, name=f"rf_{target}"
    )


def main() -> None:
    df = _load_features()
    print(f"Loaded features.parquet: {df.shape[0]} rows × {df.shape[1]} cols")

    feature_cols = [c for c in df.columns if _is_feature_col(c, df)]
    print(f"Candidate features: {len(feature_cols)}")

    targets = ["home_goals", "away_goals", "total_goals", "goal_diff"]

    # ---- Spearman ----
    print("\n=== Spearman rank correlations ===")
    spearman_cols = []
    for t in targets:
        s = _spearman_rank(df, feature_cols, t)
        spearman_cols.append(s)
    spearman_df = pd.concat(spearman_cols, axis=1)

    # ---- Random Forest ----
    print("\n=== Random Forest feature importances ===")
    rf_cols = []
    for t in targets:
        print(f"  fitting RF for {t} ...")
        s = _rf_importance(df, feature_cols, t)
        rf_cols.append(s)
    rf_df = pd.concat(rf_cols, axis=1)

    # Combine into one ranked frame
    ranking = pd.concat([spearman_df, rf_df], axis=1)
    # Overall score: mean of abs(spearman) across targets + mean RF importance
    ranking["abs_spearman_mean"] = (
        spearman_df[[f"spearman_{t}" for t in targets]].abs().mean(axis=1)
    )
    ranking["rf_importance_mean"] = rf_df.mean(axis=1)
    # Z-score each so they sum meaningfully, then combine
    for col in ("abs_spearman_mean", "rf_importance_mean"):
        v = ranking[col]
        ranking[f"{col}_z"] = (v - v.mean()) / (v.std() + 1e-9)
    ranking["combined_score"] = (
        ranking["abs_spearman_mean_z"] + ranking["rf_importance_mean_z"]
    )
    ranking = ranking.sort_values("combined_score", ascending=False)

    # ---- Print top 20 per target ----
    for t in targets:
        print(f"\n--- Top 20 features by |Spearman| vs {t} ---")
        top = (
            spearman_df[f"spearman_{t}"]
            .abs()
            .sort_values(ascending=False)
            .head(20)
        )
        for name, val in top.items():
            signed = spearman_df.loc[name, f"spearman_{t}"]
            print(f"  {signed:+.3f}  {name}")

    print("\n--- Top 30 features by COMBINED score (|Spearman| + RF gain) ---")
    for i, (name, row) in enumerate(ranking.head(30).iterrows(), start=1):
        print(
            f"  {i:2d}. {name}  "
            f"spear_mean={row['abs_spearman_mean']:.3f}  "
            f"rf_mean={row['rf_importance_mean']:.4f}  "
            f"combined={row['combined_score']:+.2f}"
        )

    # Write full ranking CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUTPUT_CSV)
    print(f"\nFull ranking written to {OUTPUT_CSV}")

    # Also print: for each (home_*, away_*) pair that both score well,
    # show the pair (this is the form we need for FP_FEATURE_COLS).
    print("\n--- Top PAIRED candidates (home_* + away_* both in top 80) ---")
    top_set = set(ranking.head(80).index)
    pairs: list[tuple[str, float]] = []
    for c in top_set:
        if c.startswith("home_"):
            partner = "away_" + c[len("home_"):]
            if partner in top_set:
                combined = (
                    ranking.loc[c, "combined_score"]
                    + ranking.loc[partner, "combined_score"]
                )
                pairs.append((c[len("home_"):], combined))
    pairs.sort(key=lambda x: x[1], reverse=True)
    for i, (stem, score) in enumerate(pairs[:25], start=1):
        print(f"  {i:2d}. home_{stem} / away_{stem}  pair_score={score:+.2f}")


if __name__ == "__main__":
    main()
