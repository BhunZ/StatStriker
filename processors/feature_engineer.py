"""
processors/feature_engineer.py
--------------------------------
Computes per-team rolling and EWMA performance features from the unified
match-level DataFrame produced by SchemaNormalizer, plus team-level seasonal
context from FBRef aggregate stats.

Design principles
-----------------
- **No data leakage**: all rolling statistics use only past matches (``.shift(1)``
  before rolling) so match-day features never include the match itself.
- **Vectorized**: uses only Pandas groupby + rolling / ewm — no Python loops
  over rows.
- **Home/away split**: features are computed separately for a team's home
  record and away record before being joined back to the match row.

Output is saved to ``data/processed/features.parquet``.
"""

import logging
from pathlib import Path

import pandas as pd

from config import (
    PROCESSED_DIR,
    ROLLING_WINDOWS,
    EWMA_SPAN,
    FBREF_TEAM_STATS_DIR,
)

logger = logging.getLogger(__name__)

_FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"

# Metrics to compute rolling stats for.
# NOTE: shots and poss are excluded — the FBRef fixtures endpoint (via ScraperAPI)
# no longer carries those columns, so they would always be 100% null at match
# level. Seasonal shot context is available via the FBRef team-stats CSVs
# (ctx_shooting_* columns) and is sufficient for the Dixon-Coles model.
_ROLLING_METRICS = [
    "goals_scored", "goals_conceded",
    "xg_scored", "xg_conceded",
    "ppda", "deep", "xpts",
    "xg_overperformance",
]

# Metrics for EWMA form (shots removed for the same reason as above)
_EWMA_METRICS = [
    "goals_scored", "xg_scored",
    "ppda",
]


class FeatureEngineer:
    """
    Computes rolling, EWMA, and team-context features for match prediction.

    Parameters
    ----------
    rolling_windows : list[int]
        Window sizes (number of past matches) for rolling means.
    ewma_span : int
        Span parameter for exponentially weighted moving average.
    save_parquet : bool
        If True (default), the feature DataFrame is persisted.
    """

    def __init__(
        self,
        rolling_windows: list[int] | None = None,
        ewma_span: int | None = None,
        save_parquet: bool = True,
    ) -> None:
        self._windows = rolling_windows or ROLLING_WINDOWS
        self._ewma_span = ewma_span or EWMA_SPAN
        self._save_parquet = save_parquet

    # ------------------------------------------------------------------
    # Internal: team view construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_team_view(
        df: pd.DataFrame,
        perspective: str,  # "home" or "away"
    ) -> pd.DataFrame:
        """
        Create a long-format view of matches from one team's perspective.

        For ``perspective="home"``, each row represents a home match for that
        team with standardized column names.

        Parameters
        ----------
        df : pd.DataFrame
            Unified merged DataFrame from SchemaNormalizer.
        perspective : str
            "home" or "away".

        Returns
        -------
        pd.DataFrame
            Long-format match view sorted by (team, date).
        """
        opp = "away" if perspective == "home" else "home"

        def _safe_col(name: str) -> pd.Series:
            """Get column or return NaN series."""
            if name in df.columns:
                return df[name]
            return pd.Series(float("nan"), index=df.index, dtype=float)

        view = pd.DataFrame({
            "team": df[f"{perspective}_team"],
            "date": df["date"],
            "match_idx": df.index,
            "goals_scored": df[f"{perspective}_goals"],
            "goals_conceded": df[f"{opp}_goals"],
            "xg_scored": _safe_col(f"{perspective}_xg"),
            "xg_conceded": _safe_col(f"{opp}_xg"),
            # shots and poss are not available at match level from the
            # FBRef fixtures endpoint — omitted to avoid 100% null columns.
            # Seasonal shot context lives in ctx_shooting_* instead.
            "ppda": _safe_col(f"{perspective}_ppda"),
            "deep": _safe_col(f"{perspective}_deep"),
            "xpts": _safe_col(f"{perspective}_xpts"),
        })

        # Derived: xG overperformance (goals - xG)
        view["xg_overperformance"] = (
            pd.to_numeric(view["goals_scored"], errors="coerce")
            - pd.to_numeric(view["xg_scored"], errors="coerce")
        )

        return view.sort_values(["team", "date"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal: rolling stats
    # ------------------------------------------------------------------

    def _rolling_stats(
        self,
        view: pd.DataFrame,
        window: int,
        prefix: str,
    ) -> pd.DataFrame:
        """
        Compute rolling means for a single window size on a team view DataFrame.

        Uses ``.shift(1)`` before ``.rolling()`` so the current match is never
        included in its own feature value (no leakage).

        Parameters
        ----------
        view : pd.DataFrame
            Output of ``_build_team_view``.
        window : int
            Number of past matches for the rolling mean.
        prefix : str
            Column prefix ("home" or "away").

        Returns
        -------
        pd.DataFrame
            DataFrame with match_idx and rolling feature columns.
        """
        result_parts = [view[["match_idx"]].copy()]

        for metric in _ROLLING_METRICS:
            col_name = f"{prefix}_{metric}_{window}"

            if metric not in view.columns or view[metric].isna().all():
                result_parts.append(
                    pd.Series(float("nan"), index=view.index, name=col_name, dtype=float)
                )
                continue

            rolled = (
                view.groupby("team", sort=False)[metric]
                .transform(
                    lambda s: s.shift(1).rolling(window, min_periods=1).mean()
                )
            )
            rolled.name = col_name
            result_parts.append(rolled)

        return pd.concat(result_parts, axis=1)

    # ------------------------------------------------------------------
    # Internal: EWMA form
    # ------------------------------------------------------------------

    def _ewma_stats(self, view: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """
        Compute EWMA form metrics.

        Parameters
        ----------
        view : pd.DataFrame
            Output of ``_build_team_view``.
        prefix : str
            "home" or "away".

        Returns
        -------
        pd.DataFrame
            DataFrame with match_idx and EWMA feature columns.
        """
        span = self._ewma_span

        def _ewm_shift(s: pd.Series) -> pd.Series:
            return s.shift(1).ewm(span=span, min_periods=1, adjust=False).mean()

        result_parts = [view[["match_idx"]].copy()]

        for metric in _EWMA_METRICS:
            col_name = f"{prefix}_{metric}_ewma"

            if metric not in view.columns or view[metric].isna().all():
                result_parts.append(
                    pd.Series(float("nan"), index=view.index, name=col_name, dtype=float)
                )
                continue

            ewma_val = (
                view.groupby("team", sort=False)[metric]
                .transform(_ewm_shift)
            )
            ewma_val.name = col_name
            result_parts.append(ewma_val)

        return pd.concat(result_parts, axis=1)

    # ------------------------------------------------------------------
    # Internal: team context from FBRef aggregate stats
    # ------------------------------------------------------------------

    @staticmethod
    def _load_team_context() -> pd.DataFrame | None:
        """
        Load FBRef team aggregate stats CSVs and build a team context DataFrame.

        Produces per-team-per-season metrics normalized to per-90 values where
        applicable. Each row represents one team's full-season profile.

        Returns
        -------
        pd.DataFrame | None
            Team context DataFrame or None if no stats files found.
        """
        from processors.entity_resolver import EntityResolver

        csv_files = sorted(FBREF_TEAM_STATS_DIR.glob("fbref_team_*.csv"))
        if not csv_files:
            logger.info("No FBRef team stats CSVs found — skipping team context")
            return None

        resolver = EntityResolver()

        # Load and tag each file
        all_stats: dict[str, pd.DataFrame] = {}
        for f in csv_files:
            df = pd.read_csv(f)
            # Extract stat_type from filename: fbref_team_{stat_type}_{season}.csv
            parts = f.stem.split("_")
            # stat_type is everything between "team" and the season slug
            # e.g., fbref_team_shooting_2023-2024 -> shooting
            if len(parts) >= 4:
                stat_type = parts[2]
            else:
                stat_type = "unknown"

            if "squad" in df.columns:
                df["squad"] = resolver.resolve_series(df["squad"])

            key = stat_type
            if key not in all_stats:
                all_stats[key] = []
            all_stats[key].append(df)

        if not all_stats:
            return None

        # Combine each stat type across seasons
        combined: dict[str, pd.DataFrame] = {}
        for stat_type, frames in all_stats.items():
            combined[stat_type] = pd.concat(frames, ignore_index=True)

        # Build context features from each stat type
        context_parts: list[pd.DataFrame] = []

        for stat_type, df in combined.items():
            if "squad" not in df.columns or "season" not in df.columns:
                continue

            # Identify per-90 and percentage columns
            # We'll extract the most useful columns per stat type
            key_cols = ["squad", "season"]
            feature_cols = [c for c in df.columns
                           if c not in ("squad", "season", "stat_type", "matches_played")
                           and df[c].dtype in ("float64", "int64", "Float64", "Int64")]

            if not feature_cols:
                continue

            # Prefix columns with stat_type to avoid collisions
            rename_map = {c: f"ctx_{stat_type}_{c}" for c in feature_cols}
            subset = df[key_cols + feature_cols].copy()
            subset = subset.rename(columns=rename_map)

            # Drop duplicate (squad, season) rows — keep first
            subset = subset.drop_duplicates(subset=["squad", "season"], keep="first")
            context_parts.append(subset)

        if not context_parts:
            return None

        # Merge all stat types on (squad, season)
        context = context_parts[0]
        for part in context_parts[1:]:
            context = pd.merge(context, part, on=["squad", "season"], how="outer")

        # ----------------------------------------------------------------
        # Anti-leakage: expose this season's stats as the NEXT season's
        # prior. A match in season S should only see stats from season S-1,
        # which were fully observable before S started.
        #
        # Season slug format: "YYYY-YYYY"  e.g. "2023-2024"
        # next_season("2023-2024") -> "2024-2025"
        # ----------------------------------------------------------------
        def _next_season(slug: str) -> str:
            try:
                start, end = slug.split("-")
                return f"{int(end)}-{int(end) + 1}"
            except (ValueError, AttributeError):
                return slug

        context["prev_season"] = context["season"].map(_next_season)
        # Drop the raw season key; matches will join on prev_season
        context = context.drop(columns=["season"]).rename(
            columns={"prev_season": "season"}
        )

        logger.info(
            "Team context loaded: %d teams × %d seasons, %d feature columns "
            "(joined as prior-season stats — no future leakage)",
            context["squad"].nunique(),
            context["season"].nunique(),
            len(context.columns) - 2,  # minus squad and season
        )
        return context

    @staticmethod
    def _join_team_context(
        df: pd.DataFrame,
        context: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Join team context features to the match DataFrame.

        Each match gets the previous season's aggregate stats for both its
        home and away teams. The context DataFrame has already been shifted
        by one season in ``_load_team_context`` (season key = the season the
        stats will be *used in*, not the season they were measured in), so
        the join on ``(team, season)`` is leakage-free: a GW-1 match in
        2024-2025 only sees full-season 2023-2024 data, which was completely
        observable before the new season started.

        Teams promoted from the Championship have no prior-season PL data
        and receive NaN context — correct behaviour, handled downstream by
        imputation or model-side treatment of missing values.

        Parameters
        ----------
        df : pd.DataFrame
            Match-level DataFrame.
        context : pd.DataFrame
            Team context DataFrame with (squad, season) keys where season is
            the season the row's stats will be USED (i.e. one season ahead
            of when they were measured).

        Returns
        -------
        pd.DataFrame
            Match DataFrame with home_ctx_* and away_ctx_* columns.
        """
        # Join for home team
        home_ctx = context.copy()
        home_rename = {c: f"home_{c}" for c in home_ctx.columns
                       if c not in ("squad", "season")}
        home_ctx = home_ctx.rename(columns=home_rename)
        home_ctx = home_ctx.rename(columns={"squad": "home_team"})

        df = pd.merge(
            df, home_ctx,
            on=["home_team", "season"],
            how="left",
        )

        # Join for away team
        away_ctx = context.copy()
        away_rename = {c: f"away_{c}" for c in away_ctx.columns
                       if c not in ("squad", "season")}
        away_ctx = away_ctx.rename(columns=away_rename)
        away_ctx = away_ctx.rename(columns={"squad": "away_team"})

        df = pd.merge(
            df, away_ctx,
            on=["away_team", "season"],
            how="left",
        )

        n_ctx_cols = sum(1 for c in df.columns if "_ctx_" in c)
        logger.info("Joined %d team context columns to match data", n_ctx_cols)

        return df

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all rolling, EWMA, and team-context features.

        Parameters
        ----------
        df : pd.DataFrame
            Unified match-level DataFrame from ``SchemaNormalizer.normalize()``.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with feature columns appended. Rows are
            sorted by date.
        """
        logger.info("FeatureEngineer: computing features for %d matches ...", len(df))
        df = df.copy().sort_values("date").reset_index(drop=True)

        # Build home and away perspective views
        home_view = self._build_team_view(df, "home")
        away_view = self._build_team_view(df, "away")

        # Accumulate all feature frames, keyed by match_idx
        feature_frames: list[pd.DataFrame] = []

        # Rolling stats
        for window in self._windows:
            logger.debug("Computing rolling stats: window=%d (home)", window)
            home_roll = self._rolling_stats(home_view, window, "home")
            feature_frames.append(home_roll.set_index("match_idx"))

            logger.debug("Computing rolling stats: window=%d (away)", window)
            away_roll = self._rolling_stats(away_view, window, "away")
            feature_frames.append(away_roll.set_index("match_idx"))

        # EWMA form
        logger.debug("Computing EWMA form (span=%d)", self._ewma_span)
        home_ewma = self._ewma_stats(home_view, "home")
        away_ewma = self._ewma_stats(away_view, "away")
        feature_frames.append(home_ewma.set_index("match_idx"))
        feature_frames.append(away_ewma.set_index("match_idx"))

        # Concatenate all feature frames horizontally, aligned by match_idx
        features = pd.concat(feature_frames, axis=1)

        # Drop any duplicate columns (can happen if same metric appears in
        # both rolling and EWMA)
        features = features.loc[:, ~features.columns.duplicated()]

        # Join back to original match DataFrame
        result = df.join(features, how="left")

        n_rolling = len(result.columns) - len(df.columns)
        logger.info(
            "Rolling/EWMA features: %d new columns added",
            n_rolling,
        )

        # Team context features from FBRef aggregate stats
        context = self._load_team_context()
        if context is not None and "season" in result.columns:
            result = self._join_team_context(result, context)

        n_total = len(result.columns) - len(df.columns)
        logger.info(
            "FeatureEngineer complete: %d rows × %d total columns (%d new features)",
            len(result), len(result.columns), n_total,
        )

        if self._save_parquet:
            result.to_parquet(_FEATURES_PARQUET, index=False, engine="pyarrow")
            logger.info("Saved feature dataset -> %s", _FEATURES_PARQUET)

        return result
