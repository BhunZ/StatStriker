"""
processors/schema_normalizer.py
--------------------------------
Merges FBRef fixtures, Understat match xG, and Understat deep stats into a
single unified match-level schema.

Merge key: (date, home_team, away_team) — after all DataFrames have had their
team names resolved to canonical form by EntityResolver.

Priority rules:
- Goals (home_goals, away_goals)     : FBRef is ground truth.
- xG (home_xg, away_xg)             : Understat is ground truth; fallback to
                                       FBRef xG when Understat data is missing.
- Standard stats (shots, possession) : FBRef only.
- Deep stats (PPDA, deep, xPTS)      : Understat only.

Output is saved to ``data/processed/merged.parquet``.
"""

import logging
from pathlib import Path

import pandas as pd

from config import PROCESSED_DIR
from processors.entity_resolver import EntityResolver

logger = logging.getLogger(__name__)

_MERGE_KEYS = ["date", "home_team", "away_team"]
_MERGED_PARQUET = PROCESSED_DIR / "merged.parquet"


class SchemaNormalizer:
    """
    Merges FBRef and Understat DataFrames into a unified match-level schema.

    Parameters
    ----------
    resolver : EntityResolver | None
        Entity resolver instance. A default instance is created if not
        provided.
    save_parquet : bool
        If True (default), the merged DataFrame is persisted to
        ``data/processed/merged.parquet``.
    """

    def __init__(
        self,
        resolver: EntityResolver | None = None,
        save_parquet: bool = True,
    ) -> None:
        self._resolver = resolver or EntityResolver()
        self._save_parquet = save_parquet

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply EntityResolver to home_team and away_team columns."""
        return self._resolver.resolve_dataframe(df, ["home_team", "away_team"])

    @staticmethod
    def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the date column is a tz-naive date-only Timestamp."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        return df.dropna(subset=["date"])

    def _pivot_deep_stats(self, deep_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot team-level deep stats into match-level rows.

        The deep stats DataFrame has one row per team per match. We need to
        pivot so each match has home_ppda, away_ppda, home_deep, away_deep, etc.

        Parameters
        ----------
        deep_df : pd.DataFrame
            Deep stats with columns: date, team, is_home, ppda, ppda_allowed,
            deep, deep_allowed, xpts, xg, xga, season_year, etc.

        Returns
        -------
        pd.DataFrame
            Match-level DataFrame with home/away prefixed deep stats columns.
        """
        df = deep_df.copy()

        # Resolve team names
        if "team" in df.columns:
            df["team"] = self._resolver.resolve_series(df["team"])

        df = self._normalize_dates(df)

        # Split into home and away
        home = df[df["is_home"]].copy()
        away = df[~df["is_home"]].copy()

        # Rename columns with home/away prefix
        home_cols = {
            "team": "home_team",
            "ppda": "home_ppda",
            "ppda_allowed": "home_ppda_allowed",
            "deep": "home_deep",
            "deep_allowed": "home_deep_allowed",
            "xpts": "home_xpts",
        }
        away_cols = {
            "team": "away_team",
            "ppda": "away_ppda",
            "ppda_allowed": "away_ppda_allowed",
            "deep": "away_deep",
            "deep_allowed": "away_deep_allowed",
            "xpts": "away_xpts",
        }

        home = home.rename(columns=home_cols)
        away = away.rename(columns=away_cols)

        # Select only the columns we need
        home_keep = ["date", "home_team"] + [
            v for v in home_cols.values() if v != "home_team"
        ]
        away_keep = ["date", "away_team"] + [
            v for v in away_cols.values() if v != "away_team"
        ]

        home = home[[c for c in home_keep if c in home.columns]]
        away = away[[c for c in away_keep if c in away.columns]]

        # Merge home and away on date — match them together
        deep_match = pd.merge(
            home, away,
            on="date",
            how="inner",
        )

        logger.info(
            "Pivoted deep stats: %d match-level rows from %d team-level rows",
            len(deep_match), len(deep_df),
        )
        return deep_match

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def normalize(
        self,
        fbref_df: pd.DataFrame,
        understat_df: pd.DataFrame,
        understat_deep_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Merge FBRef and Understat DataFrames into the canonical unified schema.

        Steps
        -----
        1. Resolve team names to canonical form in all DataFrames.
        2. Normalize date columns.
        3. Outer-merge FBRef + Understat match xG on (date, home, away).
        4. Apply source-priority rules for goals and xG.
        5. If deep stats provided, pivot and merge on (date, home, away).
        6. Build final column set, sort, and persist.

        Parameters
        ----------
        fbref_df : pd.DataFrame
            Cleaned output from ``FBRefScraper.scrape_all()``.
        understat_df : pd.DataFrame
            Cleaned output from ``UnderstatScraper.scrape_all()``.
        understat_deep_df : pd.DataFrame | None
            Optional deep stats from ``UnderstatScraper.scrape_all_deep_stats()``.

        Returns
        -------
        pd.DataFrame
            Unified match-level DataFrame.
        """
        logger.info(
            "SchemaNormalizer: merging %d FBRef rows with %d Understat rows"
            + (" + %d deep stats rows" % len(understat_deep_df) if understat_deep_df is not None else ""),
            len(fbref_df),
            len(understat_df),
        )

        # --- Step 1: Resolve team names ---
        fbref = self._resolve_names(fbref_df)
        understat = self._resolve_names(understat_df)

        # --- Step 2: Normalize dates ---
        fbref = self._normalize_dates(fbref)
        understat = self._normalize_dates(understat)

        # --- Step 3: Merge FBRef + Understat match xG ---
        merged = pd.merge(
            fbref,
            understat,
            on=_MERGE_KEYS,
            how="outer",
            suffixes=("_fbref", "_us"),
        )

        logger.info(
            "Post-merge: %d rows (FBRef-only: %d, Understat-only: %d, matched: %d)",
            len(merged),
            merged["home_goals"].isna().sum() if "home_goals" in merged.columns else 0,
            merged["home_goals_us"].isna().sum() if "home_goals_us" in merged.columns else 0,
            (merged[["home_goals", "home_goals_us"]].notna().all(axis=1).sum()
             if {"home_goals", "home_goals_us"}.issubset(merged.columns) else 0),
        )

        # --- Step 4: Apply priority rules ---

        # Goals: FBRef is authoritative
        if "home_goals" not in merged.columns:
            merged["home_goals"] = pd.NA
        if "away_goals" not in merged.columns:
            merged["away_goals"] = pd.NA

        missing_goals = merged["home_goals"].isna().sum()
        if missing_goals > 0 and "home_goals_us" in merged.columns:
            logger.warning(
                "%d matches have no FBRef goals — filling from Understat",
                missing_goals,
            )
            merged["home_goals"] = merged["home_goals"].fillna(merged["home_goals_us"])
            merged["away_goals"] = merged["away_goals"].fillna(merged["away_goals_us"])

        # xG: Understat is authoritative; fallback to FBRef
        if "home_xg" not in merged.columns:
            merged["home_xg"] = pd.NA
        if "away_xg" not in merged.columns:
            merged["away_xg"] = pd.NA

        # Ensure home_xg/away_xg columns exist from Understat side
        for xg_col, us_col in [("home_xg", "home_xg_us"), ("away_xg", "away_xg_us")]:
            if us_col in merged.columns:
                merged[xg_col] = merged[us_col]
            if merged[xg_col].isna().any() and "home_xg_fbref" in merged.columns:
                fbref_col = xg_col.replace("xg", "xg_fbref")
                if fbref_col in merged.columns:
                    merged[xg_col] = merged[xg_col].fillna(merged[fbref_col])

        # Keep FBRef xG as cross-validation column
        if "home_xg_fbref" not in merged.columns and "home_xg_fbref_fbref" in merged.columns:
            merged["home_xg_fbref"] = merged["home_xg_fbref_fbref"]
        if "away_xg_fbref" not in merged.columns and "away_xg_fbref_fbref" in merged.columns:
            merged["away_xg_fbref"] = merged["away_xg_fbref_fbref"]

        xg_null_pct = merged["home_xg"].isna().mean() * 100
        logger.info("xG null rate after merge+fill: %.1f%%", xg_null_pct)
        if xg_null_pct > 10:
            logger.warning(
                "xG null rate (%.1f%%) exceeds 10%% — check coverage",
                xg_null_pct,
            )

        # --- Step 5: Merge deep stats if provided ---
        if understat_deep_df is not None and not understat_deep_df.empty:
            deep_match = self._pivot_deep_stats(understat_deep_df)
            if not deep_match.empty:
                merged = pd.merge(
                    merged,
                    deep_match,
                    on=_MERGE_KEYS,
                    how="left",
                    suffixes=("", "_deep"),
                )
                deep_cols = [c for c in deep_match.columns if c not in _MERGE_KEYS]
                logger.info(
                    "Deep stats merge: %d/%d matches matched, columns added: %s",
                    merged[deep_cols[0]].notna().sum() if deep_cols else 0,
                    len(merged),
                    deep_cols,
                )

        # --- Step 6: Build final column set ---
        # Resolve season column (may be duplicated or suffixed after merge)
        if "season_fbref" in merged.columns:
            merged["season"] = merged.get("season_fbref", merged.get("season"))
        elif "season" not in merged.columns:
            if "season_year" in merged.columns:
                merged["season"] = merged["season_year"].astype(str)
            elif "season_year_us" in merged.columns:
                merged["season"] = merged["season_year_us"].astype(str)

        output_cols = [
            "date", "season", "home_team", "away_team",
            "home_goals", "away_goals",
            "home_xg", "away_xg",
            "home_xg_fbref", "away_xg_fbref",
            "home_shots", "away_shots",
            "home_poss", "away_poss",
            # Deep stats from Understat
            "home_ppda", "away_ppda",
            "home_ppda_allowed", "away_ppda_allowed",
            "home_deep", "away_deep",
            "home_deep_allowed", "away_deep_allowed",
            "home_xpts", "away_xpts",
        ]

        # Include only columns that actually exist
        final_cols = [c for c in output_cols if c in merged.columns]
        result = merged[final_cols].copy()

        # Type enforcement
        int_like = ["home_goals", "away_goals", "home_shots", "away_shots",
                     "home_deep", "away_deep", "home_deep_allowed", "away_deep_allowed"]
        float_like = ["home_xg", "away_xg", "home_xg_fbref", "away_xg_fbref",
                      "home_poss", "away_poss", "home_ppda", "away_ppda",
                      "home_ppda_allowed", "away_ppda_allowed",
                      "home_xpts", "away_xpts"]

        for col in int_like:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")

        for col in float_like:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")

        result = result.sort_values("date").reset_index(drop=True)

        logger.info(
            "SchemaNormalizer complete: %d rows, %d columns -> %s",
            len(result), len(result.columns), list(result.columns),
        )

        # Persist
        if self._save_parquet:
            result.to_parquet(_MERGED_PARQUET, index=False, engine="pyarrow")
            logger.info("Saved merged dataset -> %s", _MERGED_PARQUET)

        return result
