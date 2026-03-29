"""
processors/entity_resolver.py
------------------------------
Resolves team name discrepancies between FBRef and Understat by mapping all
raw names to a single canonical Premier League team list.

Strategy (two-pass):
1. Exact lookup in a hand-curated ``ALIAS_MAP`` dict  - O(1), handles the most
   common known divergences (e.g. "Man Utd" -> "Manchester United").
2. Fuzzy fallback using ``rapidfuzz.process.extractOne`` against the
   ``CANONICAL_TEAMS`` list  - catches novel or unexpected spellings.

All resolutions are logged at INFO level so the pipeline audit trail shows
exactly which name was mapped to which canonical form.
"""

import logging
from typing import Optional

import pandas as pd
from rapidfuzz import process as fuzz_process, fuzz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ground-truth canonical team names for the Premier League 2023-2026 window.
# Every resolved name must appear in this list.
# ---------------------------------------------------------------------------
CANONICAL_TEAMS: list[str] = [
    "Arsenal",
    "Aston Villa",
    "Brentford",
    "Brighton",
    "Burnley",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Ipswich Town",
    "Leicester City",
    "Liverpool",
    "Luton Town",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Sheffield United",
    "Southampton",
    "Tottenham",
    "West Ham",
    "Wolverhampton",
    "Bournemouth",
    "Leeds United",
    "Sunderland",
]

# ---------------------------------------------------------------------------
# Hand-curated alias map  - covers all *known* FBRef / Understat divergences.
# Keys are lowercased for case-insensitive lookup.
# ---------------------------------------------------------------------------
ALIAS_MAP: dict[str, str] = {
    # FBRef aliases
    "man utd": "Manchester United",
    "man united": "Manchester United",
    "manchester utd": "Manchester United",
    "man city": "Manchester City",
    "newcastle utd": "Newcastle United",
    "newcastle": "Newcastle United",
    "tottenham hotspur": "Tottenham",
    "spurs": "Tottenham",
    "wolves": "Wolverhampton",
    "wolverhampton wanderers": "Wolverhampton",
    "west ham united": "West Ham",
    "brighton & hove albion": "Brighton",
    "brighton and hove albion": "Brighton",
    "nott'ham forest": "Nottingham Forest",
    "nottm forest": "Nottingham Forest",
    "sheffield utd": "Sheffield United",
    "leicester": "Leicester City",
    "ipswich": "Ipswich Town",
    "bournemouth": "Bournemouth",
    "afc bournemouth": "Bournemouth",
    "luton": "Luton Town",
    "leeds": "Leeds United",
    "sunderland": "Sunderland",
    # FBRef team stats pages use full formal names
    "nott'ham forest": "Nottingham Forest",
    "west ham utd": "West Ham",
    "crystal pal": "Crystal Palace",
    "brentford fc": "Brentford",
    "wolverhampton wolves": "Wolverhampton",
    "brighton hove": "Brighton",
    # Understat aliases (usually full names, but a few edge cases)
    "manchester city": "Manchester City",
    "manchester united": "Manchester United",
    "newcastle united": "Newcastle United",
    "wolverhampton wanderers": "Wolverhampton",
    "nottingham forest": "Nottingham Forest",
    "sheffield united": "Sheffield United",
    "leicester city": "Leicester City",
    "ipswich town": "Ipswich Town",
    "luton town": "Luton Town",
    "leeds united": "Leeds United",
}


class EntityResolver:
    """
    Maps raw team name strings from any source to their canonical PL name.

    Parameters
    ----------
    canonical_teams : list[str]
        The authoritative list of team name strings. Defaults to
        ``CANONICAL_TEAMS``.
    threshold : int
        Minimum fuzzy-match score (0–100) required to accept a match.
        Names below this score raise ``ValueError``. Default is 85.
    """

    def __init__(
        self,
        canonical_teams: list[str] | None = None,
        threshold: int = 85,
    ) -> None:
        self._canonical = canonical_teams or CANONICAL_TEAMS
        self._threshold = threshold
        # Cache: raw name (lowered) -> canonical name
        self._cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Core resolution logic
    # ------------------------------------------------------------------

    def resolve(self, name: str) -> str:
        """
        Resolve a single raw team name to its canonical form.

        Resolution order:
        1. Cache hit (avoid redundant fuzzy calls on repeated names).
        2. Exact lookup in ``ALIAS_MAP`` (case-insensitive).
        3. Direct match in ``CANONICAL_TEAMS`` (case-insensitive).
        4. Fuzzy match via ``rapidfuzz`` WRatio scorer.

        Parameters
        ----------
        name : str
            Raw team name string from FBRef or Understat.

        Returns
        -------
        str
            Canonical team name.

        Raises
        ------
        ValueError
            If no match is found above the threshold  - manual review required.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"EntityResolver.resolve received an empty or non-string name: {name!r}")

        key = name.strip().lower()

        # 1. Cache
        if key in self._cache:
            return self._cache[key]

        # 2. Exact alias map lookup
        if key in ALIAS_MAP:
            canonical = ALIAS_MAP[key]
            logger.info("Resolved (alias): '%s' -> '%s'", name, canonical)
            self._cache[key] = canonical
            return canonical

        # 3. Direct canonical match (case-insensitive)
        canonical_lower = {c.lower(): c for c in self._canonical}
        if key in canonical_lower:
            canonical = canonical_lower[key]
            self._cache[key] = canonical
            return canonical

        # 4. Fuzzy fallback
        result = fuzz_process.extractOne(
            name,
            self._canonical,
            scorer=fuzz.WRatio,
            score_cutoff=self._threshold,
        )
        if result is not None:
            canonical, score, _ = result
            logger.info(
                "Resolved (fuzzy, score=%d): '%s' -> '%s'",
                score,
                name,
                canonical,
            )
            self._cache[key] = canonical
            return canonical

        raise ValueError(
            f"EntityResolver: could not resolve '{name}' to a canonical team name "
            f"(best fuzzy score was below threshold {self._threshold}). "
            "Add it to ALIAS_MAP in entity_resolver.py."
        )

    def resolve_series(self, series: pd.Series) -> pd.Series:
        """
        Resolve all team names in a pandas Series.

        Parameters
        ----------
        series : pd.Series
            Series of raw team name strings.

        Returns
        -------
        pd.Series
            Series of canonical team names, same index as input.
        """
        return series.map(self.resolve)

    def resolve_dataframe(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Resolve team names in-place for the specified columns of a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing raw team name columns.
        columns : list[str]
            Column names to resolve (typically ``["home_team", "away_team"]``).

        Returns
        -------
        pd.DataFrame
            DataFrame with resolved team name columns (copy, original unchanged).
        """
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                logger.warning("EntityResolver: column '%s' not found in DataFrame  - skipping", col)
                continue
            original_unique = df[col].dropna().unique().tolist()
            df[col] = self.resolve_series(df[col])
            resolved_unique = df[col].dropna().unique().tolist()
            logger.info(
                "Resolved column '%s': %d unique names -> %d unique canonical names",
                col,
                len(original_unique),
                len(resolved_unique),
            )
        return df

    def get_cache(self) -> dict[str, str]:
        """Return the current resolution cache for inspection/debugging."""
        return dict(self._cache)
