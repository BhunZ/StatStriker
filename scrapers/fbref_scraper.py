"""
scrapers/fbref_scraper.py
--------------------------
Scrapes Premier League data from FBRef:
  1. Fixture/result tables (per-match: goals, xG, possession, shots)
  2. Team aggregate stat pages (per-season: standard, shooting, passing,
     possession, defense)

FBRef pages are static HTML so JS rendering is not required.

Fixture output schema (one row per completed match):
    date          : pd.Timestamp
    home_team     : str  (raw FBRef name — resolved later by EntityResolver)
    away_team     : str
    home_goals    : int
    away_goals    : int
    home_xg_fbref : float | NaN
    away_xg_fbref : float | NaN
    home_poss     : float | NaN  (% possession)
    away_poss     : float | NaN
    home_shots    : int | NaN
    away_shots    : int | NaN
    season        : str  (e.g. "2023-2024")
"""

import io
import logging
import re
import time as _time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config import (
    SCRAPER_API_KEY,
    SCRAPER_API_KEY_SECONDARY,
    FBREF_SEASONS,
    FBREF_COMP_ID,
    FBREF_RAW_DIR,
    FBREF_TEAM_STATS_DIR,
    FBREF_STAT_TYPES,
    MAX_RETRIES,
    BACKOFF_FACTOR,
    INTER_REQUEST_SLEEP,
    REQUEST_TIMEOUT,
)
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

# Regex to parse "2–1" or "2-1" style score strings
_SCORE_RE = re.compile(r"(\d+)\s*[–\-]\s*(\d+)")

# --------------------------------------------------------------------------
# FBRef URL builders
# --------------------------------------------------------------------------

def _fixtures_url(season: str) -> str:
    """Build the FBRef schedule URL for a given season slug."""
    return (
        f"https://fbref.com/en/comps/{FBREF_COMP_ID}/{season}/schedule/"
        f"Premier-League-Scores-and-Fixtures"
    )


def _team_stats_url(season: str, stat_type: str) -> str:
    """Build the FBRef team stats URL for a given season and stat type."""
    return (
        f"https://fbref.com/en/comps/{FBREF_COMP_ID}/{season}/"
        f"{stat_type}/Premier-League-Stats"
    )


# --------------------------------------------------------------------------
# Known FBRef table IDs for team stats pages — matches the current FBRef
# interface: Standard Stats | Goalkeeping | Shooting | Playing Time | Misc
# --------------------------------------------------------------------------

_STATS_TABLE_IDS: dict[str, str] = {
    "stats":       "stats_squads_standard_for",
    "keepers":     "stats_squads_keeper_for",
    "shooting":    "stats_squads_shooting_for",
    "playingtime": "stats_squads_playing_time_for",
    "misc":        "stats_squads_misc_for",
}


def _parse_score(score_str: str) -> tuple[Optional[int], Optional[int]]:
    """
    Extract home and away goals from a score string like "2–1".

    Returns (None, None) when the match has not yet been played.
    """
    if not isinstance(score_str, str):
        return None, None
    match = _SCORE_RE.search(score_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


# --------------------------------------------------------------------------
# Column mapping helpers
# --------------------------------------------------------------------------

def _find_column(columns: list[str], candidates: list[str],
                 regex_pattern: str | None = None) -> str | None:
    """
    Find the first matching column name from a list of candidates.
    Falls back to regex if no exact match is found.

    Parameters
    ----------
    columns : list[str]
        Available column names (lowercase, normalized).
    candidates : list[str]
        Exact names to try first, in priority order.
    regex_pattern : str | None
        Regex to use as fallback if no candidate matches.

    Returns
    -------
    str | None
        Matched column name or None.
    """
    for cand in candidates:
        if cand in columns:
            return cand
    if regex_pattern:
        pat = re.compile(regex_pattern)
        for col in columns:
            if pat.fullmatch(col):
                return col
    return None


def _find_paired_columns(columns: list[str],
                         regex_pattern: str) -> tuple[str | None, str | None]:
    """
    Find a pair of columns matching a regex (home = first, away = second).

    FBRef's fixtures table repeats column names for home/away (e.g., two 'xg'
    columns). After ``pd.read_html`` flattening, they appear as 'xg' and
    'xg_1', or with various prefixes. This function collects ALL matches and
    returns the first two in order.

    Returns
    -------
    (home_col, away_col)
        Column names for the home and away version, or None if not found.
    """
    pat = re.compile(regex_pattern)
    matches = [c for c in columns if pat.search(c)]
    home = matches[0] if len(matches) >= 1 else None
    away = matches[1] if len(matches) >= 2 else None
    return home, away


# ==========================================================================
# FBRefScraper
# ==========================================================================

class FBRefScraper(BaseScraper):
    """
    Scrapes Premier League match data and team aggregate stats from FBRef.

    Uses ScraperAPI for IP rotation. Parses HTML tables via ``pd.read_html``,
    then cleans and normalises the resulting DataFrames.

    Parameters
    ----------
    api_key : str
        ScraperAPI key. Defaults to the value from ``config.py``.
    seasons : list[str]
        FBRef season slugs to scrape. Defaults to ``config.FBREF_SEASONS``.
    save_raw : bool
        If True (default), each scraped DataFrame is saved as CSV immediately.
    """

    def __init__(
        self,
        api_key: str = SCRAPER_API_KEY,
        seasons: list[str] | None = None,
        save_raw: bool = True,
    ) -> None:
        super().__init__(
            api_key=api_key,
            max_retries=MAX_RETRIES,
            backoff_factor=BACKOFF_FACTOR,
            inter_request_sleep=INTER_REQUEST_SLEEP,
            timeout=REQUEST_TIMEOUT,
        )
        self._seasons = seasons or FBREF_SEASONS
        self._save_raw = save_raw

    # ======================================================================
    #  PART A: FIXTURES (per-match data)
    # ======================================================================

    def _fetch_raw_table(self, season: str) -> pd.DataFrame:
        """
        Fetch the HTML fixture table for *season* and return it as a raw
        (uncleaned) DataFrame.

        Uses a multi-layered fallback strategy to locate the correct table:
        1. Canonical table ID: ``sched_{season}_{comp_id}_1``
        2. Regex search for any ``sched_*_1`` ID
        3. Heuristic: table containing a "score" or "result" column
        4. Last resort: first table on the page
        """
        url = _fixtures_url(season)
        logger.info("Fetching FBRef fixtures for season %s ...", season)
        response = self._get(url, render_js=False)
        html = response.text

        # Strategy 1: deterministic canonical table ID
        canonical_id = f"sched_{season}_{FBREF_COMP_ID}_1"
        if canonical_id in html:
            logger.debug("Found canonical table id='%s'", canonical_id)
            try:
                tables = pd.read_html(io.StringIO(html), attrs={"id": canonical_id})
                if tables:
                    return tables[0]
            except ValueError:
                logger.debug("Could not parse table id='%s' — falling back", canonical_id)
        else:
            # Strategy 2: regex fallback for any sched_..._1 id
            match = re.search(r'id="(sched_[^"]+_1)"', html)
            if match and "_sh" not in match.group(1) and "_link" not in match.group(1):
                table_id = match.group(1)
                logger.debug("Using regex-found table id='%s'", table_id)
                try:
                    tables = pd.read_html(io.StringIO(html), attrs={"id": table_id})
                    if tables:
                        return tables[0]
                except ValueError:
                    pass
            logger.debug("No primary table id found for season %s — falling back", season)

        # Strategy 3 & 4: heuristic scan of all tables
        try:
            all_tables = pd.read_html(io.StringIO(html))
            for tbl in all_tables:
                cols_lower = [str(c).lower() for c in tbl.columns]
                if any("score" in c or "result" in c for c in cols_lower):
                    logger.warning(
                        "Using heuristic fallback table for season %s — verify column mapping",
                        season,
                    )
                    return tbl
            if all_tables:
                logger.warning(
                    "Using first table as last-resort fallback for season %s",
                    season,
                )
                return all_tables[0]
        except ValueError:
            pass

        raise ValueError(f"No parseable HTML table found on FBRef page for season {season}")

    @staticmethod
    def _clean_table(raw: pd.DataFrame, season: str) -> pd.DataFrame:
        """
        Transform the raw FBRef fixture table into the canonical output schema.

        Steps
        -----
        1. Flatten multi-level columns.
        2. Normalize to lowercase snake_case.
        3. Drop separator rows.
        4. Map columns via candidates + regex fallback.
        5. Parse Score → home_goals / away_goals.
        6. Drop unplayed matches, parse dates, cast types.
        """
        df = raw.copy()

        # --- 1. Flatten multi-level column headers ---
        if isinstance(df.columns, pd.MultiIndex):
            new_cols = []
            for col in df.columns:
                parts = [str(c).strip() for c in col
                         if str(c).strip() and str(c).strip().lower() != "nan"
                         and not str(c).strip().lower().startswith("unnamed")]
                new_cols.append("_".join(parts) if parts else str(col))
            df.columns = new_cols

        # --- 2. Normalize column names ---
        df.columns = [
            re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_")
            for c in df.columns
        ]

        logger.debug("FBRef columns after normalization for %s: %s", season, list(df.columns))

        # --- 3. Drop repeated header rows ---
        if "wk" in df.columns:
            df = df[pd.to_numeric(df["wk"], errors="coerce").notna()].copy()

        # --- 4. Map columns (candidates + regex fallback) ---
        cols = list(df.columns)
        col_map: dict[str, str] = {}

        # Home team
        home_col = _find_column(cols, ["home", "home_team"], r"home.*")
        if home_col:
            col_map[home_col] = "home_team"

        # Away team
        away_col = _find_column(cols, ["away", "visitor", "away_team"], r"(away|visitor).*")
        if away_col:
            col_map[away_col] = "away_team"

        # Score
        score_col = _find_column(cols, ["score", "result"], r"(score|result).*")
        if score_col:
            col_map[score_col] = "score_raw"

        # --- xG: use paired-column finder (home first, away second) ---
        # Exclude columns that look like 'xg_per90' or 'npxg' variants
        xg_home, xg_away = _find_paired_columns(cols, r"^(?:(?!npxg).)*xg(?!.*per).*$")
        # Simpler fallback: just look for anything with 'xg'
        if xg_home is None and xg_away is None:
            xg_home, xg_away = _find_paired_columns(cols, r".*\bxg\b.*")
        if xg_home and xg_home not in col_map:
            col_map[xg_home] = "home_xg_fbref"
        if xg_away and xg_away not in col_map:
            col_map[xg_away] = "away_xg_fbref"

        # --- Possession: paired columns ---
        poss_home, poss_away = _find_paired_columns(cols, r".*\bposs(?:ession)?\b.*")
        if poss_home and poss_home not in col_map:
            col_map[poss_home] = "home_poss"
        if poss_away and poss_away not in col_map:
            col_map[poss_away] = "away_poss"

        # --- Shots: paired columns ---
        # Be careful not to match 'sot' (shots on target) — match 'sh' or 'shots'
        sh_home, sh_away = _find_paired_columns(cols, r"^(?:sh|shots)(?:_\d+)?$")
        if sh_home and sh_home not in col_map:
            col_map[sh_home] = "home_shots"
        if sh_away and sh_away not in col_map:
            col_map[sh_away] = "away_shots"

        # Log mapping results
        logger.debug("FBRef column mapping for %s: %s", season, col_map)

        # Warn about unmapped expected columns
        mapped_targets = set(col_map.values())
        for expected in ("home_xg_fbref", "away_xg_fbref", "home_poss", "away_poss",
                         "home_shots", "away_shots"):
            if expected not in mapped_targets:
                logger.warning(
                    "FBRef season %s: could not map column '%s' — "
                    "available columns: %s",
                    season, expected, cols,
                )

        df = df.rename(columns=col_map)

        # --- 5. Parse score → goals ---
        if "score_raw" in df.columns:
            parsed = df["score_raw"].apply(_parse_score)
            df["home_goals"] = parsed.apply(lambda t: t[0])
            df["away_goals"] = parsed.apply(lambda t: t[1])
            df = df.drop(columns=["score_raw"])

        # Drop unplayed matches
        df = df.dropna(subset=["home_goals", "away_goals"]).copy()

        # --- 6. Parse date ---
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])

        # --- 7. Numeric casting ---
        int_cols = ["home_goals", "away_goals", "home_shots", "away_shots"]
        float_cols = ["home_xg_fbref", "away_xg_fbref", "home_poss", "away_poss"]

        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Strip whitespace from team names
        for col in ("home_team", "away_team"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Attach season label
        df["season"] = season

        # Select and order output columns
        desired_cols = [
            "date", "home_team", "away_team",
            "home_goals", "away_goals",
            "home_xg_fbref", "away_xg_fbref",
            "home_poss", "away_poss",
            "home_shots", "away_shots",
            "season",
        ]
        output_cols = [c for c in desired_cols if c in df.columns]
        df = df[output_cols].reset_index(drop=True)

        logger.info(
            "FBRef season %s: %d completed matches parsed, columns: %s",
            season, len(df), list(df.columns),
        )
        return df

    def _save_fixtures_csv(self, df: pd.DataFrame, season: str) -> Path:
        """Persist a season's fixture DataFrame to ``data/raw/fbref/``."""
        path = FBREF_RAW_DIR / f"fbref_{season}.csv"
        df.to_csv(path, index=False)
        logger.info("Saved raw FBRef fixtures -> %s", path)
        return path

    # ======================================================================
    #  PART B: TEAM AGGREGATE STATS (per-season per-team)
    # ======================================================================

    def _fetch_stats_table(self, season: str, stat_type: str) -> pd.DataFrame:
        """
        Fetch a team aggregate stats table from FBRef.

        Parameters
        ----------
        season : str
            Season slug, e.g. "2023-2024".
        stat_type : str
            One of the current FBRef stat categories: stats, keepers, shooting,
            playingtime, misc.

        Returns
        -------
        pd.DataFrame
            Raw (uncleaned) DataFrame from the FBRef stats page.
        """
        url = _team_stats_url(season, stat_type)
        expected_id = _STATS_TABLE_IDS.get(stat_type, "")

        logger.info("Fetching FBRef %s stats for season %s ...", stat_type, season)

        # FBRef aggressively blocks datacenter IPs on stats sub-pages (HTTP 500).
        # Rotate to a fresh session, sleep 6s, and route through ScraperAPI's
        # residential proxy pool (premium=True) to bypass the block.
        # Residential proxies are slower — temporarily raise timeout to 120s.
        self._session = requests.Session()
        saved_timeout = self._timeout
        self._timeout = 120
        _time.sleep(6.0)

        try:
            response = self._get(url, render_js=False, premium=True)
        finally:
            self._timeout = saved_timeout
        html = response.text

        # Strategy 1: known table ID
        if expected_id and expected_id in html:
            try:
                tables = pd.read_html(io.StringIO(html), attrs={"id": expected_id})
                if tables:
                    logger.debug("Found table id='%s' for %s/%s", expected_id, stat_type, season)
                    return tables[0]
            except ValueError:
                logger.debug("Could not parse table id='%s' — falling back", expected_id)

        # Strategy 2: regex search for stats_squads_*_for
        match = re.search(r'id="(stats_squads_[^"]*_for)"', html)
        if match:
            table_id = match.group(1)
            logger.debug("Using regex-found table id='%s' for %s/%s", table_id, stat_type, season)
            try:
                tables = pd.read_html(io.StringIO(html), attrs={"id": table_id})
                if tables:
                    return tables[0]
            except ValueError:
                pass

        # Strategy 3: scan all tables for one with a "Squad" column
        try:
            all_tables = pd.read_html(io.StringIO(html))
            for tbl in all_tables:
                flat_cols = []
                for c in tbl.columns:
                    if isinstance(c, tuple):
                        flat_cols.append("_".join(str(x) for x in c).lower())
                    else:
                        flat_cols.append(str(c).lower())
                if any("squad" in c for c in flat_cols):
                    logger.warning(
                        "Using heuristic fallback table for %s/%s — verify columns",
                        stat_type, season,
                    )
                    return tbl
            if all_tables:
                logger.warning("Using first table for %s/%s", stat_type, season)
                return all_tables[0]
        except ValueError:
            pass

        raise ValueError(
            f"No parseable stats table found for {stat_type}/{season} on FBRef"
        )

    @staticmethod
    def _clean_stats_table(
        raw: pd.DataFrame,
        stat_type: str,
        season: str,
    ) -> pd.DataFrame:
        """
        Clean a raw FBRef team stats table into a normalized DataFrame.

        Handles multi-level headers by joining with underscores and normalizing.
        Drops aggregate rows (like league totals) and the repeated header rows.

        Parameters
        ----------
        raw : pd.DataFrame
            Raw DataFrame from ``pd.read_html``.
        stat_type : str
            Stat type name (for logging and column prefixing).
        season : str
            Season slug — stored as a column.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with one row per squad.
        """
        df = raw.copy()

        # Flatten multi-level headers
        if isinstance(df.columns, pd.MultiIndex):
            new_cols = []
            for col_tuple in df.columns:
                parts = []
                for level_val in col_tuple:
                    s = str(level_val).strip()
                    # Skip generic/empty level names
                    if (s.lower().startswith("unnamed") or s.lower() == "nan"
                            or s == "" or s.lower() == "for"):
                        continue
                    parts.append(s)
                new_cols.append("_".join(parts) if parts else "unknown")
            df.columns = new_cols

        # Normalize column names
        df.columns = [
            re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_")
            for c in df.columns
        ]

        # Deduplicate column names (FBRef sometimes has e.g. two 'sot' columns
        # for SoT and SoT% which flatten identically). Append _2, _3, etc.
        seen: dict[str, int] = {}
        deduped: list[str] = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                deduped.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 1
                deduped.append(col)
        df.columns = deduped

        logger.debug(
            "FBRef %s stats columns for %s: %s",
            stat_type, season, list(df.columns),
        )

        # Find the squad column
        squad_col = None
        for cand in df.columns:
            if "squad" in cand:
                squad_col = cand
                break

        if squad_col is None:
            logger.warning(
                "No 'squad' column found in %s stats for %s — columns: %s",
                stat_type, season, list(df.columns),
            )
            return pd.DataFrame()

        # Rename squad column
        if squad_col != "squad":
            df = df.rename(columns={squad_col: "squad"})

        # Drop repeated header rows and aggregate rows (e.g., "Squad" text or NaN)
        df = df[df["squad"].notna()].copy()
        df = df[~df["squad"].astype(str).str.contains(
            r"squad|vs", case=False, na=False
        )].copy()

        # Strip whitespace from squad names
        df["squad"] = df["squad"].astype(str).str.strip()

        # Drop rows that are clearly not teams (league totals, etc.)
        # FBRef sometimes has a row with squad = "Premier League" or similar
        df = df[~df["squad"].str.contains(
            r"league|total|average", case=False, na=False
        )].copy()

        # Add metadata columns
        df["season"] = season
        df["stat_type"] = stat_type

        # Find matches_played column
        for cand in ("mp", "matches_played"):
            if cand in df.columns:
                df["matches_played"] = pd.to_numeric(df[cand], errors="coerce")
                break

        # Convert all numeric-looking columns to numeric
        for col in df.columns:
            if col not in ("squad", "season", "stat_type"):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.reset_index(drop=True)
        logger.info(
            "FBRef %s stats for %s: %d squads, %d columns",
            stat_type, season, len(df), len(df.columns),
        )
        return df

    def _save_stats_csv(
        self, df: pd.DataFrame, stat_type: str, season: str
    ) -> Path:
        """Persist team stats to ``data/raw/fbref/team_stats/``."""
        path = FBREF_TEAM_STATS_DIR / f"fbref_team_{stat_type}_{season}.csv"
        df.to_csv(path, index=False)
        logger.info("Saved FBRef %s stats -> %s", stat_type, path)
        return path

    def scrape_team_stats_season(
        self, season: str
    ) -> dict[str, pd.DataFrame]:
        """
        Scrape all team aggregate stat pages for a single season.

        Parameters
        ----------
        season : str
            Season slug, e.g. "2023-2024".

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of stat_type -> cleaned DataFrame.
        """
        results: dict[str, pd.DataFrame] = {}
        for stat_type in FBREF_STAT_TYPES:
            try:
                raw = self._fetch_stats_table(season, stat_type)
                clean = self._clean_stats_table(raw, stat_type, season)
                if not clean.empty and self._save_raw:
                    self._save_stats_csv(clean, stat_type, season)
                results[stat_type] = clean
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to scrape FBRef %s stats for %s: %s — skipping",
                    stat_type, season, exc,
                    exc_info=True,
                )
        return results

    def scrape_all_team_stats(self) -> dict[str, pd.DataFrame]:
        """
        Scrape all team aggregate stats across all configured seasons.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of stat_type -> concatenated DataFrame across all seasons.
        """
        combined: dict[str, list[pd.DataFrame]] = {st: [] for st in FBREF_STAT_TYPES}

        for season in self._seasons:
            season_results = self.scrape_team_stats_season(season)
            for stat_type, df in season_results.items():
                if not df.empty:
                    combined[stat_type].append(df)

        result: dict[str, pd.DataFrame] = {}
        for stat_type, frames in combined.items():
            if frames:
                result[stat_type] = (
                    pd.concat(frames, ignore_index=True)
                    .reset_index(drop=True)
                )
                logger.info(
                    "FBRef %s stats: %d total rows across %d seasons",
                    stat_type, len(result[stat_type]), len(frames),
                )
            else:
                logger.warning("FBRef %s stats: no data scraped", stat_type)

        return result

    # ======================================================================
    #  PUBLIC INTERFACE (implements BaseScraper)
    # ======================================================================

    def scrape_season(self, season: str) -> pd.DataFrame:
        """
        Scrape and clean FBRef fixture data for a single season.

        Parameters
        ----------
        season : str
            Season slug in "YYYY-YYYY" format, e.g. "2023-2024".

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with the canonical FBRef fixture column set.
        """
        raw = self._fetch_raw_table(season)
        clean = self._clean_table(raw, season)
        if self._save_raw:
            self._save_fixtures_csv(clean, season)
        return clean

    def scrape_all(self) -> pd.DataFrame:
        """
        Scrape all configured seasons' fixtures and concatenate.

        Seasons that fail to scrape are skipped with an error log so a single
        bad season does not abort the entire pipeline.

        Returns
        -------
        pd.DataFrame
            Combined, date-sorted DataFrame across all seasons.
        """
        frames: list[pd.DataFrame] = []
        for season in self._seasons:
            try:
                frames.append(self.scrape_season(season))
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to scrape FBRef season %s: %s — skipping",
                    season, exc,
                    exc_info=True,
                )

        if not frames:
            raise RuntimeError("FBRefScraper.scrape_all: no seasons could be scraped")

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("date").reset_index(drop=True)
        logger.info(
            "FBRefScraper: %d total matches across %d seasons",
            len(combined), len(frames),
        )
        return combined
