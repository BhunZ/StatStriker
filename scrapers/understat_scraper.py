"""
scrapers/understat_scraper.py
------------------------------
Scrapes Premier League data from Understat via its internal JSON API:
    GET https://understat.com/main/getLeagueData/EPL/{year}

The response contains three top-level keys:
  - ``dates``   : match-level data (goals, xG) — one entry per fixture
  - ``teams``   : team-level history with deep stats (PPDA, deep, xPTS)
  - ``players`` : player-level data (not used in this pipeline)

Both match xG and deep team stats are extracted from a SINGLE API call per
season — no extra ScraperAPI credits needed.

Match xG output schema (one row per completed match):
    date, home_team, away_team, home_goals_us, away_goals_us, home_xg,
    away_xg, season_year

Deep stats output schema (one row per team per match):
    date, team, is_home, goals_scored, goals_conceded, xg, xga, npxg, npxga,
    ppda, ppda_allowed, deep, deep_allowed, xpts, result, season_year
"""

import json
import logging
import time as _time
from pathlib import Path

import pandas as pd
import requests as _req

from config import (
    SCRAPER_API_KEY,
    UNDERSTAT_YEARS,
    UNDERSTAT_RAW_DIR,
    UNDERSTAT_DEEP_DIR,
    MAX_RETRIES,
    BACKOFF_FACTOR,
    INTER_REQUEST_SLEEP,
    REQUEST_TIMEOUT,
)
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

_UNDERSTAT_API_BASE = "https://understat.com/main/getLeagueData/EPL"

# Extra headers required for the Understat AJAX endpoint
_UNDERSTAT_HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://understat.com/league/EPL",
}


def _api_url(year: int) -> str:
    """Return the Understat getLeagueData endpoint for a given season year."""
    return f"{_UNDERSTAT_API_BASE}/{year}"


# --------------------------------------------------------------------------
# Flattening helpers
# --------------------------------------------------------------------------

def _flatten_match(match_dict: dict, season_year: int) -> dict | None:
    """
    Flatten a single Understat match object from the ``dates`` array.

    Returns None if the match has not yet been played.
    """
    if not match_dict.get("isResult", False):
        return None

    try:
        return {
            "date": pd.to_datetime(
                match_dict.get("datetime", ""), errors="coerce"
            ).normalize(),
            "home_team": str(match_dict.get("h", {}).get("title", "")).strip(),
            "away_team": str(match_dict.get("a", {}).get("title", "")).strip(),
            "home_goals_us": int(match_dict["goals"]["h"]),
            "away_goals_us": int(match_dict["goals"]["a"]),
            "home_xg": float(match_dict["xG"]["h"]),
            "away_xg": float(match_dict["xG"]["a"]),
            "season_year": season_year,
        }
    except (TypeError, ValueError, KeyError) as exc:
        logger.warning(
            "Could not flatten match id=%s: %s",
            match_dict.get("id"), exc,
        )
        return None


def _flatten_team_history(
    team_id: str,
    team_info: dict,
    year: int,
) -> list[dict]:
    """
    Flatten a single team's ``history`` array into row dicts for deep stats.

    Each entry in ``history`` is a per-match record containing PPDA, deep
    completions, xPTS, etc.

    Parameters
    ----------
    team_id : str
        Understat team ID.
    team_info : dict
        Team object with keys: id, title, history.
    year : int
        Season start year.

    Returns
    -------
    list[dict]
        List of flat row dicts, one per match.
    """
    team_name = str(team_info.get("title", "")).strip()
    history = team_info.get("history", [])
    rows: list[dict] = []

    for match in history:
        try:
            # PPDA: passes allowed per defensive action
            ppda_obj = match.get("ppda", {})
            ppda_att = float(ppda_obj.get("att", 0))
            ppda_def = float(ppda_obj.get("def", 1))
            ppda_val = ppda_att / ppda_def if ppda_def > 0 else 0.0

            # Opponent's PPDA
            ppda_allowed_obj = match.get("ppda_allowed", {})
            ppda_allowed_att = float(ppda_allowed_obj.get("att", 0))
            ppda_allowed_def = float(ppda_allowed_obj.get("def", 1))
            ppda_allowed_val = (
                ppda_allowed_att / ppda_allowed_def
                if ppda_allowed_def > 0 else 0.0
            )

            rows.append({
                "date": pd.to_datetime(
                    match.get("date", ""), errors="coerce"
                ).normalize(),
                "team": team_name,
                "is_home": str(match.get("h_a", "")).strip().lower() == "h",
                "goals_scored": int(match.get("scored", 0)),
                "goals_conceded": int(match.get("missed", 0)),
                "xg": float(match.get("xG", 0)),
                "xga": float(match.get("xGA", 0)),
                "npxg": float(match.get("npxG", 0)),
                "npxga": float(match.get("npxGA", 0)),
                "ppda": ppda_val,
                "ppda_allowed": ppda_allowed_val,
                "deep": int(match.get("deep", 0)),
                "deep_allowed": int(match.get("deep_allowed", 0)),
                "xpts": float(match.get("xpts", 0)),
                "result": str(match.get("result", "")).strip(),
                "season_year": year,
            })
        except (TypeError, ValueError, KeyError) as exc:
            logger.warning(
                "Could not flatten deep stats for %s match on %s: %s",
                team_name, match.get("date", "?"), exc,
            )

    return rows


class UnderstatScraper(BaseScraper):
    """
    Scrapes Premier League match xG and deep team stats from Understat.

    Both data types come from a single JSON API endpoint per season, so
    no extra ScraperAPI credits are consumed for deep stats.

    Parameters
    ----------
    api_key : str
        ScraperAPI key. Defaults to the value from ``config.py``.
    years : list[int]
        Season start years to scrape. Defaults to ``config.UNDERSTAT_YEARS``.
    save_raw : bool
        If True (default), DataFrames are saved as CSV immediately.
    """

    def __init__(
        self,
        api_key: str = SCRAPER_API_KEY,
        years: list[int] | None = None,
        save_raw: bool = True,
    ) -> None:
        super().__init__(
            api_key=api_key,
            max_retries=MAX_RETRIES,
            backoff_factor=BACKOFF_FACTOR,
            inter_request_sleep=INTER_REQUEST_SLEEP,
            timeout=REQUEST_TIMEOUT,
        )
        self._years = years or UNDERSTAT_YEARS
        self._save_raw = save_raw

        # Add Understat-specific headers to the shared session
        self._session.headers.update(_UNDERSTAT_HEADERS)

        # Cache for full API payloads to avoid duplicate requests
        self._payload_cache: dict[int, dict] = {}

    # ======================================================================
    #  CORE: Fetch full API payload (shared by match xG + deep stats)
    # ======================================================================

    def _fetch_payload(self, year: int) -> dict:
        """
        Fetch the full getLeagueData payload for a season.

        Uses the Understat JSON API directly (not ScraperAPI) because
        ScraperAPI incorrectly routes JSON endpoints. Results are cached
        so both match xG and deep stats can share a single request.

        Parameters
        ----------
        year : int
            Season start year.

        Returns
        -------
        dict
            Full API response with keys: dates, teams, players.
        """
        if year in self._payload_cache:
            logger.debug("Using cached payload for year %d", year)
            return self._payload_cache[year]

        url = _api_url(year)
        logger.info("Fetching Understat EPL data for year %d via API ...", year)

        last_exc = None
        response = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._session.get(url, timeout=self._timeout)
                if response.status_code == 404:
                    raise ValueError(
                        f"Understat API 404 for year {year} — "
                        "season may not exist yet on Understat."
                    )
                response.raise_for_status()
                _time.sleep(self._inter_request_sleep)
                break
            except Exception as exc:
                delay = 2 ** attempt * self._backoff_factor
                logger.warning(
                    "Understat API attempt %d/%d for year %d: %s — retrying in %.1fs",
                    attempt + 1, self._max_retries + 1, year, exc, delay,
                )
                last_exc = exc
                _time.sleep(delay)
        else:
            raise _req.HTTPError(
                f"All {self._max_retries + 1} attempts failed for Understat year {year}"
            ) from last_exc

        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Understat API response for year {year} is not valid JSON: {exc}"
            ) from exc

        # Validate expected keys
        for key in ("dates", "teams"):
            if key not in payload:
                raise ValueError(
                    f"Understat API response for year {year} missing '{key}' key. "
                    f"Top-level keys: {list(payload.keys())}"
                )

        logger.info(
            "Understat year %d: %d matches, %d teams in API response",
            year, len(payload["dates"]), len(payload["teams"]),
        )

        self._payload_cache[year] = payload
        return payload

    # ======================================================================
    #  PART A: MATCH-LEVEL xG (from ``dates`` key)
    # ======================================================================

    def _scrape_year_matches(self, year: int) -> pd.DataFrame:
        """
        Extract match-level xG data for one season.

        Parameters
        ----------
        year : int
            Season start year.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame of match-level xG data.
        """
        payload = self._fetch_payload(year)
        raw_matches = payload["dates"]

        rows = [_flatten_match(m, year) for m in raw_matches]
        rows = [r for r in rows if r is not None]

        if not rows:
            logger.warning("Understat year %d: no completed matches found", year)
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.dropna(subset=["date"]).copy()
        df = df.sort_values("date").reset_index(drop=True)

        logger.info("Understat year %d: %d completed matches parsed", year, len(df))

        if self._save_raw:
            path = UNDERSTAT_RAW_DIR / f"understat_{year}.csv"
            df.to_csv(path, index=False)
            logger.info("Saved raw Understat match data -> %s", path)

        return df

    # ======================================================================
    #  PART B: DEEP STATS (PPDA, deep, xPTS from ``teams`` key)
    # ======================================================================

    def _scrape_year_deep(self, year: int) -> pd.DataFrame:
        """
        Extract deep team stats for one season from the ``teams`` key.

        Parameters
        ----------
        year : int
            Season start year.

        Returns
        -------
        pd.DataFrame
            Deep stats DataFrame (one row per team per match).
        """
        payload = self._fetch_payload(year)
        teams_data = payload["teams"]

        all_rows: list[dict] = []
        for team_id, team_info in teams_data.items():
            all_rows.extend(_flatten_team_history(team_id, team_info, year))

        if not all_rows:
            logger.warning("Understat year %d: no deep stats rows produced", year)
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df = df.dropna(subset=["date"]).copy()
        df = df.sort_values(["date", "team"]).reset_index(drop=True)

        logger.info(
            "Understat year %d: %d deep stats rows (%d teams × ~%d matches)",
            year, len(df),
            df["team"].nunique(),
            len(df) // max(df["team"].nunique(), 1),
        )

        if self._save_raw:
            path = UNDERSTAT_DEEP_DIR / f"understat_deep_{year}.csv"
            df.to_csv(path, index=False)
            logger.info("Saved Understat deep stats -> %s", path)

        return df

    def scrape_deep_stats(self, year: int) -> pd.DataFrame:
        """Public wrapper for deep stats extraction."""
        return self._scrape_year_deep(year)

    def scrape_all_deep_stats(self) -> pd.DataFrame:
        """
        Scrape deep stats for all configured seasons.

        Returns
        -------
        pd.DataFrame
            Combined deep stats across all seasons.
        """
        frames: list[pd.DataFrame] = []
        for year in self._years:
            try:
                df = self._scrape_year_deep(year)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to scrape Understat deep stats for year %d: %s — skipping",
                    year, exc,
                    exc_info=True,
                )

        if not frames:
            logger.warning(
                "UnderstatScraper.scrape_all_deep_stats: no years could be scraped"
            )
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(["date", "team"]).reset_index(drop=True)
        logger.info(
            "UnderstatScraper deep stats: %d total rows across %d years",
            len(combined), len(frames),
        )
        return combined

    # ======================================================================
    #  PUBLIC INTERFACE (implements BaseScraper)
    # ======================================================================

    def scrape_season(self, season: str) -> pd.DataFrame:
        """Scrape Understat match xG data for a single season."""
        return self._scrape_year_matches(int(season))

    def scrape_all(self) -> pd.DataFrame:
        """Scrape all configured season years' match xG data and concatenate."""
        frames: list[pd.DataFrame] = []
        for year in self._years:
            try:
                df = self._scrape_year_matches(year)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to scrape Understat year %d: %s — skipping",
                    year, exc,
                    exc_info=True,
                )

        if not frames:
            raise RuntimeError(
                "UnderstatScraper.scrape_all: no years could be scraped"
            )

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("date").reset_index(drop=True)
        logger.info(
            "UnderstatScraper: %d total matches across %d years",
            len(combined), len(frames),
        )
        return combined
