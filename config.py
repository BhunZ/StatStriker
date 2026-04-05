"""
config.py
---------
Central configuration module. Loads environment variables from .env and
exposes typed constants used across the entire pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = ROOT_DIR / "logs"

FBREF_RAW_DIR = RAW_DIR / "fbref"
FBREF_TEAM_STATS_DIR = RAW_DIR / "fbref" / "team_stats"
UNDERSTAT_RAW_DIR = RAW_DIR / "understat"
UNDERSTAT_DEEP_DIR = RAW_DIR / "understat" / "deep_stats"

# Ensure directories exist at import time
for _d in (FBREF_RAW_DIR, FBREF_TEAM_STATS_DIR, UNDERSTAT_RAW_DIR, UNDERSTAT_DEEP_DIR, PROCESSED_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv(ROOT_DIR / ".env")

# Primary key (from .env). If exhausted, the pipeline falls back to the
# secondary key automatically via BaseScraper._get().
SCRAPER_API_KEY: str = os.environ.get("SCRAPER_API_KEY", "")
SCRAPER_API_KEY_SECONDARY: str = os.environ.get("SCRAPER_API_KEY_SECONDARY", "")
SCRAPER_API_BASE: str = "http://api.scraperapi.com"

if not SCRAPER_API_KEY:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "SCRAPER_API_KEY is not set — scrapers will not work. "
        "Add it to your .env file or environment if you need to scrape."
    )

# ---------------------------------------------------------------------------
# Scraping targets
# ---------------------------------------------------------------------------
# FBRef uses "{start_year}-{end_year}" slugs in the URL
FBREF_SEASONS: list[str] = ["2023-2024", "2024-2025", "2025-2026"]

# Understat uses the season *start* year
UNDERSTAT_YEARS: list[int] = [2023, 2024, 2025]

# FBRef Premier League competition ID
FBREF_COMP_ID: str = "9"

# ---------------------------------------------------------------------------
# HTTP / retry settings
# ---------------------------------------------------------------------------
MAX_RETRIES: int = 3
BACKOFF_FACTOR: float = 2.0        # seconds; delay = 2^attempt * backoff_factor
REQUEST_TIMEOUT: int = 60          # seconds per individual request
INTER_REQUEST_SLEEP: float = 1.5   # polite delay between consecutive requests

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
# FBRef team-level aggregate stat pages — matches the current FBRef interface:
# Standard Stats | Goalkeeping | Shooting | Playing Time | Miscellaneous Stats
FBREF_STAT_TYPES: list[str] = ["stats", "keepers", "shooting", "playingtime", "misc"]

MODEL_OUTPUT_DIR = DATA_DIR / "models"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FUZZY_MATCH_THRESHOLD: int = 85    # minimum rapidfuzz score to accept a team name match
ROLLING_WINDOWS: list[int] = [5, 10]
EWMA_SPAN: int = 10
