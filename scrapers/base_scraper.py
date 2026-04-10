"""
scrapers/base_scraper.py
------------------------
Abstract base class for all data scrapers. Encapsulates ScraperAPI integration,
rotating User-Agent headers, and exponential-backoff retry logic so concrete
scrapers only need to implement their parsing logic.
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from urllib.parse import urlencode

import requests
import pandas as pd

from config import SCRAPER_API_KEY_SECONDARY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rotating User-Agent pool — mimics real browser traffic
# ---------------------------------------------------------------------------
_USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]

# HTTP status codes that warrant a retry attempt
_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class BaseScraper(ABC):
    """
    Abstract base for FBRef and Understat scrapers.

    All HTTP calls are routed through ScraperAPI to avoid IP bans and handle
    JavaScript rendering where required. Subclasses must implement
    ``scrape_season`` and ``scrape_all``.

    Parameters
    ----------
    api_key : str
        ScraperAPI key.
    max_retries : int
        Maximum number of retry attempts on transient failures. Default 3.
    backoff_factor : float
        Base delay (seconds) for exponential backoff: ``2^attempt * backoff_factor``.
    inter_request_sleep : float
        Fixed delay (seconds) between consecutive requests to avoid hammering
        the target server even when ScraperAPI handles rate limiting.
    timeout : int
        Per-request timeout in seconds. Default 60.
    """

    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        inter_request_sleep: float = 1.5,
        timeout: int = 60,
    ) -> None:
        self._api_key = api_key
        self._fallback_key = SCRAPER_API_KEY_SECONDARY
        self._using_fallback = False
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._inter_request_sleep = inter_request_sleep
        self._timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rotating_headers(self) -> dict[str, str]:
        """Return a headers dict with a randomly selected User-Agent."""
        return {
            "User-Agent": random.choice(_USER_AGENTS),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    def _build_scraper_url(
        self,
        target_url: str,
        render_js: bool = False,
        premium: bool = False,
    ) -> str:
        """
        Wrap *target_url* with ScraperAPI parameters.

        Parameters
        ----------
        target_url : str
            The real destination URL to fetch.
        render_js : bool
            When True, instructs ScraperAPI to execute JavaScript before
            returning the page source (required for Understat).
        premium : bool
            When True, routes through ScraperAPI's residential/premium proxy
            pool. Costs more credits but bypasses aggressive anti-bot measures
            (e.g. FBRef stats sub-pages that block datacenter IPs).

        Returns
        -------
        str
            Fully-formed ScraperAPI request URL.
        """
        key = self._fallback_key if self._using_fallback else self._api_key
        params: dict[str, str] = {
            "api_key": key,
            "url": target_url,
        }
        if render_js:
            params["render"] = "true"
        if premium:
            params["premium"] = "true"
        return f"http://api.scraperapi.com/?{urlencode(params)}"

    def _get(
        self,
        url: str,
        render_js: bool = False,
        premium: bool = False,
    ) -> requests.Response:
        """
        Execute a GET request through ScraperAPI with retry logic.

        Retries on network errors and HTTP 429/5xx responses using exponential
        backoff. Raises ``requests.HTTPError`` if all attempts are exhausted.

        Parameters
        ----------
        url : str
            The *real* target URL (not yet wrapped with ScraperAPI params).
        render_js : bool
            Forward to ``_build_scraper_url`` to enable JS rendering.
        premium : bool
            Forward to ``_build_scraper_url`` to enable residential proxies.

        Returns
        -------
        requests.Response
            Successful response object.

        Raises
        ------
        requests.HTTPError
            After *max_retries* consecutive failures.
        """
        scraper_url = self._build_scraper_url(url, render_js=render_js, premium=premium)
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                logger.debug("GET %s (attempt %d/%d)", url, attempt + 1, self._max_retries + 1)
                response = self._session.get(
                    scraper_url,
                    headers=self._rotating_headers(),
                    timeout=self._timeout,
                )

                # ScraperAPI returns 403 when credits are exhausted, or 500 on
                # premium-plan access failures. Switch to fallback key on either.
                if (
                    response.status_code in (403, 500)
                    and not self._using_fallback
                    and self._fallback_key
                ):
                    logger.warning(
                        "ScraperAPI primary key failed with HTTP %d — "
                        "switching to secondary key for %s",
                        response.status_code, url,
                    )
                    self._using_fallback = True
                    scraper_url = self._build_scraper_url(url, render_js=render_js, premium=premium)
                    continue

                if response.status_code in _RETRYABLE_STATUSES:
                    logger.warning(
                        "Received HTTP %d for %s — retrying in %.1fs",
                        response.status_code,
                        url,
                        2 ** attempt * self._backoff_factor,
                    )
                    time.sleep(2 ** attempt * self._backoff_factor)
                    last_exc = requests.HTTPError(response=response)
                    continue

                response.raise_for_status()
                time.sleep(self._inter_request_sleep)
                logger.debug("Successfully fetched %s (%d bytes)", url, len(response.content))
                return response

            except requests.exceptions.RequestException as exc:
                delay = 2 ** attempt * self._backoff_factor
                logger.error(
                    "Request error on attempt %d/%d for %s: %s — retrying in %.1fs",
                    attempt + 1,
                    self._max_retries + 1,
                    url,
                    exc,
                    delay,
                )
                last_exc = exc
                time.sleep(delay)

        raise requests.HTTPError(
            f"All {self._max_retries + 1} attempts failed for {url}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def scrape_season(self, season: str) -> pd.DataFrame:
        """
        Scrape data for a single season.

        Parameters
        ----------
        season : str
            Season identifier (format depends on the source — e.g. "2023-2024"
            for FBRef, or "2023" for Understat).

        Returns
        -------
        pd.DataFrame
            Raw, source-specific DataFrame for the requested season.
        """

    @abstractmethod
    def scrape_all(self) -> pd.DataFrame:
        """
        Scrape all configured seasons and return a concatenated DataFrame.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame across all seasons, sorted by date.
        """
