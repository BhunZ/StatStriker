"""
main.py
--------
CLI entry point for the Premier League match prediction pipeline.

Execution Modes
---------------
    python main.py --auto                  # Smart: scrape/train/infer as needed
    python main.py --predict Arsenal Chelsea  # Load models and predict
    python main.py --force-train           # Retrain immediately
    python main.py --scrape                # Manual scrape only
    python main.py --process              # Manual process only
    python main.py --evaluate             # Full temporal CV evaluation

    # Legacy modes (still supported)
    python main.py --all                   # Full scrape + process pipeline
    python main.py --train                 # Train models on existing data

Exit Codes (for CI/CD orchestration)
-------------------------------------
    0 : Success — inference only, no state change
    2 : New data scraped and processed
    3 : Full model retraining completed
    1 : Error occurred

Storage Backends
-----------------
    python main.py --auto --storage local          # Default: local disk
    python main.py --auto --storage s3 --s3-bucket my-bucket --s3-prefix prod/v1

Logging
-------
Structured output is written to both stdout (INFO+) and a rotating file at
``logs/pipeline.log`` (DEBUG+) so every HTTP call and processing step is
fully traceable.
"""

import argparse
import logging
import sys
import time
from datetime import date, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd

from config import (
    FBREF_RAW_DIR,
    FBREF_TEAM_STATS_DIR,
    UNDERSTAT_RAW_DIR,
    UNDERSTAT_DEEP_DIR,
    LOGS_DIR,
    FBREF_SEASONS,
    UNDERSTAT_YEARS,
    PROCESSED_DIR,
    MODEL_OUTPUT_DIR,
)
from scrapers import FBRefScraper, UnderstatScraper
from processors import EntityResolver, SchemaNormalizer, FeatureEngineer
from storage import StorageManager, StorageBackend


# ---------------------------------------------------------------------------
# Constants — automation thresholds
# ---------------------------------------------------------------------------

SCRAPE_INTERVAL_DAYS: int = 7      # minimum days between scrapes
RETRAIN_MATCH_THRESHOLD: int = 80  # new matches needed to trigger retrain
DECAY_THRESHOLD: float = 1.05      # 5% log-loss increase triggers retrain


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    """
    Set up a root logger with two handlers:
    - StreamHandler -> INFO level (human-readable pipeline progress)
    - RotatingFileHandler -> DEBUG level (full request/parsing trace)
    """
    log_format = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    log_file = LOGS_DIR / "pipeline.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    root.addHandler(console)
    root.addHandler(file_handler)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline stages: SCRAPING
# ---------------------------------------------------------------------------

def run_fbref_scrape() -> pd.DataFrame:
    """Scrape all FBRef fixtures and return the combined DataFrame."""
    logger.info("=== Stage: FBRef Fixtures Scraping ===")
    t0 = time.perf_counter()
    scraper = FBRefScraper(save_raw=True)
    df = scraper.scrape_all()
    elapsed = time.perf_counter() - t0
    logger.info("FBRef fixtures scrape complete: %d matches in %.1fs", len(df), elapsed)
    return df


def run_fbref_team_stats_scrape() -> dict[str, pd.DataFrame]:
    """Scrape FBRef team aggregate stats (5 stat pages x N seasons)."""
    logger.info("=== Stage: FBRef Team Stats Scraping ===")
    t0 = time.perf_counter()
    scraper = FBRefScraper(save_raw=True)
    stats = scraper.scrape_all_team_stats()
    elapsed = time.perf_counter() - t0
    total_rows = sum(len(df) for df in stats.values())
    logger.info(
        "FBRef team stats scrape complete: %d stat types, %d total rows in %.1fs",
        len(stats), total_rows, elapsed,
    )
    return stats


def run_understat_scrape() -> pd.DataFrame:
    """Scrape all Understat match xG data and return the combined DataFrame."""
    logger.info("=== Stage: Understat Match xG Scraping ===")
    t0 = time.perf_counter()
    scraper = UnderstatScraper(save_raw=True)
    df = scraper.scrape_all()
    elapsed = time.perf_counter() - t0
    logger.info("Understat match xG scrape complete: %d matches in %.1fs", len(df), elapsed)
    return df


def run_understat_deep_scrape() -> pd.DataFrame:
    """Scrape Understat deep stats (PPDA, deep completions, xPTS)."""
    logger.info("=== Stage: Understat Deep Stats Scraping ===")
    t0 = time.perf_counter()
    scraper = UnderstatScraper(save_raw=True)
    df = scraper.scrape_all_deep_stats()
    elapsed = time.perf_counter() - t0
    logger.info(
        "Understat deep stats scrape complete: %d rows in %.1fs",
        len(df), elapsed,
    )
    return df


# ---------------------------------------------------------------------------
# Pipeline stages: LOADING from saved CSVs
# ---------------------------------------------------------------------------

def load_raw_fbref() -> pd.DataFrame:
    """Load all saved FBRef fixture CSVs from disk and concatenate them."""
    csv_files = sorted(FBREF_RAW_DIR.glob("fbref_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No FBRef CSVs found in {FBREF_RAW_DIR}. Run with --scrape first."
        )
    frames = [pd.read_csv(f, parse_dates=["date"]) for f in csv_files]
    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    logger.info("Loaded %d FBRef rows from %d CSV files", len(df), len(csv_files))
    return df


def load_raw_understat() -> pd.DataFrame:
    """Load all saved Understat match xG CSVs from disk and concatenate them."""
    csv_files = sorted(UNDERSTAT_RAW_DIR.glob("understat_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No Understat CSVs found in {UNDERSTAT_RAW_DIR}. Run with --scrape first."
        )
    frames = [pd.read_csv(f, parse_dates=["date"]) for f in csv_files]
    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    logger.info("Loaded %d Understat rows from %d CSV files", len(df), len(csv_files))
    return df


def load_raw_understat_deep() -> pd.DataFrame | None:
    """Load saved Understat deep stats CSVs. Returns None if not available."""
    csv_files = sorted(UNDERSTAT_DEEP_DIR.glob("understat_deep_*.csv"))
    if not csv_files:
        logger.info("No Understat deep stats CSVs found — deep stats will be skipped")
        return None
    frames = [pd.read_csv(f, parse_dates=["date"]) for f in csv_files]
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "team"]).reset_index(drop=True)
    logger.info("Loaded %d Understat deep stats rows from %d CSV files", len(df), len(csv_files))
    return df


# ---------------------------------------------------------------------------
# Pipeline stages: PROCESSING
# ---------------------------------------------------------------------------

def run_processing(
    fbref_df: pd.DataFrame,
    understat_df: pd.DataFrame,
    understat_deep_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the full processing pipeline: normalize -> engineer features."""
    logger.info("=== Stage: Schema Normalization ===")
    t0 = time.perf_counter()
    normalizer = SchemaNormalizer(save_parquet=True)
    merged = normalizer.normalize(fbref_df, understat_df, understat_deep_df)
    logger.info(
        "Normalization complete in %.1fs — %d rows, %d columns",
        time.perf_counter() - t0, len(merged), len(merged.columns),
    )

    logger.info("=== Stage: Feature Engineering ===")
    t0 = time.perf_counter()
    engineer = FeatureEngineer(save_parquet=True)
    features = engineer.engineer(merged)
    logger.info(
        "Feature engineering complete in %.1fs — %d rows x %d columns",
        time.perf_counter() - t0,
        len(features),
        len(features.columns),
    )
    return features


# ---------------------------------------------------------------------------
# Auto-mode decision engine
# ---------------------------------------------------------------------------

def _should_scrape(state: dict) -> bool:
    """Check if a scrape is due based on pipeline state."""
    last = state.get("last_scrape_date")
    if last is None:
        return True
    last_date = date.fromisoformat(last) if isinstance(last, str) else last
    days_since = (date.today() - last_date).days
    logger.info(
        "Days since last scrape: %d (threshold: %d)", days_since, SCRAPE_INTERVAL_DAYS
    )
    return days_since >= SCRAPE_INTERVAL_DAYS


def _should_train(state: dict, current_matches: int) -> bool:
    """
    Check if retraining is due based on:
    1. Number of new matches since last training
    2. Model decay (if recent predictions are tracked)
    """
    last_train_matches = state.get("last_train_matches", 0)
    new_matches = current_matches - last_train_matches
    logger.info(
        "Matches since last train: %d (threshold: %d)",
        new_matches, RETRAIN_MATCH_THRESHOLD,
    )

    # Condition 1: enough new matches
    if new_matches >= RETRAIN_MATCH_THRESHOLD:
        logger.info("Retrain triggered: %d new matches >= %d threshold",
                     new_matches, RETRAIN_MATCH_THRESHOLD)
        return True

    # Condition 2: model decay detected
    baseline_ll = state.get("last_train_log_loss")
    recent_preds = state.get("recent_predictions", [])
    if baseline_ll and len(recent_preds) >= 20:
        import numpy as np
        actuals_available = [p for p in recent_preds if p.get("actual")]
        if len(actuals_available) >= 20:
            recent_ll = _compute_recent_log_loss(actuals_available)
            decay_ratio = recent_ll / baseline_ll
            logger.info(
                "Model decay check: recent_LL=%.4f, baseline_LL=%.4f, ratio=%.3f",
                recent_ll, baseline_ll, decay_ratio,
            )
            if decay_ratio > DECAY_THRESHOLD:
                logger.info("Retrain triggered: decay ratio %.3f > %.3f threshold",
                             decay_ratio, DECAY_THRESHOLD)
                return True

    # Condition 3: never trained
    if state.get("model_version", 0) == 0:
        logger.info("Retrain triggered: no model has been trained yet")
        return True

    logger.info("No retrain needed — model still fresh")
    return False


def _compute_recent_log_loss(predictions: list[dict]) -> float:
    """Compute log-loss on recent predictions that have actuals."""
    import numpy as np
    eps = 1e-10
    total_ll = 0.0
    n = 0
    for p in predictions:
        probs = p.get("predicted", [1/3, 1/3, 1/3])
        actual = p.get("actual")
        if actual == "home":
            total_ll -= np.log(max(probs[0], eps))
        elif actual == "draw":
            total_ll -= np.log(max(probs[1], eps))
        elif actual == "away":
            total_ll -= np.log(max(probs[2], eps))
        else:
            continue
        n += 1
    return total_ll / max(n, 1)


def run_auto_mode(store: StorageBackend, tiers: list[int] | None = None) -> int:
    """
    Smart execution: check state and decide whether to scrape, train, or infer.

    Returns exit code:
        0 : Inference only (no state change)
        2 : New data scraped and processed
        3 : Full retraining completed
    """
    from models import MatchPredictor

    state = store.load_state()
    exit_code = 0
    features_path = PROCESSED_DIR / "features.parquet"

    # ------------------------------------------------------------------
    # Step 1: Decide whether to scrape
    # ------------------------------------------------------------------
    if _should_scrape(state):
        logger.info("=== AUTO: Scraping new data ===")
        fbref_df = run_fbref_scrape()
        run_fbref_team_stats_scrape()
        us_scraper = UnderstatScraper(save_raw=True)
        understat_df = us_scraper.scrape_all()
        understat_deep_df = us_scraper.scrape_all_deep_stats()
        features = run_processing(fbref_df, understat_df, understat_deep_df)

        state["last_scrape_date"] = date.today().isoformat()
        state["last_scrape_matches"] = len(features)
        store.save_state(state)
        exit_code = 2
        logger.info("Scrape + process complete: %d total matches", len(features))
    else:
        logger.info("=== AUTO: Scrape not due — using existing data ===")

    # ------------------------------------------------------------------
    # Step 2: Decide whether to retrain
    # ------------------------------------------------------------------
    if not features_path.exists():
        logger.error("features.parquet not found. Run --scrape or --process first.")
        return 1

    current_matches = len(pd.read_parquet(features_path))

    if _should_train(state, current_matches):
        logger.info("=== AUTO: Retraining models ===")
        predictor = MatchPredictor.from_parquet(features_path)
        predictor.fit_all(tiers=tiers)
        predictor.save_models(MODEL_OUTPUT_DIR)

        # Sync models to storage backend
        for pkl_file in MODEL_OUTPUT_DIR.glob("*.pkl"):
            store.upload_model(pkl_file.name, pkl_file)

        state["last_train_date"] = date.today().isoformat()
        state["last_train_matches"] = current_matches
        state["model_version"] = state.get("model_version", 0) + 1
        store.save_state(state)
        exit_code = 3
        logger.info(
            "Retrain complete: model v%d, trained on %d matches",
            state["model_version"], current_matches,
        )
    else:
        logger.info("=== AUTO: Using existing models (v%s) ===",
                     state.get("model_version", "?"))

    return exit_code


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def run_prediction(
    home: str, away: str,
    store: StorageBackend,
    tiers: list[int] | None = None,
) -> int:
    """
    Load saved models and predict a single match.

    Returns exit code 0 on success, 1 on error.
    """
    from models import MatchPredictor

    features_path = PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        logger.error("features.parquet not found. Run --scrape --process first.")
        return 1

    df = pd.read_parquet(features_path)

    # Try loading saved models from storage
    model_names = store.list_models()
    if model_names:
        # Download models from storage to local MODEL_OUTPUT_DIR
        for name in model_names:
            local_path = MODEL_OUTPUT_DIR / name
            if not local_path.exists():
                store.download_model(name, local_path)
        try:
            predictor = MatchPredictor.load_models(MODEL_OUTPUT_DIR, df)
            logger.info("Loaded %d saved models", len(predictor._models))
        except Exception as e:
            logger.warning("Failed to load saved models (%s) — training ...", e)
            predictor = MatchPredictor(df)
            predictor.fit_all(tiers=tiers)
            predictor.save_models(MODEL_OUTPUT_DIR)
    else:
        logger.info("No saved models found — training all tiers ...")
        predictor = MatchPredictor(df)
        predictor.fit_all(tiers=tiers)
        predictor.save_models(MODEL_OUTPUT_DIR)

    result = predictor.predict(home, away)
    logger.info("\n=== Prediction: %s vs %s ===", home, away)
    for name, probs in result["tiers"].items():
        logger.info(
            "  %-20s  H=%.1f%%  D=%.1f%%  A=%.1f%%",
            name, probs["home"] * 100, probs["draw"] * 100,
            probs["away"] * 100,
        )
    logger.info("  Most likely scoreline: %s", result["most_likely_scoreline"])
    eg = result["expected_goals"]
    logger.info("  Expected goals: %.2f - %.2f", eg["home"], eg["away"])
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Premier League match prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Primary execution modes (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--auto",
        action="store_true",
        help="Smart mode: check state, scrape/train if due, then infer.",
    )
    group.add_argument(
        "--force-train",
        action="store_true",
        help="Force immediate model retraining regardless of state.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Full pipeline: scrape all sources then process.",
    )
    group.add_argument(
        "--scrape",
        action="store_true",
        help="Scrape FBRef + Understat, save raw CSVs.",
    )
    group.add_argument(
        "--process",
        action="store_true",
        help="Load existing raw CSVs, merge, and engineer features.",
    )
    group.add_argument(
        "--fbref",
        action="store_true",
        help="Scrape FBRef only (fixtures + team stats).",
    )
    group.add_argument(
        "--understat",
        action="store_true",
        help="Scrape Understat only (match xG + deep stats).",
    )
    group.add_argument(
        "--train",
        action="store_true",
        help="Fit model tiers on features.parquet.",
    )
    group.add_argument(
        "--evaluate",
        action="store_true",
        help="Run temporal CV evaluation on all tiers.",
    )

    # Non-exclusive options
    parser.add_argument(
        "--predict",
        nargs=2,
        metavar=("HOME", "AWAY"),
        help="Predict a match (e.g. --predict Arsenal Chelsea).",
    )
    parser.add_argument(
        "--tiers",
        nargs="*",
        type=int,
        default=None,
        help="Which model tiers to train/evaluate (1-5). Default: all.",
    )

    # Storage backend options
    parser.add_argument(
        "--storage",
        choices=["local", "s3"],
        default="local",
        help="Persistence backend (default: local).",
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help="S3 bucket name (required if --storage s3).",
    )
    parser.add_argument(
        "--s3-prefix",
        default="",
        help="S3 key prefix (e.g. 'prod/v1').",
    )

    return parser


def _create_storage(args: argparse.Namespace) -> StorageBackend:
    """Create the storage backend from CLI arguments."""
    if args.storage == "s3":
        if not args.s3_bucket:
            raise ValueError("--s3-bucket is required when using --storage s3")
        return StorageManager.create(
            "s3", bucket=args.s3_bucket, prefix=args.s3_prefix
        )
    return StorageManager.create("local", base_dir=str(MODEL_OUTPUT_DIR))


def main() -> None:
    _configure_logging()
    args = _build_parser().parse_args()

    pipeline_start = time.perf_counter()
    logger.info("Premier League Prediction Pipeline — starting")
    logger.info(
        "Configured seasons: FBRef=%s | Understat=%s",
        FBREF_SEASONS,
        UNDERSTAT_YEARS,
    )

    # Require at least one action
    has_action = any([
        args.auto, args.force_train, args.all, args.scrape, args.process,
        args.fbref, args.understat, args.train, args.evaluate, args.predict,
    ])
    if not has_action:
        _build_parser().print_help()
        sys.exit(1)

    exit_code = 0

    try:
        store = _create_storage(args)

        # ==================================================================
        # AUTO MODE — smart scrape/train/infer
        # ==================================================================
        if args.auto:
            exit_code = run_auto_mode(store, tiers=args.tiers)

        # ==================================================================
        # FORCE TRAIN — retrain immediately
        # ==================================================================
        elif args.force_train:
            from models import MatchPredictor
            features_path = PROCESSED_DIR / "features.parquet"
            if not features_path.exists():
                logger.error("features.parquet not found. Run --scrape --process first.")
                sys.exit(1)

            predictor = MatchPredictor.from_parquet(features_path)
            predictor.fit_all(tiers=args.tiers)
            predictor.save_models(MODEL_OUTPUT_DIR)

            # Sync to storage
            for pkl_file in MODEL_OUTPUT_DIR.glob("*.pkl"):
                store.upload_model(pkl_file.name, pkl_file)

            # Update state
            state = store.load_state()
            current_matches = len(pd.read_parquet(features_path))
            state["last_train_date"] = date.today().isoformat()
            state["last_train_matches"] = current_matches
            state["model_version"] = state.get("model_version", 0) + 1
            store.save_state(state)
            exit_code = 3

        # ==================================================================
        # Legacy modes
        # ==================================================================
        elif args.fbref:
            run_fbref_scrape()
            run_fbref_team_stats_scrape()

        elif args.understat:
            scraper = UnderstatScraper(save_raw=True)
            logger.info("=== Stage: Understat Match xG Scraping ===")
            t0 = time.perf_counter()
            understat_df = scraper.scrape_all()
            logger.info("Understat match xG: %d matches in %.1fs",
                         len(understat_df), time.perf_counter() - t0)
            logger.info("=== Stage: Understat Deep Stats ===")
            t0 = time.perf_counter()
            deep_df = scraper.scrape_all_deep_stats()
            logger.info("Understat deep stats: %d rows in %.1fs",
                         len(deep_df), time.perf_counter() - t0)

        elif args.scrape:
            run_fbref_scrape()
            run_fbref_team_stats_scrape()
            scraper = UnderstatScraper(save_raw=True)
            logger.info("=== Stage: Understat Match xG Scraping ===")
            t0_us = time.perf_counter()
            scraper.scrape_all()
            logger.info("Understat match xG complete in %.1fs",
                         time.perf_counter() - t0_us)
            logger.info("=== Stage: Understat Deep Stats ===")
            t0_us = time.perf_counter()
            scraper.scrape_all_deep_stats()
            logger.info("Understat deep stats complete in %.1fs",
                         time.perf_counter() - t0_us)

        elif args.process:
            fbref_df = load_raw_fbref()
            understat_df = load_raw_understat()
            understat_deep_df = load_raw_understat_deep()
            run_processing(fbref_df, understat_df, understat_deep_df)

        elif args.all:
            fbref_df = run_fbref_scrape()
            run_fbref_team_stats_scrape()
            us_scraper = UnderstatScraper(save_raw=True)
            logger.info("=== Stage: Understat Scraping (match xG + deep stats) ===")
            t0_us = time.perf_counter()
            understat_df = us_scraper.scrape_all()
            understat_deep_df = us_scraper.scrape_all_deep_stats()
            logger.info(
                "Understat scraping complete: %d matches + %d deep rows in %.1fs",
                len(understat_df), len(understat_deep_df),
                time.perf_counter() - t0_us,
            )
            run_processing(fbref_df, understat_df, understat_deep_df)

        elif args.train:
            from models import MatchPredictor
            features_path = PROCESSED_DIR / "features.parquet"
            predictor = MatchPredictor.from_parquet(features_path)
            predictor.fit_all(tiers=args.tiers)
            predictor.save_models(MODEL_OUTPUT_DIR)

            # Update state
            state = store.load_state()
            current_matches = len(pd.read_parquet(features_path))
            state["last_train_date"] = date.today().isoformat()
            state["last_train_matches"] = current_matches
            state["model_version"] = state.get("model_version", 0) + 1
            store.save_state(state)
            exit_code = 3

        elif args.evaluate:
            from models import MatchPredictor
            features_path = PROCESSED_DIR / "features.parquet"
            predictor = MatchPredictor.from_parquet(features_path)
            predictor.fit_all(tiers=args.tiers)
            results = predictor.evaluate()
            logger.info("\n=== Model Comparison ===\n%s", results.to_string())

        # ------------------------------------------------------------------
        # Handle --predict (can be combined with other flags)
        # ------------------------------------------------------------------
        if args.predict:
            home, away = args.predict
            predict_code = run_prediction(home, away, store, tiers=args.tiers)
            if predict_code != 0:
                exit_code = predict_code

        total = time.perf_counter() - pipeline_start
        logger.info("Pipeline finished successfully in %.1fs (exit code %d)", total, exit_code)
        sys.exit(exit_code)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Pipeline failed with unhandled exception: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
