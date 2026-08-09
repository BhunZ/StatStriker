# Premier League Match Prediction — CLAUDE.md

## Project Overview
Premier League match prediction: FBRef + Understat scraping → Poisson family models
(Dixon-Coles, bivariate, xG-blended, feature GLM) blended by a stacked ensemble →
FastAPI backend serving a static HTML dashboard. The whole pipeline runs unattended
every Monday via GitHub Actions.

## Architecture
- `scrapers/`        — Data fetching (FBRef, Understat) via ScraperAPI
- `processors/`      — Entity resolution, schema normalization, feature engineering
- `models/`          — Dixon-Coles model (Phase 2)
- `app/`             — FastAPI backend
- `frontend/`        — static HTML dashboard (no build step, no Streamlit)
- `quality.py`       — data quality gate; runs between processing and training
- `tests/`           — pytest suite; runs in CI before the pipeline
- `data/raw/`        — Raw per-source CSVs (never edit manually)
- `data/processed/`  — Merged/feature-engineered Parquet files
- `logs/`            — Rotating log files from pipeline runs

## Key Conventions
- API key lives in `.env` — never hardcode, never commit `.env`
- All scrapers inherit `BaseScraper`; implement `scrape_season()` and `scrape_all()`
- Team names must always be resolved to canonical form via `EntityResolver` before any merge
- FBRef = ground truth for goals scored; Understat = ground truth for xG
- Logging: use module-level `logger = logging.getLogger(__name__)` — no bare `print()`
- Vectorized Pandas ops only — no row-wise loops on DataFrames
- Save intermediate outputs: raw CSVs after scraping, Parquet after processing

## Data Schema (after merge)
Key columns in `data/processed/merged.parquet`:
- `date` (pd.Timestamp), `season` (str), `home_team`, `away_team` (canonical names)
- `home_goals`, `away_goals` (int, from FBRef)
- `home_xg`, `away_xg` (float, from Understat, fallback to FBRef)
- `home_shots`, `away_shots`, `home_poss`, `away_poss` (from FBRef)

## Running the Pipeline
```bash
pip install -r requirements-dev.txt   # requirements.txt + pytest
cp .env.example .env                  # Add your SCRAPER_API_KEY
pytest -q                             # ~4s; run before touching anything
python main.py --auto                 # what the weekly job runs
python main.py --scrape               # Scrape only (saves raw CSVs)
python main.py --process              # Process existing raw data (merge + features)
```

## Exit codes (the workflow reads these)
- `0` nothing changed · `2` new data · `3` retrained — all fine, run stays green
- `4` the data quality gate rejected the scrape — nothing trained, nothing committed
- `1` crash

## Things that are easy to break
- **Never remove the `.shift(1)` in `feature_engineer.py`.** Rolling features would then
  contain the match they describe, every metric would improve, and nothing would look
  wrong. `tests/test_feature_engineer.py` is the guard.
- **Team names must resolve through `EntityResolver` before any merge.** A wrong resolve
  gives one club another club's history with the row count still correct.
- **Keep module-level work out of scripts.** `scripts/verify_task3b_fix.py` once ran its
  whole body at import and cost pytest 28 seconds per collection.
- **The ensemble weights are not identifiable.** The four base models correlate 0.97–0.998,
  so the log-loss surface is nearly flat and repeated fits land on different corners.
  Do not read the weights as a ranking of model quality.
