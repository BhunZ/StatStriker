# Premier League Match Prediction — CLAUDE.md

## Project Overview
Production-grade soccer match prediction app: FBRef + Understat scraping →
Dixon-Coles Poisson model → FastAPI/Streamlit web interface.

## Architecture
- `scrapers/`        — Data fetching (FBRef, Understat) via ScraperAPI
- `processors/`      — Entity resolution, schema normalization, feature engineering
- `models/`          — Dixon-Coles model (Phase 2)
- `app/`             — FastAPI backend + Streamlit UI (Phase 3)
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
pip install -r requirements.txt
cp .env.example .env          # Add your SCRAPER_API_KEY
python main.py --all           # Full scrape + process pipeline
python main.py --scrape        # Scrape only (saves raw CSVs)
python main.py --process       # Process existing raw data (merge + features)
```

## Phase Roadmap
- Phase 1 (current): Scraper + data pipeline
- Phase 2: Dixon-Coles Poisson model with rho (draw inflation) and xi (time decay)
- Phase 3: FastAPI + Streamlit web dashboard with 1X2 probs and xG heatmaps
