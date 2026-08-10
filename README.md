<p align="center">
  <img src="frontend/realistic-ball.png" alt="StatStriker" width="120" />
</p>

<h1 align="center">StatStriker</h1>

<p align="center">
  <strong>Premier League Match Prediction Engine</strong><br>
  Scraping &rarr; Feature Engineering &rarr; Stacked Match Models &rarr; Web Dashboard
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License" />
  <img src="https://img.shields.io/github/actions/workflow/status/BhunZ/StatStriker/pipeline.yml?label=weekly%20pipeline&logo=github" alt="Pipeline Status" />
</p>

---

## Overview

StatStriker is a production-grade soccer match prediction system that scrapes real match data from FBRef and Understat, engineers predictive features, blends three complementary models into a stacked ensemble, and serves predictions through an interactive web dashboard.

The entire pipeline — from data collection to model retraining — runs automatically every week via GitHub Actions.

## Features

- **Live Match Predictions** — Select any two Premier League teams and get win/draw/loss probabilities with scoreline distributions
- **Three-Model Ensemble** — Dixon-Coles, a feature Poisson GLM and a gradient-boosted classifier, combined via learned stacking weights
- **Team Analytics** — Season stats, goal difference charts, xG trends, home/away splits, and form streaks
- **Upcoming Fixtures** — Live fixture data from the Fantasy Premier League API with one-click predictions
- **Automated Pipeline** — Weekly scraping, processing, and retraining via GitHub Actions with smart change detection

## Model Architecture

Five base models are implemented. **Three of them are blended** — an ensemble is only worth its cost when its members disagree, and the other two did not.

| Tier | Model | In blend | Description |
|------|-------|----------|-------------|
| 1 | **Dixon-Coles** | yes | Independent Poisson with rho correction for low-score draws and exponential time decay (Dixon & Coles, 1997) |
| 2 | **Bivariate Poisson** | no | Shared latent variable lambda_3 captures goal correlation across all scorelines (Karlis & Ntzoufras, 2003) |
| 3 | **xG Dixon-Coles** | no | Fits on a grid-searched blend of xG and actual goals for smoother, lower-variance attack/defense parameters |
| 4 | **Feature Poisson GLM** | yes | Adds rolling form features (PPDA, xG overperformance, prior-season stats) as log-linear covariates with L2 regularization |
| 5 | **Gradient Boost** | yes | Discriminative classifier over rolling form features — the one tier that never models goals, mapping form straight to 1X2 and calibrated by shrinkage towards the base rate |
| 6 | **Stacked Ensemble** | — | Softmax-weighted convex combination of Tiers 1, 4 and 5, trained on out-of-fold temporal CV predictions |

All models are evaluated using **Ranked Probability Score (RPS)** via expanding-window temporal cross-validation — no data leakage, no shuffling.

Tiers 2 and 3 are kept and can still be fitted or compared with `--tiers`, but out of fold they track Dixon-Coles closely enough to be the same model written twice. Every subset of the five was scored on the same out-of-fold matches before the two were dropped, and dropping them cost nothing measurable.

## Project Structure

```
StatStriker/
├── scrapers/               # Data collection
│   ├── base_scraper.py     # Abstract base with rate limiting & retry logic
│   ├── fbref_scraper.py    # FBRef: goals, shots, possession, xG
│   └── understat_scraper.py # Understat: match xG, PPDA, deep stats
├── processors/             # Data processing
│   ├── entity_resolver.py  # Fuzzy team name matching (rapidfuzz)
│   ├── schema_normalizer.py # Cross-source schema alignment
│   └── feature_engineer.py # Rolling features, form metrics
├── models/                 # Prediction models
│   ├── dixon_coles.py      # Tier 1: Classic Dixon-Coles
│   ├── bivariate_poisson.py # Tier 2: Bivariate Poisson
│   ├── xg_dixon_coles.py   # Tier 3: xG-augmented
│   ├── feature_poisson.py  # Tier 4: Feature GLM
│   ├── gradient_boost.py   # Tier 5: Gradient-boosted classifier
│   ├── ensemble.py         # Tier 6: Stacked ensemble
│   ├── evaluation.py       # Temporal CV + RPS scoring
│   └── predict.py          # Unified prediction interface
├── app/
│   └── api.py              # FastAPI backend (serves frontend + API)
├── frontend/
│   └── index.html          # Single-page dashboard (vanilla JS + GSAP)
├── data/
│   ├── raw/                # Per-source CSVs (gitignored)
│   ├── processed/          # Merged Parquet files
│   └── models/             # Trained model artifacts (.pkl)
├── main.py                 # CLI entry point
├── render.yaml             # Render deployment config
└── .github/workflows/
    └── pipeline.yml        # Weekly automated pipeline
```

## Quick Start

### Prerequisites

- Python 3.12+
- A [ScraperAPI](https://www.scraperapi.com/) key (for data collection only)

### Installation

```bash
git clone https://github.com/BhunZ/StatStriker.git
cd StatStriker
pip install -r requirements.txt
```

### Run the API locally

```bash
uvicorn app.api:app --reload
```

Open `http://localhost:8000` in your browser — the dashboard is served directly by FastAPI.

### Run the full pipeline

```bash
cp .env.example .env        # Add your SCRAPER_API_KEY
python main.py --auto        # Smart mode: scrape, process, train as needed
```

### CLI Options

```
python main.py --auto              # Smart: only scrape/train if data is stale
python main.py --scrape            # Scrape only (saves raw CSVs)
python main.py --process           # Process existing raw data
python main.py --force-train       # Force model retraining
python main.py --evaluate          # Full temporal cross-validation
python main.py --predict Arsenal Chelsea  # Quick prediction
```

## Data Sources

| Source | What | How |
|--------|------|-----|
| **FBRef** | Goals, shots, possession, xG | HTML scraping via ScraperAPI |
| **Understat** | Match xG, PPDA, deep stats | Internal JSON API |
| **FPL API** | Upcoming fixtures | Public REST API (no key needed) |

FBRef is the ground truth for goals scored. Understat is the ground truth for xG. Team names from all sources are resolved to canonical form via `EntityResolver` using a hand-curated alias map with rapidfuzz fallback.

## Deployment

StatStriker is configured for one-click deployment on **Render** (free tier):

1. Fork/push to GitHub
2. Connect repo on [render.com](https://render.com)
3. Render auto-detects `render.yaml` — click Deploy

The frontend and API run as a single service. The API URL is auto-detected (localhost for dev, deployed origin in production).

## Automation

The GitHub Actions workflow (`.github/workflows/pipeline.yml`) runs every Monday at 06:00 UTC:

1. Scrapes latest match data from FBRef + Understat
2. Processes and merges new data
3. Retrains models if new matches are found
4. Commits updated artifacts back to the repo

Exit codes enable smart CI/CD: `0` = no changes, `2` = new data, `3` = retrained.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Scraping | requests, BeautifulSoup, ScraperAPI |
| Data | pandas, PyArrow (Parquet) |
| Modeling | scipy (L-BFGS-B optimization), NumPy, scikit-learn |
| Matching | rapidfuzz |
| API | FastAPI, uvicorn, httpx |
| Frontend | Vanilla JS, GSAP, CSS Glass UI |
| CI/CD | GitHub Actions |
| Deployment | Render |

## License

MIT

---

<p align="center">
  Built with data, math, and too much coffee.
</p>
