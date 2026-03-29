"""
app/api.py
----------
FastAPI backend for the Premier League match prediction API.

Endpoints
---------
    GET /                           Health check
    GET /teams                      List all available teams
    GET /predict/{home}/{away}      Predict match outcome (all tiers)
    GET /predict/{home}/{away}?tier=ensemble   Single-tier prediction
    GET /teams/{team}/form          Recent form & rolling stats
    GET /status                     Pipeline state (last scrape, model version, etc.)
    GET /scoreline/{home}/{away}    Scoreline probability matrix (9x9)

Run
---
    uvicorn app.api:app --reload
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Project imports ──────────────────────────────────────────────────
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.predict import MatchPredictor
from processors.entity_resolver import EntityResolver

logger = logging.getLogger(__name__)

# ── Global state (loaded once at startup) ────────────────────────────
_predictor: MatchPredictor | None = None
_resolver: EntityResolver | None = None
_teams: list[str] = []
_pipeline_state: dict = {}
_df: pd.DataFrame | None = None

FEATURES_PATH = ROOT_DIR / "data" / "processed" / "features.parquet"
MODELS_DIR = ROOT_DIR / "data" / "models"
STATE_PATH = MODELS_DIR / "pipeline_state.json"


def _load_models():
    """Load trained models and feature data into memory."""
    global _predictor, _resolver, _teams, _pipeline_state, _df

    # Load features
    if not FEATURES_PATH.exists():
        logger.error("features.parquet not found at %s", FEATURES_PATH)
        raise FileNotFoundError(
            f"features.parquet not found at {FEATURES_PATH}. "
            "Run: python main.py --process"
        )

    _df = pd.read_parquet(FEATURES_PATH)
    _df["date"] = pd.to_datetime(_df["date"])
    logger.info("Loaded %d matches from features.parquet", len(_df))

    # Load models
    pkl_files = list(MODELS_DIR.glob("*.pkl"))
    if not pkl_files:
        logger.error("No .pkl model files found in %s", MODELS_DIR)
        raise FileNotFoundError(
            f"No trained models in {MODELS_DIR}. "
            "Run: python main.py --train"
        )

    _predictor = MatchPredictor.load_models(MODELS_DIR, _df)
    logger.info("Loaded %d models", len(_predictor._models))

    # Entity resolver for fuzzy team name matching
    _resolver = EntityResolver()

    # Available teams (from training data)
    _teams = sorted(
        set(_df["home_team"].unique()) | set(_df["away_team"].unique())
    )

    # Pipeline state
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            _pipeline_state = json.load(f)
    else:
        _pipeline_state = {}


# ── Lifespan (startup/shutdown) ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models into memory on startup."""
    _load_models()
    logger.info("API ready — %d teams, %d models", len(_teams), len(_predictor._models))
    yield
    logger.info("API shutting down")


# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="StatStriker API",
    description="Premier League match prediction using 5-tier Poisson models",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────

def _resolve_team(name: str) -> str:
    """Resolve a user-supplied team name to canonical form."""
    try:
        return _resolver.resolve(name)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Team not found: '{name}'. Use GET /teams for valid names.",
        )


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "teams": len(_teams),
        "models": list(_predictor._models.keys()) if _predictor else [],
        "matches_trained_on": len(_df) if _df is not None else 0,
    }


@app.get("/teams")
async def list_teams():
    """List all available teams."""
    return {"teams": _teams, "count": len(_teams)}


@app.get("/predict/{home_team}/{away_team}")
async def predict_match(
    home_team: str,
    away_team: str,
    tier: str | None = Query(default=None, description="Model tier name (e.g. 'ensemble', 'dixon_coles')"),
):
    """
    Predict match outcome probabilities.

    Returns home/draw/away probabilities from all tiers (or a specific tier),
    plus the most likely scoreline and expected goals.
    """
    home = _resolve_team(home_team)
    away = _resolve_team(away_team)

    if home == away:
        raise HTTPException(status_code=400, detail="Home and away teams must be different.")

    try:
        result = _predictor.predict(home, away, tier=tier)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "home_team": home,
        "away_team": away,
        "tiers": result["tiers"],
        "most_likely_scoreline": result["most_likely_scoreline"],
        "expected_goals": {
            "home": round(result["expected_goals"]["home"], 2),
            "away": round(result["expected_goals"]["away"], 2),
        },
    }


@app.get("/scoreline/{home_team}/{away_team}")
async def scoreline_matrix(home_team: str, away_team: str):
    """
    Return the full 9x9 scoreline probability matrix.

    matrix[i][j] = P(home scores i, away scores j).
    """
    home = _resolve_team(home_team)
    away = _resolve_team(away_team)

    if home == away:
        raise HTTPException(status_code=400, detail="Home and away teams must be different.")

    result = _predictor.predict(home, away)
    mat = result["scoreline_matrix"]

    # Convert numpy matrix to list of lists with rounded values
    matrix_list = [[round(float(mat[i, j]), 4) for j in range(mat.shape[1])] for i in range(mat.shape[0])]

    # Top 5 most likely scorelines
    flat_indices = np.argsort(mat.ravel())[::-1][:5]
    top_scorelines = []
    for idx in flat_indices:
        i, j = divmod(idx, mat.shape[1])
        top_scorelines.append({
            "score": f"{i}-{j}",
            "probability": round(float(mat[i, j]), 4),
        })

    return {
        "home_team": home,
        "away_team": away,
        "matrix": matrix_list,
        "goals_range": list(range(mat.shape[0])),
        "top_scorelines": top_scorelines,
    }


@app.get("/teams/{team_name}/form")
async def team_form(team_name: str, last_n: int = Query(default=5, ge=1, le=20)):
    """
    Get a team's recent form: last N matches with results and rolling stats.
    """
    team = _resolve_team(team_name)

    # Get matches where this team played (home or away)
    home_mask = _df["home_team"] == team
    away_mask = _df["away_team"] == team
    matches = _df[home_mask | away_mask].sort_values("date", ascending=False).head(last_n)

    if matches.empty:
        raise HTTPException(status_code=404, detail=f"No matches found for '{team}'.")

    form = []
    for _, row in matches.iterrows():
        is_home = row["home_team"] == team
        goals_for = int(row["home_goals"]) if is_home else int(row["away_goals"])
        goals_against = int(row["away_goals"]) if is_home else int(row["home_goals"])

        if goals_for > goals_against:
            result = "W"
        elif goals_for == goals_against:
            result = "D"
        else:
            result = "L"

        match_data = {
            "date": row["date"].strftime("%Y-%m-%d"),
            "opponent": row["away_team"] if is_home else row["home_team"],
            "venue": "home" if is_home else "away",
            "goals_for": goals_for,
            "goals_against": goals_against,
            "result": result,
        }

        # Add xG if available
        if pd.notna(row.get("home_xg")) and pd.notna(row.get("away_xg")):
            match_data["xg_for"] = round(float(row["home_xg"] if is_home else row["away_xg"]), 2)
            match_data["xg_against"] = round(float(row["away_xg"] if is_home else row["home_xg"]), 2)

        form.append(match_data)

    # Summary stats
    results = [m["result"] for m in form]
    return {
        "team": team,
        "last_n": len(form),
        "summary": {
            "wins": results.count("W"),
            "draws": results.count("D"),
            "losses": results.count("L"),
            "goals_scored": sum(m["goals_for"] for m in form),
            "goals_conceded": sum(m["goals_against"] for m in form),
        },
        "matches": form,
    }


@app.get("/status")
async def pipeline_status():
    """Return pipeline state: last scrape/train dates, model version, etc."""
    models_info = {}
    if _predictor:
        for name, model in _predictor._models.items():
            models_info[name] = {
                "type": type(model).__name__,
            }

    return {
        "pipeline": _pipeline_state,
        "models": models_info,
        "data": {
            "total_matches": len(_df) if _df is not None else 0,
            "teams": len(_teams),
            "seasons": sorted(_df["season"].unique().tolist()) if _df is not None else [],
        },
    }
