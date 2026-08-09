"""Shared fixtures.

Everything here is synthetic and tiny. The point is that the suite runs in seconds and
gives the same answer every time — the existing `smoke_test.py` refits every model, takes
about seven minutes, and depends on whatever happens to be in data/processed, so it can
only ever be run by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def matches() -> pd.DataFrame:
    """Twelve matches between four teams, played in a fixed order.

    Goals are deliberately distinctive per team so a rolling feature can be checked by
    hand: Alpha always scores 3 at home, Bravo always 1, and so on.
    """
    rows = []
    teams = ["Alpha FC", "Bravo United", "Charlie City", "Delta Rovers"]
    goals = {"Alpha FC": 3, "Bravo United": 1, "Charlie City": 2, "Delta Rovers": 0}
    date = pd.Timestamp("2024-08-01")
    for round_ in range(3):
        for h, a in [(0, 1), (2, 3)]:
            rows.append({
                "date": date,
                "season": "2024-2025",
                "home_team": teams[h],
                "away_team": teams[a],
                "home_goals": goals[teams[h]],
                "away_goals": goals[teams[a]],
                "home_xg": goals[teams[h]] - 0.5,
                "away_xg": goals[teams[a]] + 0.5,
            })
            date += pd.Timedelta(days=7)
        for h, a in [(1, 0), (3, 2)]:
            rows.append({
                "date": date,
                "season": "2024-2025",
                "home_team": teams[h],
                "away_team": teams[a],
                "home_goals": goals[teams[h]],
                "away_goals": goals[teams[a]],
                "home_xg": goals[teams[h]] - 0.5,
                "away_xg": goals[teams[a]] + 0.5,
            })
            date += pd.Timedelta(days=7)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


@pytest.fixture
def outcomes() -> tuple[np.ndarray, np.ndarray]:
    """(y_true one-hot, y_pred) for six matches — three home wins, two draws, one away."""
    y_true = np.array([
        [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [0, 1, 0], [0, 1, 0],
        [0, 0, 1],
    ], dtype=float)
    y_pred = np.array([
        [0.60, 0.25, 0.15], [0.50, 0.30, 0.20], [0.70, 0.20, 0.10],
        [0.30, 0.45, 0.25], [0.35, 0.40, 0.25],
        [0.20, 0.30, 0.50],
    ])
    return y_true, y_pred
