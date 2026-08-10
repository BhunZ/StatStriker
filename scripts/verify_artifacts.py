"""Prove the committed model files can be loaded by something that did not train them.

This exists because they once could not. `data/models/*.pkl` were fitted on a laptop and
committed; the deployed API unpickled them on a different scikit-learn build and got
`ModuleNotFoundError: No module named '_loss'` from inside the library. Two of the four
files failed, `load_models` let the first failure propagate, and the site served an empty
team list and a 503 on every prediction. Nothing in CI had any opinion about it — the
pipeline had gone green and pushed the artifacts itself.

Training and checking in the same process proves nothing: the objects are already in
memory and never travel through pickle. This runs as its own process, reads only what is
on disk, and is the last thing to pass before the workflow commits.

Exit codes: 0 every model loaded and predicted · 1 something did not.
"""

from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from models.predict import ENSEMBLE_TIERS, MatchPredictor, _TIER_FACTORIES  # noqa: E402

MODELS_DIR = ROOT / "data" / "models"
FEATURES = ROOT / "data" / "processed" / "features.parquet"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    print(f"python {platform.python_version()} · scikit-learn {sklearn.__version__} · "
          f"numpy {np.__version__} · pandas {pd.__version__}")

    if not FEATURES.exists():
        print(f"FAIL: {FEATURES} is missing — nothing to verify against")
        return 1

    pkl_files = sorted(MODELS_DIR.glob("*.pkl"))
    if not pkl_files:
        print(f"FAIL: no .pkl files in {MODELS_DIR}")
        return 1

    df = pd.read_parquet(FEATURES)
    predictor = MatchPredictor.load_models(MODELS_DIR, df)

    failures = dict(predictor.load_failures_)
    loaded = set(predictor._models)
    expected = {_TIER_FACTORIES[t]().name for t in ENSEMBLE_TIERS} | {"ensemble"}

    for pkl in pkl_files:
        why = failures.get(pkl.name)
        print(f"  {'FAIL' if why else 'ok  '}  {pkl.name:24s} {why or ''}")

    problems: list[str] = []
    if failures:
        problems.append(f"{len(failures)} file(s) could not be unpickled")

    missing = expected - loaded
    if missing:
        problems.append(f"missing model(s) the ensemble needs: {sorted(missing)}")

    # Loading is not the same as working. A model can unpickle and then fail on the first
    # prediction because an attribute it needs was never saved.
    if not missing:
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        if len(teams) < 2:
            problems.append("features.parquet has fewer than two teams")
        else:
            for name, model in sorted(predictor._models.items()):
                try:
                    probs = model.predict_1x2(teams[0], teams[1])
                    total = sum(probs.values())
                    if not np.isfinite(total) or abs(total - 1.0) > 1e-6:
                        problems.append(f"{name}: probabilities sum to {total!r}")
                except Exception as exc:
                    problems.append(f"{name}: predicting raised {type(exc).__name__}: {exc}")

    if problems:
        print("\nFAIL — these artifacts must not be committed:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"\nOK — {len(loaded)} models loaded and predicted: {sorted(loaded)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
