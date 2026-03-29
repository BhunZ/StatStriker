"""
smoke_test.py
-------------
Minimal sanity check for the full pipeline: load data → train → predict → save/load parity.
Runs in ~3 minutes (no evaluation CV). Use after any code change to verify nothing is broken.

Usage:
    python smoke_test.py
"""

import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("smoke_test")

FEATURES_PATH = Path("data/processed/features.parquet")
MODEL_DIR = Path("data/models")
PASS = "[PASS]"
FAIL = "[FAIL]"


def check(name: str, condition: bool, detail: str = ""):
    """Print a pass/fail check."""
    status = PASS if condition else FAIL
    msg = f"  {status}  {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if not condition:
        logger.error("FAILED: %s %s", name, detail)
    return condition


def main():
    t0 = time.time()
    all_ok = True
    print("=" * 60)
    print("SMOKE TEST — Premier League Match Prediction Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Data loading
    # ------------------------------------------------------------------
    print("\n[1/6] Data Loading")
    if not FEATURES_PATH.exists():
        print(f"  {FAIL}  features.parquet not found at {FEATURES_PATH}")
        print("  Run: python main.py --process")
        sys.exit(1)

    df = pd.read_parquet(FEATURES_PATH)
    all_ok &= check("Loaded features.parquet", len(df) > 0, f"{len(df)} matches")
    all_ok &= check("Required columns exist",
                     all(c in df.columns for c in ["home_team", "away_team", "home_goals", "away_goals", "date"]))
    all_ok &= check("No null goals", df[["home_goals", "away_goals"]].notna().all().all())
    all_ok &= check("Home goals are integers", (df["home_goals"] == df["home_goals"].astype(int)).all())

    # Check xG coverage
    xg_coverage = df["home_xg"].notna().mean() * 100
    all_ok &= check("xG coverage", xg_coverage > 50, f"{xg_coverage:.0f}% of matches have xG")

    # Check feature columns exist
    from models.config_models import FP_FEATURE_COLS
    available_feats = [c for c in FP_FEATURE_COLS if c in df.columns]
    all_ok &= check("Feature columns available", len(available_feats) >= 10,
                     f"{len(available_feats)}/{len(FP_FEATURE_COLS)} features present")

    # ------------------------------------------------------------------
    # 2. Model training (all 5 tiers)
    # ------------------------------------------------------------------
    print("\n[2/6] Model Training")
    from models.predict import MatchPredictor

    predictor = MatchPredictor.from_parquet(FEATURES_PATH)
    predictor.fit_all()
    all_ok &= check("All 5 tiers trained", len(predictor._models) == 5,
                     f"got {len(predictor._models)} models")

    # ------------------------------------------------------------------
    # 3. Prediction sanity checks
    # ------------------------------------------------------------------
    print("\n[3/6] Prediction Sanity Checks")

    result = predictor.predict("Arsenal", "Chelsea")
    tiers = result["tiers"]

    for tier_name, probs in tiers.items():
        h, d, a = probs["home"], probs["draw"], probs["away"]
        total = h + d + a
        all_ok &= check(f"{tier_name} probabilities sum to 1",
                         abs(total - 1.0) < 0.01, f"sum={total:.4f}")
        all_ok &= check(f"{tier_name} no negative probs",
                         h >= 0 and d >= 0 and a >= 0)

    # Arsenal at home vs Chelsea should favor Arsenal
    ars_h = tiers["dixon_coles"]["home"]
    all_ok &= check("Arsenal home > 40% (sanity)",
                     ars_h > 0.40, f"got {ars_h:.1%}")

    # Scoreline matrix check
    mat = result["scoreline_matrix"]
    all_ok &= check("Scoreline matrix shape", mat.shape == (9, 9), f"shape={mat.shape}")
    all_ok &= check("Scoreline matrix sums to ~1",
                     abs(mat.sum() - 1.0) < 0.01, f"sum={mat.sum():.4f}")

    # ------------------------------------------------------------------
    # 4. Cold-start handling
    # ------------------------------------------------------------------
    print("\n[4/6] Cold-Start Handling")
    try:
        cold_result = predictor.predict("FakePromotedFC", "Arsenal")
        cold_tiers = cold_result["tiers"]
        for name, probs in cold_tiers.items():
            total = probs["home"] + probs["draw"] + probs["away"]
            all_ok &= check(f"{name} cold-start valid probs",
                             abs(total - 1.0) < 0.01 and probs["home"] >= 0)
        all_ok &= check("Cold-start did not crash", True)
    except Exception as e:
        all_ok &= check("Cold-start did not crash", False, str(e))

    # ------------------------------------------------------------------
    # 5. Save / Load parity
    # ------------------------------------------------------------------
    print("\n[5/6] Save/Load Parity")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "models"
        predictor.save_models(save_dir)

        saved_files = list(save_dir.glob("*.pkl"))
        all_ok &= check("Model files saved", len(saved_files) >= 5,
                         f"{len(saved_files)} .pkl files")

        # Reload and compare predictions
        predictor2 = MatchPredictor.load_models(save_dir, df)

        result2 = predictor2.predict("Arsenal", "Chelsea")
        for tier_name in tiers:
            if tier_name not in result2["tiers"]:
                all_ok &= check(f"{tier_name} loaded", False, "missing after reload")
                continue
            orig = tiers[tier_name]
            loaded = result2["tiers"][tier_name]
            diff = max(
                abs(orig["home"] - loaded["home"]),
                abs(orig["draw"] - loaded["draw"]),
                abs(orig["away"] - loaded["away"]),
            )
            all_ok &= check(f"{tier_name} save/load parity",
                             diff < 1e-10, f"max diff={diff:.2e}")

    # ------------------------------------------------------------------
    # 6. Ensemble weight sanity
    # ------------------------------------------------------------------
    print("\n[6/6] Ensemble Weight Sanity")
    ensemble = predictor._models.get("ensemble")
    if ensemble is not None:
        weights = ensemble._weights
        all_ok &= check("Ensemble weights sum to 1",
                         abs(weights.sum() - 1.0) < 1e-6, f"sum={weights.sum():.6f}")
        all_ok &= check("All weights positive",
                         (weights > 0).all(),
                         ", ".join(f"{m.name}={w:.3f}" for m, w in zip(ensemble.base_models_, weights)))
        all_ok &= check("No single model dominates (>95%)",
                         weights.max() < 0.95, f"max weight={weights.max():.3f}")
    else:
        all_ok &= check("Ensemble exists", False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    if all_ok:
        print(f"ALL CHECKS PASSED in {elapsed:.0f}s")
    else:
        print(f"SOME CHECKS FAILED — review output above ({elapsed:.0f}s)")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
