"""
scripts/ab_backtest_fp_v1_vs_v2.py
---------------------------------
Walk-forward A/B backtest for the FeaturePoisson architecture change.

Compares:
  v1 — symmetric (pre-v2.0 behavior): one shared weight vector applied
       to both sides of the match. Reproduced here by instantiating
       FeaturePoisson with ``FP_FEATURE_COLS_V1_PAIRS`` (44 columns) and
       then forcing ``feature_weights_away_ = feature_weights_home_``
       after each fit so the post-fit model acts symmetrically.

  v2 — asymmetric (current head): the same 22 stems as FP_FEATURE_COLS,
       two independent weight vectors learned jointly in the NLL.

The comparison runs on the same temporal CV splits from ModelEvaluator
and reports:
  - pooled log-loss
  - pooled Brier score
  - pooled Ranked Probability Score (RPS)
  - outcome accuracy
  - which L2 penalty each fit selected (median across folds)

Usage
-----
    python scripts/ab_backtest_fp_v1_vs_v2.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.config_models import FP_FEATURE_COLS, FP_FEATURE_COLS_V1_PAIRS
from models.evaluation import ModelEvaluator
from models.feature_poisson import FeaturePoisson

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ab_backtest")

FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"


class FeaturePoissonV1Symmetric(FeaturePoisson):
    """
    Post-hoc symmetric emulation of the pre-v2.0 FeaturePoisson.

    The NLL and optimizer still fit asymmetric weights (the parameter
    space is unchanged), but AFTER each fit we overwrite the away
    weight vector with the home one so predictions are forced to be
    symmetric. This is the closest faithful re-creation of the v1
    model's effective capacity without reverting the module.
    """

    def fit(self, df: pd.DataFrame, **kwargs) -> "FeaturePoissonV1Symmetric":
        super().fit(df, **kwargs)
        # Force symmetric weights post-fit: use the mean of home/away
        # (keeps the optimizer's discovered mid-point rather than an
        # arbitrary side) then duplicate.
        shared = 0.5 * (self.feature_weights_home_ + self.feature_weights_away_)
        self.feature_weights_home_ = shared.copy()
        self.feature_weights_away_ = shared.copy()
        self.feature_weights_ = np.concatenate([shared, shared])
        return self


def _run_eval(name: str, model: FeaturePoisson, df: pd.DataFrame) -> dict[str, float]:
    """Run walk-forward CV and return pooled metrics with picked L2."""
    evaluator = ModelEvaluator()
    l2_picks: list[float] = []

    # We want to capture per-fold L2. Monkey-patch the model's fit to
    # stash the last chosen L2 into a list. The evaluator calls fit()
    # once per fold.
    original_fit = model.fit

    def _wrapped_fit(train_df, **kw):
        res = original_fit(train_df, **kw)
        l2_picks.append(float(model.l2_used_))
        return res

    model.fit = _wrapped_fit  # type: ignore[method-assign]
    logger.info("==> evaluating %s", name)
    metrics = evaluator.evaluate_model(model, df)
    model.fit = original_fit  # type: ignore[method-assign]

    metrics["l2_median"] = median(l2_picks) if l2_picks else float("nan")
    metrics["l2_picks"] = l2_picks
    return metrics


def main() -> None:
    if not FEATURES_PATH.exists():
        raise SystemExit(f"features.parquet not found at {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    logger.info("Loaded %d matches from %s", len(df), FEATURES_PATH)

    # v1 — symmetric, frozen 44-column pair list. The constructor will
    # collapse those to 22 stems via ``_coerce_to_stems``, so the
    # underlying stem set matches v2; the difference is the POST-FIT
    # symmetrization forced by FeaturePoissonV1Symmetric.
    v1 = FeaturePoissonV1Symmetric(
        feature_cols=FP_FEATURE_COLS_V1_PAIRS,
        half_life_years=1.5,
    )

    # v2 — asymmetric, current FP_FEATURE_COLS (22 stems)
    v2 = FeaturePoisson(
        feature_cols=FP_FEATURE_COLS,
        half_life_years=1.5,
    )

    m_v1 = _run_eval("v1 (symmetric)", v1, df)
    m_v2 = _run_eval("v2 (asymmetric)", v2, df)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def _fmt(m: dict[str, float]) -> str:
        return (
            f"LL={m['log_loss']:.4f}  "
            f"Brier={m['brier_score']:.4f}  "
            f"RPS={m['rps']:.4f}  "
            f"Acc={m['accuracy']*100:.2f}%  "
            f"L2_median={m['l2_median']:.3f}"
        )

    print()
    print("=" * 78)
    print("  FeaturePoisson v1 (symmetric, 22 stems w_home=w_away)  vs")
    print("  FeaturePoisson v2 (asymmetric, 22 stems w_home, w_away)")
    print("=" * 78)
    print(f"  v1: {_fmt(m_v1)}")
    print(f"  v2: {_fmt(m_v2)}")
    print("-" * 78)

    dll = m_v2["log_loss"] - m_v1["log_loss"]
    dbs = m_v2["brier_score"] - m_v1["brier_score"]
    drps = m_v2["rps"] - m_v1["rps"]
    dacc = m_v2["accuracy"] - m_v1["accuracy"]
    better = "YES" if dll < -1e-4 else ("NEUTRAL" if abs(dll) <= 1e-4 else "NO")

    print(
        f"  Delta (v2 - v1):   "
        f"LL={dll:+.4f}  Brier={dbs:+.4f}  RPS={drps:+.4f}  "
        f"Acc={dacc*100:+.2f}%"
    )
    print(f"  v2 improves log-loss? {better}")
    print(f"  v1 L2 picks per fold: {m_v1['l2_picks']}")
    print(f"  v2 L2 picks per fold: {m_v2['l2_picks']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
