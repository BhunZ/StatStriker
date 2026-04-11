"""
scripts/convert_fp_pickle_to_symmetric.py
-----------------------------------------
One-off: convert a trained v2 asymmetric FeaturePoisson pickle to v1
symmetric behavior by averaging ``w_home`` and ``w_away`` element-wise
in place.

This produces a pickle that is bit-for-bit equivalent to what
``FeaturePoisson(symmetric_mode=True).fit(df)`` would emit on the same
training data, because the NLL and team parameters are unchanged — the
only difference between the two fit paths is the post-fit reshape of
the feature weight vectors. Skipping the ~60-minute retrain is
therefore safe.

The converted pickle keeps ``fp_version_ = 2`` (the underlying
architecture is still v2 asymmetric) and sets ``symmetric_mode_ = True``
so ``models/predict.py``'s loader recognizes it as a production-approved
symmetric pickle.

Run once, after the ``symmetric_mode`` flag lands:

    python scripts/convert_fp_pickle_to_symmetric.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.feature_poisson import FeaturePoisson  # noqa: E402

FP_PKL = ROOT / "data" / "models" / "feature_poisson.pkl"
ENSEMBLE_PKL = ROOT / "data" / "models" / "ensemble.pkl"


def _symmetrize(fp: FeaturePoisson, label: str) -> None:
    """Average w_home/w_away in place and mark the instance symmetric."""
    w_h = np.asarray(fp.feature_weights_home_)
    w_a = np.asarray(fp.feature_weights_away_)
    if w_h.shape != w_a.shape:
        raise SystemExit(
            f"{label}: weight shapes differ: home={w_h.shape} away={w_a.shape}"
        )
    before_delta = float(np.linalg.norm(w_h - w_a))
    shared = 0.5 * (w_h + w_a)
    fp.feature_weights_home_ = shared.copy()
    fp.feature_weights_away_ = shared.copy()
    fp.feature_weights_ = np.concatenate([shared, shared])
    fp.symmetric_mode_ = True
    # Leave fp_version_ = 2 — architecture unchanged, only the weight tie.
    print(
        f"  {label}: pre-avg ||w_h - w_a||={before_delta:.6f} "
        f"||shared||={float(np.linalg.norm(shared)):.6f} "
        f"symmetric_mode_=True fp_version_={getattr(fp, 'fp_version_', None)}"
    )


def main() -> None:
    if not FP_PKL.exists():
        raise SystemExit(f"pickle not found: {FP_PKL}")

    # ---- Tier 4 standalone pickle ----
    with open(FP_PKL, "rb") as f:
        fp = pickle.load(f)
    if not isinstance(fp, FeaturePoisson):
        raise SystemExit(f"{FP_PKL} is not a FeaturePoisson: {type(fp)}")
    print(f"converting {FP_PKL}")
    _symmetrize(fp, "tier4")
    with open(FP_PKL, "wb") as f:
        pickle.dump(fp, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ---- Ensemble pickle: symmetrize its internal FP base model too ----
    # The ensemble stores its own FeaturePoisson instance inside
    # ``base_models_``; leaving it asymmetric would mean the Tier 5
    # ensemble output still leaks v2 asymmetric FP predictions, which
    # defeats the point of the revert.
    if ENSEMBLE_PKL.exists():
        with open(ENSEMBLE_PKL, "rb") as f:
            ens = pickle.load(f)
        touched = False
        for i, bm in enumerate(getattr(ens, "base_models_", []) or []):
            if isinstance(bm, FeaturePoisson):
                print(f"converting {ENSEMBLE_PKL}")
                _symmetrize(bm, f"ensemble.base_models_[{i}]")
                touched = True
        if touched:
            with open(ENSEMBLE_PKL, "wb") as f:
                pickle.dump(ens, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print(f"{ENSEMBLE_PKL}: no FeaturePoisson base model found")


if __name__ == "__main__":
    main()
