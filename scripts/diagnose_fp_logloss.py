"""
scripts/diagnose_fp_logloss.py
------------------------------
Diagnostic script for Task 3b: quantify WHY Feature Poisson walk-forward
log-loss is ~4.3 instead of the expected ~1.0 for calibrated EPL 1X2.

Outputs:
  1. Per-match LL contribution (worst 20 matches)
  2. Prediction extreme histogram (p_true < 0.01/0.02/0.05/0.10)
  3. Per-fold LL at eps=1e-15 vs eps=1e-7
  4. Team param spread per fold (alpha/beta min/max/sum)
  5. Lambda distribution (p50/p90/p99)
  6. Eps sensitivity table (aggregate LL at multiple eps values)

Usage:
    python scripts/diagnose_fp_logloss.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.evaluation import ModelEvaluator
from models.feature_poisson import FeaturePoisson

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diagnose_fp")

FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"


def _outcome_label(hg: int, ag: int) -> str:
    if hg > ag:
        return "H"
    elif hg == ag:
        return "D"
    return "A"


def _log_loss_at_eps(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
    clipped = np.clip(y_pred, eps, 1.0 - eps)
    return -float(np.mean(np.sum(y_true * np.log(clipped), axis=1)))


def _per_match_ll(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> np.ndarray:
    """Return per-match log-loss (n,) array."""
    clipped = np.clip(y_pred, eps, 1.0 - eps)
    return -np.sum(y_true * np.log(clipped), axis=1)


def main() -> None:
    if not FEATURES_PATH.exists():
        raise SystemExit(f"features.parquet not found at {FEATURES_PATH}")

    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} matches from {FEATURES_PATH}")

    evaluator = ModelEvaluator()
    splits = evaluator.temporal_cv_splits(df)
    print(f"Generated {len(splits)} walk-forward folds\n")

    # Collectors
    all_rows: list[dict] = []
    fold_summaries: list[dict] = []

    # Use fixed L2=1.0 to skip the 5-point grid search per fold.
    # This cuts runtime from ~70 min to ~12 min. The grid search itself
    # uses the broken eps=1e-15 evaluator anyway, so its selection is
    # suspect — a fixed midrange L2 is fine for diagnostics.
    for fold_idx, (train, test) in enumerate(splits):
        model = FeaturePoisson(symmetric_mode=True, l2_penalty=1.0)
        model.fit(train)

        preds = model.predict_batch(test)

        # Team param stats
        alphas = np.array(list(model.attack_.values()))
        betas = np.array(list(model.defense_.values()))

        fold_summaries.append({
            "fold": fold_idx,
            "n_train": len(train),
            "n_test": len(test),
            "n_teams": len(model.teams_),
            "alpha_min": alphas.min(),
            "alpha_max": alphas.max(),
            "alpha_sum": alphas.sum(),
            "beta_min": betas.min(),
            "beta_max": betas.max(),
            "beta_sum": betas.sum(),
            "rho": model.rho_,
            "l2_used": model.l2_used_,
            "intercept": model.intercept_,
            "home_adv": model.home_adv_,
        })

        for i, (_, row) in enumerate(test.iterrows()):
            ht = row["home_team"]
            at = row["away_team"]
            hg = int(row["home_goals"])
            ag = int(row["away_goals"])
            outcome = _outcome_label(hg, ag)

            p_h = preds.iloc[i]["p_home"]
            p_d = preds.iloc[i]["p_draw"]
            p_a = preds.iloc[i]["p_away"]

            p_true = {"H": p_h, "D": p_d, "A": p_a}[outcome]

            # Compute lambdas for this match
            alpha_h = model.attack_.get(ht, 0.85)
            beta_h = model.defense_.get(ht, 1.15)
            alpha_a = model.attack_.get(at, 0.85)
            beta_a = model.defense_.get(at, 1.15)

            log_lam_h = (
                model.intercept_
                + np.log(max(alpha_h, 1e-10))
                - np.log(max(beta_a, 1e-10))
                + model.home_adv_
            )
            log_lam_a = (
                model.intercept_
                + np.log(max(alpha_a, 1e-10))
                - np.log(max(beta_h, 1e-10))
            )
            lam_h = np.exp(np.clip(log_lam_h, -10, 5))
            lam_a = np.exp(np.clip(log_lam_a, -10, 5))

            all_rows.append({
                "fold": fold_idx,
                "date": row["date"],
                "home": ht,
                "away": at,
                "score": f"{hg}-{ag}",
                "outcome": outcome,
                "p_home": p_h,
                "p_draw": p_d,
                "p_away": p_a,
                "p_true": p_true,
                "lam_h": lam_h,
                "lam_a": lam_a,
                "alpha_h": alpha_h,
                "beta_h": beta_h,
                "alpha_a": alpha_a,
                "beta_a": beta_a,
            })

        if (fold_idx + 1) % 5 == 0:
            print(f"  ... fold {fold_idx + 1}/{len(splits)} done")

    results = pd.DataFrame(all_rows)
    n = len(results)
    print(f"\nTotal OOF predictions: {n}\n")

    # ==================================================================
    # 1. Worst 20 matches by per-match LL contribution
    # ==================================================================
    print("=" * 80)
    print("  1. WORST 20 MATCHES (per-match log-loss, eps=1e-15)")
    print("=" * 80)
    results["match_ll"] = -np.log(np.clip(results["p_true"].values, 1e-15, 1.0))
    worst = results.nlargest(20, "match_ll")
    for _, r in worst.iterrows():
        print(
            f"  fold={r['fold']:2d}  {r['home']:>20s} {r['score']} "
            f"{r['away']:<20s}  outcome={r['outcome']}  "
            f"p_true={r['p_true']:.6f}  LL={r['match_ll']:.2f}  "
            f"lam_h={r['lam_h']:.2f}  lam_a={r['lam_a']:.2f}"
        )

    # ==================================================================
    # 2. Prediction extreme histogram
    # ==================================================================
    print()
    print("=" * 80)
    print("  2. PREDICTION EXTREME HISTOGRAM")
    print("=" * 80)
    total_ll = results["match_ll"].sum()
    for threshold in [0.001, 0.01, 0.02, 0.05, 0.10]:
        mask = results["p_true"] < threshold
        cnt = mask.sum()
        ll_from = results.loc[mask, "match_ll"].sum()
        pct_matches = 100 * cnt / n
        pct_ll = 100 * ll_from / total_ll if total_ll > 0 else 0
        print(
            f"  p_true < {threshold:.3f}:  {cnt:4d} matches "
            f"({pct_matches:5.1f}%)  |  LL contribution: "
            f"{ll_from:8.1f} / {total_ll:.1f} ({pct_ll:5.1f}%)"
        )

    # ==================================================================
    # 3. Per-fold LL at eps=1e-15 vs eps=1e-7
    # ==================================================================
    print()
    print("=" * 80)
    print("  3. PER-FOLD LOG-LOSS (eps=1e-15 vs eps=1e-7)")
    print("=" * 80)
    print(f"  {'Fold':>4s}  {'n_test':>6s}  {'LL(1e-15)':>10s}  "
          f"{'LL(1e-7)':>10s}  {'Delta':>8s}")
    print("  " + "-" * 46)
    for fold_idx in range(len(splits)):
        mask = results["fold"] == fold_idx
        fold_data = results[mask]
        y_true = np.zeros((len(fold_data), 3))
        for i, outcome in enumerate(fold_data["outcome"].values):
            y_true[i] = {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}[outcome]
        y_pred = fold_data[["p_home", "p_draw", "p_away"]].values

        ll_15 = _log_loss_at_eps(y_true, y_pred, 1e-15)
        ll_7 = _log_loss_at_eps(y_true, y_pred, 1e-7)
        print(
            f"  {fold_idx:4d}  {len(fold_data):6d}  "
            f"{ll_15:10.4f}  {ll_7:10.4f}  {ll_15 - ll_7:+8.4f}"
        )

    # ==================================================================
    # 4. Team param spread per fold
    # ==================================================================
    print()
    print("=" * 80)
    print("  4. TEAM PARAMETER SPREAD PER FOLD")
    print("=" * 80)
    print(f"  {'Fold':>4s}  {'n_teams':>7s}  "
          f"{'a_min':>6s} {'a_max':>6s} {'a_sum':>7s}  "
          f"{'b_min':>6s} {'b_max':>6s} {'b_sum':>7s}  "
          f"{'rho':>7s}  {'L2':>6s}")
    print("  " + "-" * 80)
    for fs in fold_summaries:
        print(
            f"  {fs['fold']:4d}  {fs['n_teams']:7d}  "
            f"{fs['alpha_min']:6.3f} {fs['alpha_max']:6.3f} "
            f"{fs['alpha_sum']:7.2f}  "
            f"{fs['beta_min']:6.3f} {fs['beta_max']:6.3f} "
            f"{fs['beta_sum']:7.2f}  "
            f"{fs['rho']:7.4f}  {fs['l2_used']:6.2f}"
        )

    # ==================================================================
    # 5. Lambda distribution
    # ==================================================================
    print()
    print("=" * 80)
    print("  5. LAMBDA DISTRIBUTION (across all OOF test matches)")
    print("=" * 80)
    for name, col in [("lambda_home", "lam_h"), ("lambda_away", "lam_a")]:
        vals = results[col].values
        pcts = np.percentile(vals, [50, 90, 95, 99, 99.9])
        print(
            f"  {name:>12s}:  "
            f"p50={pcts[0]:.3f}  p90={pcts[1]:.3f}  p95={pcts[2]:.3f}  "
            f"p99={pcts[3]:.3f}  p99.9={pcts[4]:.3f}  "
            f"max={vals.max():.3f}  min={vals.min():.3f}"
        )
    high_lam = (results["lam_h"] > 5).sum() + (results["lam_a"] > 5).sum()
    print(f"  Matches with any lambda > 5: {high_lam}")

    # ==================================================================
    # 6. Eps sensitivity table
    # ==================================================================
    print()
    print("=" * 80)
    print("  6. EPS SENSITIVITY (pooled aggregate log-loss)")
    print("=" * 80)
    y_true_all = np.zeros((n, 3))
    for i, outcome in enumerate(results["outcome"].values):
        y_true_all[i] = {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}[outcome]
    y_pred_all = results[["p_home", "p_draw", "p_away"]].values

    for eps in [1e-15, 1e-10, 1e-7, 1e-5, 1e-3]:
        ll = _log_loss_at_eps(y_true_all, y_pred_all, eps)
        print(f"  eps={eps:.0e}:  LL = {ll:.4f}")

    # Also compute Brier and RPS at baseline for reference
    brier = float(np.mean(np.sum((y_pred_all - y_true_all) ** 2, axis=1)))
    print(f"\n  Brier score (eps-independent): {brier:.4f}")

    # ==================================================================
    # Summary
    # ==================================================================
    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    p_true_vals = results["p_true"].values
    print(f"  p_true distribution:")
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"    p{pct:2d} = {np.percentile(p_true_vals, pct):.6f}")
    print(f"    min = {p_true_vals.min():.8f}")
    print(f"    max = {p_true_vals.max():.8f}")
    print(f"  Mean p_true: {p_true_vals.mean():.4f}")
    print(f"  Matches where model was 'right' (p_true > 0.5): "
          f"{(p_true_vals > 0.5).sum()} / {n} "
          f"({100 * (p_true_vals > 0.5).sum() / n:.1f}%)")


if __name__ == "__main__":
    main()
