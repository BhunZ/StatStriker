"""
scripts/verify_task3b_fix.py
----------------------------
Walk-forward CV verification for Task 3b fix (Commit C).
Compares FP (8-stem, no team L2, no beta constraint, bounds [-2,2])
against Tier 1 Dixon-Coles baseline.

Success criteria:
  - FP walk-forward LL ≤ 1.05
  - Brier ≤ 0.62
  - Floor hits < 10
"""
import sys, logging, time
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, ".")
from models.feature_poisson import FeaturePoisson
from models.dixon_coles import DixonColesModel
from models.evaluation import ModelEvaluator
from models.config_models import PRED_MIN_PROB

df = pd.read_parquet("data/processed/features.parquet")
df = df.sort_values("date").reset_index(drop=True)
print(f"Dataset: {len(df)} matches, {df['season'].nunique()} seasons")
print(f"PRED_MIN_PROB = {PRED_MIN_PROB}")

evaluator = ModelEvaluator()
splits = evaluator.temporal_cv_splits(df)
print(f"Walk-forward folds: {len(splits)}")

# ── Run both models ──────────────────────────────────────────────
results = {}
for name, model in [
    ("DC (baseline)", DixonColesModel()),
    ("FP 8-stem (fix)", FeaturePoisson(l2_penalty=1.0, symmetric_mode=True)),
]:
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")
    t0 = time.time()

    all_y_true, all_y_pred = [], []
    floor_hits = 0

    for fold_idx, (train, test) in enumerate(splits):
        model_copy = type(model)(
            **({"l2_penalty": 1.0, "symmetric_mode": True} if isinstance(model, FeaturePoisson) else {})
        )
        model_copy.fit(train)
        preds = model_copy.predict_batch(test)

        y_true = np.array([
            evaluator._outcome_vector(int(r["home_goals"]), int(r["away_goals"]))
            for _, r in test.iterrows()
        ])
        y_pred = preds[["p_home", "p_draw", "p_away"]].values

        # Count floor hits
        for _, row in preds.iterrows():
            for col in ["p_home", "p_draw", "p_away"]:
                if abs(row[col] - PRED_MIN_PROB / (PRED_MIN_PROB * 2 + (1 - PRED_MIN_PROB))) < 0.001:
                    floor_hits += 1

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

        fold_ll = evaluator.log_loss(y_true, y_pred)
        print(f"  Fold {fold_idx+1:2d}/{len(splits)}: "
              f"train={len(train):4d} test={len(test):3d} LL={fold_ll:.4f}")

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    metrics = {
        "log_loss": evaluator.log_loss(y_true_all, y_pred_all),
        "brier": evaluator.brier_score(y_true_all, y_pred_all),
        "rps": evaluator.ranked_probability_score(y_true_all, y_pred_all),
        "accuracy": evaluator.accuracy(y_true_all, y_pred_all),
        "floor_hits": floor_hits,
        "time": time.time() - t0,
    }
    results[name] = metrics

    print(f"\n  {name} RESULTS:")
    print(f"    LL    = {metrics['log_loss']:.4f}")
    print(f"    Brier = {metrics['brier']:.4f}")
    print(f"    RPS   = {metrics['rps']:.4f}")
    print(f"    Acc   = {metrics['accuracy']:.4f}")
    print(f"    Floor = {metrics['floor_hits']}")
    print(f"    Time  = {metrics['time']:.1f}s")

# ── Summary ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for name, m in results.items():
    print(f"  {name:20s}  LL={m['log_loss']:.4f}  Brier={m['brier']:.4f}  "
          f"RPS={m['rps']:.4f}  Acc={m['accuracy']:.3f}  Floor={m['floor_hits']}")

fp = results.get("FP 8-stem (fix)", {})
print(f"\n  SUCCESS CRITERIA:")
print(f"    LL ≤ 1.05:    {'PASS' if fp.get('log_loss', 99) <= 1.05 else 'FAIL'} ({fp.get('log_loss', '?'):.4f})")
print(f"    Brier ≤ 0.62: {'PASS' if fp.get('brier', 99) <= 0.62 else 'FAIL'} ({fp.get('brier', '?'):.4f})")
print(f"    Floor < 10:   {'PASS' if fp.get('floor_hits', 99) < 10 else 'FAIL'} ({fp.get('floor_hits', '?')})")
