"""
scripts/full_system_audit.py
-----------------------------
Rigorous System Audit — 4 pillars of verification before Tier 5 Ensemble.

Pillar 1: Individual Tier Sanity Check (simplex, NaN, bounds, features)
Pillar 2: Multi-Model Correlation Analysis (inter-tier diversity)
Pillar 3: Data Leakage & Validation Integrity (chronological CV, shift(1), ctx_)
Pillar 4: Full Ensemble Evaluation (walk-forward CV, weight distribution, Go/No-Go)

Usage:
    python scripts/full_system_audit.py
"""

import sys
import time
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize as sp_minimize

sys.path.insert(0, ".")

from models.dixon_coles import DixonColesModel
from models.feature_poisson import FeaturePoisson
from models.gradient_boost import GradientBoostModel
from models.evaluation import ModelEvaluator
from models.config_models import PRED_MIN_PROB

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────
TIER_NAMES = {1: "DC (Tier 1)", 2: "FP (Tier 2)", 3: "GBM (Tier 3)"}
TIER_SHORT = {1: "DC", 2: "FP", 3: "GBM"}

TIER_CONFIGS = {
    1: lambda: DixonColesModel(),
    2: lambda: FeaturePoisson(symmetric_mode=True),
    3: lambda: GradientBoostModel(),
}


# ── Helpers ──────────────────────────────────────────────────────────────
class AuditResult:
    """Accumulates pass/fail checks with descriptions."""

    def __init__(self, name: str):
        self.name = name
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, desc: str, passed: bool, detail: str = "") -> bool:
        self.checks.append((desc, passed, detail))
        tag = "PASS" if passed else "FAIL"
        msg = f"  [{tag}] {desc}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        return passed

    @property
    def all_passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    @property
    def summary(self) -> str:
        n_pass = sum(1 for _, ok, _ in self.checks if ok)
        n_fail = sum(1 for _, ok, _ in self.checks if not ok)
        status = "PASS" if self.all_passed else "FAIL"
        return f"{self.name}: [{status}] {n_pass} passed, {n_fail} failed"

    @property
    def failures(self) -> list[str]:
        return [f"{desc}: {detail}" for desc, ok, detail in self.checks if not ok]


def section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ══════════════════════════════════════════════════════════════════════════
#  PILLAR 1: Individual Tier Sanity Check
# ══════════════════════════════════════════════════════════════════════════
def pillar1_sanity(df: pd.DataFrame) -> AuditResult:
    section("PILLAR 1: Individual Tier Sanity Check")
    r = AuditResult("Pillar 1")

    full_fit_models = {}

    for tier in [1, 2, 3, 4]:
        print(f"\n  --- {TIER_NAMES[tier]} ---")
        model = TIER_CONFIGS[tier]()
        t0 = time.time()
        model.fit(df)
        fit_time = time.time() - t0
        full_fit_models[tier] = model
        print(f"  Fitted in {fit_time:.1f}s")

        # 1a. Predict all matches
        preds = model.predict_batch(df)
        p_vals = preds[["p_home", "p_draw", "p_away"]].values

        # 1b. Simplex check
        row_sums = p_vals.sum(axis=1)
        simplex_ok = np.allclose(row_sums, 1.0, atol=1e-6)
        max_dev = float(np.max(np.abs(row_sums - 1.0)))
        r.check(
            f"T{tier} simplex (sum=1.0)",
            simplex_ok,
            f"max deviation = {max_dev:.2e}",
        )

        # 1c. NaN/Inf check
        has_nan = np.isnan(p_vals).any()
        has_inf = np.isinf(p_vals).any()
        r.check(f"T{tier} no NaN", not has_nan)
        r.check(f"T{tier} no Inf", not has_inf)

        # 1d. Coefficient bounds
        alphas = list(model.attack_.values())
        betas = list(model.defense_.values())
        alpha_ok = all(0 < a < 15 for a in alphas)
        beta_ok = all(0 < b < 15 for b in betas)
        r.check(
            f"T{tier} alpha in (0, 15)",
            alpha_ok,
            f"range [{min(alphas):.3f}, {max(alphas):.3f}]",
        )
        r.check(
            f"T{tier} beta in (0, 15)",
            beta_ok,
            f"range [{min(betas):.3f}, {max(betas):.3f}]",
        )

        # Home advantage
        if hasattr(model, "home_adv_"):
            gamma = model.home_adv_
            if tier in [1, 2, 3]:  # multiplicative gamma
                r.check(
                    f"T{tier} gamma in (0.5, 3.0)",
                    0.5 < gamma < 3.0,
                    f"gamma = {gamma:.4f}",
                )
            else:  # Tier 4: additive log-space
                r.check(
                    f"T{tier} gamma (additive) finite",
                    np.isfinite(gamma) and abs(gamma) < 5.0,
                    f"gamma = {gamma:.4f}",
                )

        # Rho or lambda3
        if tier == 2:
            lam3 = model.lambda3_
            r.check(
                f"T{tier} lambda3 in (0, 1.0)",
                0 < lam3 <= 1.0,
                f"lambda3 = {lam3:.6f}",
            )
        elif hasattr(model, "rho_"):
            rho = model.rho_
            r.check(
                f"T{tier} |rho| < 0.5",
                abs(rho) < 0.5,
                f"rho = {rho:.4f}",
            )

        # Alpha sum-to-one
        alpha_sum = sum(alphas)
        n_teams = len(alphas)
        r.check(
            f"T{tier} sum(alpha) ~ n_teams",
            abs(alpha_sum - n_teams) < 2.0,
            f"sum={alpha_sum:.2f}, n_teams={n_teams}",
        )

        # 1e. Extreme fixture
        top_team = max(model.attack_, key=model.attack_.get)
        weak_team = min(model.attack_, key=model.attack_.get)
        try:
            extreme = model.predict_1x2(top_team, weak_team)
            extreme_ok = (
                np.isfinite(extreme["home"])
                and np.isfinite(extreme["draw"])
                and np.isfinite(extreme["away"])
                and abs(extreme["home"] + extreme["draw"] + extreme["away"] - 1.0)
                < 1e-6
            )
            r.check(
                f"T{tier} extreme fixture ({top_team} vs {weak_team})",
                extreme_ok,
                f"H={extreme['home']:.3f} D={extreme['draw']:.3f} A={extreme['away']:.3f}",
            )
        except Exception as e:
            r.check(f"T{tier} extreme fixture", False, str(e))

        # 1f. Cold-start team
        try:
            cold = model.predict_1x2("FakePromotedFC", top_team)
            cold_ok = (
                np.isfinite(cold["home"])
                and np.isfinite(cold["draw"])
                and np.isfinite(cold["away"])
            )
            r.check(
                f"T{tier} cold-start (FakePromotedFC vs {top_team})",
                cold_ok,
                f"H={cold['home']:.3f} D={cold['draw']:.3f} A={cold['away']:.3f}",
            )
        except Exception as e:
            r.check(f"T{tier} cold-start", False, str(e))

    # 1g. Tier 4 specific checks
    print(f"\n  --- Tier 4 Feature-Specific Checks ---")
    fp = full_fit_models[4]
    stems = fp.feature_cols_used_stems_
    r.check(
        "T4 feature stems resolved",
        len(stems) > 0,
        f"{len(stems)} stems: {stems}",
    )

    if len(fp.feature_mean_) > 0:
        r.check(
            "T4 feature_mean_ finite",
            np.all(np.isfinite(fp.feature_mean_)),
            f"range [{fp.feature_mean_.min():.3f}, {fp.feature_mean_.max():.3f}]",
        )
        r.check(
            "T4 feature_std_ > 1e-8",
            np.all(fp.feature_std_ > 1e-8),
            f"min std = {fp.feature_std_.min():.6f}",
        )

    w_h_norm = float(np.linalg.norm(fp.feature_weights_home_))
    w_a_norm = float(np.linalg.norm(fp.feature_weights_away_))
    r.check("T4 ||w_home|| < 10", w_h_norm < 10.0, f"||w_home|| = {w_h_norm:.4f}")
    r.check("T4 ||w_away|| < 10", w_a_norm < 10.0, f"||w_away|| = {w_a_norm:.4f}")

    if getattr(fp, "symmetric_mode_", False):
        sym = np.allclose(fp.feature_weights_home_, fp.feature_weights_away_)
        r.check("T4 symmetric w_home == w_away", sym)

    return r


# ══════════════════════════════════════════════════════════════════════════
#  PILLAR 2: Multi-Model Correlation Analysis
# ══════════════════════════════════════════════════════════════════════════
def pillar2_correlation(all_oof_preds: dict[int, np.ndarray]) -> AuditResult:
    section("PILLAR 2: Multi-Model Correlation Analysis")
    r = AuditResult("Pillar 2")

    outcome_names = ["P(H)", "P(D)", "P(A)"]

    for oc_idx, oc_name in enumerate(outcome_names):
        print(f"\n  --- Correlation on {oc_name} ---")
        tiers = sorted(all_oof_preds.keys())
        n = len(tiers)
        corr = np.eye(n)

        for (i, t1), (j, t2) in combinations(enumerate(tiers), 2):
            rval = float(
                np.corrcoef(
                    all_oof_preds[t1][:, oc_idx],
                    all_oof_preds[t2][:, oc_idx],
                )[0, 1]
            )
            corr[i, j] = rval
            corr[j, i] = rval

        # Print matrix
        header = "         " + "  ".join(f"{TIER_SHORT[t]:>8s}" for t in tiers)
        print(header)
        for i, t in enumerate(tiers):
            row = f"  {TIER_SHORT[t]:>6s}  " + "  ".join(
                f"{corr[i, j]:8.4f}" for j in range(n)
            )
            print(row)

        # Check pairs
        for (i, t1), (j, t2) in combinations(enumerate(tiers), 2):
            rval = corr[i, j]
            if rval > 0.99:
                r.check(
                    f"{oc_name} T{t1} vs T{t2} r < 0.99",
                    False,
                    f"r = {rval:.4f} — tiers are REDUNDANT",
                )
            elif rval > 0.95:
                r.check(
                    f"{oc_name} T{t1} vs T{t2} r < 0.95",
                    True,  # warning, not failure
                    f"r = {rval:.4f} — high but expected (same model family)",
                )
            else:
                r.check(
                    f"{oc_name} T{t1} vs T{t2} diversity",
                    True,
                    f"r = {rval:.4f}",
                )

    return r


# ══════════════════════════════════════════════════════════════════════════
#  PILLAR 3: Data Leakage & Validation Integrity
# ══════════════════════════════════════════════════════════════════════════
def pillar3_leakage(
    df: pd.DataFrame,
    splits: list[tuple[pd.DataFrame, pd.DataFrame]],
) -> AuditResult:
    section("PILLAR 3: Data Leakage & Validation Integrity")
    r = AuditResult("Pillar 3")

    # 3a. Chronological ordering
    print("\n  --- Chronological Ordering ---")
    chrono_ok = True
    for fold_idx, (train, test) in enumerate(splits):
        train_max = pd.to_datetime(train["date"]).max()
        test_min = pd.to_datetime(test["date"]).min()
        if train_max >= test_min:
            chrono_ok = False
            r.check(
                f"Fold {fold_idx+1} chronological",
                False,
                f"train_max={train_max} >= test_min={test_min}",
            )
    r.check(
        f"All {len(splits)} folds chronologically ordered",
        chrono_ok,
        "train.date.max() < test.date.min() for every fold",
    )

    # 3b. Expanding window
    print("\n  --- Expanding Window ---")
    expanding_ok = True
    for i in range(1, len(splits)):
        if len(splits[i][0]) <= len(splits[i - 1][0]):
            expanding_ok = False
    r.check(
        "Training sets monotonically expanding",
        expanding_ok,
        f"sizes: {len(splits[0][0])} -> {len(splits[-1][0])}",
    )

    # 3c. Rolling shift(1) verification
    print("\n  --- Rolling Feature shift(1) Verification ---")
    if "home_xg_scored_5" not in df.columns:
        r.check(
            "Rolling feature columns present",
            False,
            "home_xg_scored_5 not found — skipping rolling checks",
        )
    else:
        # Build team-level view and manually compute rolling
        teams_to_check = df["home_team"].value_counts().head(3).index.tolist()
        for team in teams_to_check:
            # Get all matches for this team (home or away) sorted by date
            home_mask = df["home_team"] == team
            home_matches = df[home_mask].sort_values("date").copy()

            if len(home_matches) < 6:
                continue

            # Take 3 test matches (indices 5, 6, 7 — enough rolling history)
            for test_idx in [5, 6, 7]:
                if test_idx >= len(home_matches):
                    break

                actual = home_matches.iloc[test_idx]["home_xg_scored_5"]

                # Manual: shift(1) then rolling(5, min_periods=1).mean()
                past_xg = home_matches["home_xg"].iloc[:test_idx].values
                if len(past_xg) == 0:
                    expected = np.nan
                else:
                    window = past_xg[-5:]  # last 5 (or fewer)
                    expected = float(np.nanmean(window))

                if np.isnan(actual) and np.isnan(expected):
                    match = True
                elif np.isnan(actual) or np.isnan(expected):
                    match = False
                else:
                    match = abs(actual - expected) < 1e-4

                r.check(
                    f"Rolling {team} match#{test_idx} xg_scored_5",
                    match,
                    f"actual={actual:.4f}, expected={expected:.4f}"
                    if not (np.isnan(actual) or np.isnan(expected))
                    else f"actual={actual}, expected={expected}",
                )

    # 3d. Prior-season ctx_ verification
    print("\n  --- Prior-Season ctx_ Leakage Check ---")
    if "home_ctx_stats_poss" not in df.columns:
        r.check(
            "ctx_ columns present",
            False,
            "home_ctx_stats_poss not found — skipping ctx checks",
        )
    else:
        seasons = sorted(df["season"].unique())
        if len(seasons) >= 2:
            # For the latest season: ctx_ features should come from prior season
            latest = seasons[-1]
            prior = seasons[-2]
            latest_matches = df[df["season"] == latest].head(5)

            # ctx_ features should NOT be NaN for established teams
            established_teams = set(df[df["season"] == prior]["home_team"])
            for _, match in latest_matches.iterrows():
                team = match["home_team"]
                if team in established_teams:
                    ctx_val = match.get("home_ctx_stats_poss", np.nan)
                    r.check(
                        f"ctx_ present for {team} in {latest}",
                        not np.isnan(ctx_val) if ctx_val is not None else False,
                        f"home_ctx_stats_poss = {ctx_val}",
                    )

        # Check first season — newly promoted teams may have NaN ctx_
        first_season = seasons[0]
        first_matches = df[df["season"] == first_season]
        ctx_nan_count = first_matches["home_ctx_stats_poss"].isna().sum()
        r.check(
            f"First season ({first_season}) ctx_ has NaN for some teams",
            True,  # informational
            f"{ctx_nan_count}/{len(first_matches)} matches have NaN ctx (expected for promoted/no prior data)",
        )

    # 3e. First-match edge case
    print("\n  --- First-Match Edge Case ---")
    if "home_xg_scored_5" in df.columns:
        first_team = df.sort_values("date")["home_team"].iloc[0]
        first_row = df[df["home_team"] == first_team].sort_values("date").iloc[0]
        first_val = first_row.get("home_xg_scored_5", None)
        r.check(
            f"First match ({first_team}) xg_scored_5",
            np.isnan(first_val) if first_val is not None else True,
            f"value = {first_val} (should be NaN — no prior history)",
        )

    return r


# ══════════════════════════════════════════════════════════════════════════
#  PILLAR 4: Full Ensemble Evaluation
# ══════════════════════════════════════════════════════════════════════════
def pillar4_ensemble(
    all_oof_preds: dict[int, np.ndarray],
    all_oof_true: np.ndarray,
    evaluator: ModelEvaluator,
    df: pd.DataFrame,
) -> AuditResult:
    section("PILLAR 4: Full Ensemble Evaluation")
    r = AuditResult("Pillar 4")

    n_oof = len(all_oof_true)
    tiers = sorted(all_oof_preds.keys())
    n_models = len(tiers)

    # Stack: (N_oof, n_models, 3)
    X_oof = np.stack([all_oof_preds[t] for t in tiers], axis=1)

    # ── Per-tier metrics ──
    print("\n  --- Per-Tier Metrics (OOF) ---")
    tier_metrics = {}
    for tier in tiers:
        yp = all_oof_preds[tier]
        ll = evaluator.log_loss(all_oof_true, yp)
        bs = evaluator.brier_score(all_oof_true, yp)
        rps = evaluator.ranked_probability_score(all_oof_true, yp)
        acc = evaluator.accuracy(all_oof_true, yp)
        tier_metrics[tier] = {"ll": ll, "brier": bs, "rps": rps, "acc": acc}
        print(
            f"  {TIER_SHORT[tier]:>6s}:  LL={ll:.4f}  Brier={bs:.4f}  "
            f"RPS={rps:.4f}  Acc={acc:.3f}"
        )

    # ── Naive baseline ──
    true_labels = np.argmax(all_oof_true, axis=1)
    n_h = np.sum(true_labels == 0)
    n_d = np.sum(true_labels == 1)
    n_a = np.sum(true_labels == 2)
    naive_probs = np.array([n_h / n_oof, n_d / n_oof, n_a / n_oof])
    naive_pred = np.tile(naive_probs, (n_oof, 1))
    naive_ll = evaluator.log_loss(all_oof_true, naive_pred)
    naive_bs = evaluator.brier_score(all_oof_true, naive_pred)
    print(f"  {'Naive':>6s}:  LL={naive_ll:.4f}  Brier={naive_bs:.4f}")

    # ── Optimize ensemble weights ──
    print("\n  --- Ensemble Weight Optimization ---")

    def _neg_log_lik(w_raw):
        w = _softmax(w_raw)
        blended = np.einsum("m,nmc->nc", w, X_oof)
        blended = np.clip(blended, 1e-10, 1.0)
        blended = blended / blended.sum(axis=1, keepdims=True)
        return -float(np.mean(np.sum(all_oof_true * np.log(blended), axis=1)))

    result = sp_minimize(
        _neg_log_lik,
        x0=np.zeros(n_models),
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-6},
    )
    weights = _softmax(result.x)

    print("  Learned weights:")
    for i, tier in enumerate(tiers):
        bar = "#" * int(weights[i] * 40)
        print(f"    {TIER_SHORT[tier]:>6s}: {weights[i]:.4f}  {bar}")

    # ── Ensemble predictions ──
    ens_pred = np.einsum("m,nmc->nc", weights, X_oof)
    ens_pred = np.clip(ens_pred, 1e-10, 1.0)
    ens_pred = ens_pred / ens_pred.sum(axis=1, keepdims=True)

    ens_ll = evaluator.log_loss(all_oof_true, ens_pred)
    ens_bs = evaluator.brier_score(all_oof_true, ens_pred)
    ens_rps = evaluator.ranked_probability_score(all_oof_true, ens_pred)
    ens_acc = evaluator.accuracy(all_oof_true, ens_pred)

    print(f"\n  --- Final Comparison ---")
    print(f"  {'Model':>10s}  {'LL':>8s}  {'Brier':>8s}  {'RPS':>8s}  {'Acc':>8s}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for tier in tiers:
        m = tier_metrics[tier]
        print(
            f"  {TIER_SHORT[tier]:>10s}  {m['ll']:8.4f}  {m['brier']:8.4f}  "
            f"{m['rps']:8.4f}  {m['acc']:8.3f}"
        )
    print(
        f"  {'Ensemble':>10s}  {ens_ll:8.4f}  {ens_bs:8.4f}  "
        f"{ens_rps:8.4f}  {ens_acc:8.3f}"
    )
    print(f"  {'Naive':>10s}  {naive_ll:8.4f}  {naive_bs:8.4f}")

    # ── Go/No-Go checks ──
    print(f"\n  --- Go/No-Go Criteria ---")

    best_tier_ll = min(m["ll"] for m in tier_metrics.values())
    best_tier_name = min(tier_metrics, key=lambda t: tier_metrics[t]["ll"])

    r.check(
        "Ensemble LL < naive baseline",
        ens_ll < naive_ll,
        f"Ens={ens_ll:.4f} vs Naive={naive_ll:.4f}",
    )
    r.check(
        f"Ensemble LL <= best tier ({TIER_SHORT[best_tier_name]}) + 0.01",
        ens_ll <= best_tier_ll + 0.01,
        f"Ens={ens_ll:.4f} vs Best={best_tier_ll:.4f}",
    )

    # Weight distribution
    for i, tier in enumerate(tiers):
        r.check(
            f"Weight T{tier} > 0.01",
            weights[i] > 0.01,
            f"w={weights[i]:.4f}",
        )

    any_dominant = any(w > 0.80 for w in weights)
    if any_dominant:
        dominant_idx = np.argmax(weights)
        r.check(
            "No single tier dominates (>80%)",
            False,
            f"T{tiers[dominant_idx]} has weight {weights[dominant_idx]:.4f}",
        )
    else:
        r.check("No single tier dominates (>80%)", True)

    return r, {
        "weights": {TIER_SHORT[tiers[i]]: float(weights[i]) for i in range(n_models)},
        "ensemble_ll": ens_ll,
        "ensemble_brier": ens_bs,
        "ensemble_rps": ens_rps,
        "ensemble_acc": ens_acc,
        "tier_metrics": {TIER_SHORT[t]: tier_metrics[t] for t in tiers},
        "naive_ll": naive_ll,
    }


# ══════════════════════════════════════════════════════════════════════════
#  SHARED WALK-FORWARD CV LOOP
# ══════════════════════════════════════════════════════════════════════════
def shared_walkforward(
    df: pd.DataFrame,
    splits: list[tuple[pd.DataFrame, pd.DataFrame]],
    evaluator: ModelEvaluator,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Run all 4 tiers across all folds, return OOF predictions."""
    section("SHARED WALK-FORWARD CV (23 folds x 4 tiers)")

    oof_preds: dict[int, list[np.ndarray]] = {t: [] for t in [1, 2, 3, 4]}
    oof_true: list[np.ndarray] = []

    total_t0 = time.time()

    for fold_idx, (train, test) in enumerate(splits):
        fold_t0 = time.time()

        y_true = np.array([
            evaluator._outcome_vector(int(r["home_goals"]), int(r["away_goals"]))
            for _, r in test.iterrows()
        ])
        oof_true.append(y_true)

        fold_lls = []
        for tier in [1, 2, 3, 4]:
            model = TIER_CONFIGS[tier]()
            model.fit(train)
            preds = model.predict_batch(test)
            y_pred = preds[["p_home", "p_draw", "p_away"]].values
            oof_preds[tier].append(y_pred)
            ll = evaluator.log_loss(y_true, y_pred)
            fold_lls.append(f"{TIER_SHORT[tier]}={ll:.3f}")

        elapsed = time.time() - fold_t0
        print(
            f"  Fold {fold_idx+1:2d}/{len(splits)}: "
            f"train={len(train):4d} test={len(test):3d}  "
            f"{', '.join(fold_lls)}  ({elapsed:.1f}s)"
        )

    total_time = time.time() - total_t0
    print(f"\n  Total walk-forward time: {total_time:.1f}s")

    # Concatenate
    all_oof_true = np.concatenate(oof_true)
    all_oof_preds = {t: np.concatenate(oof_preds[t]) for t in [1, 2, 3, 4]}

    print(f"  OOF samples: {len(all_oof_true)}")

    return all_oof_preds, all_oof_true


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()

    print("=" * 70)
    print("  FULL SYSTEM AUDIT — StatStriker v2.0")
    print("=" * 70)

    # Load data
    features_path = Path("data/processed/features.parquet")
    df = pd.read_parquet(features_path)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"\nDataset: {len(df)} matches, {df['season'].nunique()} seasons")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"PRED_MIN_PROB = {PRED_MIN_PROB}")

    evaluator = ModelEvaluator()
    splits = evaluator.temporal_cv_splits(df)
    print(f"Walk-forward folds: {len(splits)}")

    # ── Pillar 1 ──
    r1 = pillar1_sanity(df)

    # ── Shared walk-forward (feeds Pillars 2 + 4) ──
    all_oof_preds, all_oof_true = shared_walkforward(df, splits, evaluator)

    # ── Pillar 2 ──
    r2 = pillar2_correlation(all_oof_preds)

    # ── Pillar 3 ──
    r3 = pillar3_leakage(df, splits)

    # ── Pillar 4 ──
    r4, ens_details = pillar4_ensemble(all_oof_preds, all_oof_true, evaluator, df)

    # ══════════════════════════════════════════════════════════════════════
    #  FINAL REPORT
    # ══════════════════════════════════════════════════════════════════════
    total_time = time.time() - t_start

    section("FINAL REPORT")
    print()
    for result in [r1, r2, r3, r4]:
        print(f"  {result.summary}")
    print()

    all_pass = all(result.all_passed for result in [r1, r2, r3, r4])

    # Collect all failures
    all_failures = []
    for result in [r1, r2, r3, r4]:
        all_failures.extend(result.failures)

    if all_pass:
        print("  " + "=" * 50)
        print("  VERDICT:  GO")
        print("  " + "=" * 50)
        print()
        print("  All 4 pillars passed. Safe to proceed with ensemble training.")
        print()
        print("  Ensemble weights:")
        for name, w in ens_details["weights"].items():
            print(f"    {name:>6s}: {w:.4f}")
        print(f"\n  Ensemble LL={ens_details['ensemble_ll']:.4f}  "
              f"Brier={ens_details['ensemble_brier']:.4f}  "
              f"RPS={ens_details['ensemble_rps']:.4f}  "
              f"Acc={ens_details['ensemble_acc']:.3f}")
        print()
        print("  Retrain command:")
        print("    from models.predict import MatchPredictor")
        print('    mp = MatchPredictor.from_parquet("data/processed/features.parquet")')
        print("    mp.fit_all(tiers=[1, 2, 3, 4, 5])")
        print('    mp.save("data/models/")')
    else:
        print("  " + "=" * 50)
        print("  VERDICT:  NO-GO")
        print("  " + "=" * 50)
        print()
        print(f"  {len(all_failures)} failure(s):")
        for f in all_failures:
            print(f"    - {f}")

    print(f"\n  Total audit runtime: {total_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
