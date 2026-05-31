"""
src/ml/research/train_candidate.py

Train a CHALLENGER model for the champion/challenger shadow A/B loop
(see docs/MODEL_RESEARCH_AGENT.md). The model-research agent calls this to turn
ONE hypothesis into a candidate pickle + calibration-first metrics, which the
agent then routes through a leakage critique before deploying as the active
shadow challenger.

WHAT IT DOES
------------
1. Reuses src/ml/train.py's frozen pipeline: the SAME load → normalize →
   feature build and the SAME frozen time split (train ≤2023, val 2024,
   test 2025). The split is never re-defined here (CLAUDE.md #5).
2. Builds a candidate per the spec:
     - model class:   'logreg' | 'gbm'
     - calibration:   'beta' | 'isotonic' | 'sigmoid'
     - regularization / hyperparams (C for logreg; n_estimators/max_depth/lr
       for gbm)
     - optional feature mask: a subset of the 15 AUGMENTED_FEATURES to train on
3. SHADOW-COMPATIBILITY (critical): the emitted pipeline ALWAYS accepts the full
   15-feature AUGMENTED_FEATURES matrix as input — a feature mask is applied as
   an internal pipeline step. paper_trader's shadow scan feeds the fixed
   15-feature matrix and calls `challenger.predict_proba(X)` unchanged, so the
   candidate MUST take 15 columns in. We assert this against the live champion
   before saving.
4. Selects hyperparams CALIBRATION-FIRST (min val log-loss; never accuracy —
   see LITERATURE_REVIEW.md, Walsh & Joshi 2024), then reports out-of-time
   metrics on val AND test: log-loss (primary), ECE, Brier, AUC, accuracy.
   The live champion is scored on the same val/test for a head-to-head delta.
5. Emits:
     data/research/candidates/<id>.pkl           (the fitted pipeline)
     data/research/candidates/<id>.metrics.json   (spec + metrics + deltas)
   Both are gitignored (they are data, per CLAUDE.md #4). This harness does NOT
   touch active_challenger.json — deploying a candidate to the live shadow slot
   is a separate, leakage-gated step the agent performs (see --register, used
   only after the devil's-advocate APPROVES).

NO LOOKAHEAD
------------
Uses only train.py's existing cutoff-respecting feature pipeline and the frozen
split — adds no data source and no new column. Adding a genuinely NEW feature
(e.g. surface-Elo) is OUT OF SCOPE here: it also requires inference-side
plumbing in matches_to_feature_matrix + the shadow path so the feature can be
computed for live, unplayed markets. This harness varies model / calibration /
regularization / feature-subset over the EXISTING 15 features, which is exactly
what is shadow-testable today.

DUMMY EXAMPLE (what you'd run)
------------------------------
    # Same features, swap calibration beta→isotonic (a calibration A/B):
    python -m src.ml.research.train_candidate \
        --id cal_isotonic_0531 --model logreg --cal isotonic \
        --hypothesis "Isotonic may fit the favorite-overconfidence tail better than beta on thin Challenger fields"

    # Swap model class to gradient boosting on the same 15 features:
    python -m src.ml.research.train_candidate \
        --id gbm_0531 --model gbm --cal beta --n-estimators 200 --max-depth 2

Output (abridged):
    val  logloss 0.6612  ece 0.0314   (champion 0.6630 / 0.0339)
    test logloss 0.6588  ece 0.0298   (champion 0.6601 / 0.0327)
    Δ test logloss -0.0013  Δ test ece -0.0029   → candidate slightly better-calibrated
    saved data/research/candidates/cal_isotonic_0531.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.ml.calibration_methods import BetaCalibratedClassifier
from src.ml.research.transformers import ColumnMask
from src.ml.train import (
    AUGMENTED_FEATURES,
    C_GRID,
    build_features,
    filter_irregular,
    load_and_normalize,
    time_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CANDIDATES_DIR = _REPO / "data" / "research" / "candidates"
CHAMPION_PATH  = _REPO / "data" / "processed" / "model_augmented_beta.pkl"
ACTIVE_CHALLENGER_PATH = _REPO / "data" / "research" / "active_challenger.json"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray,
                               n_bins: int = 10) -> float:
    """
    Expected Calibration Error: average gap between predicted probability and
    observed win rate, weighted by bin population, over `n_bins` equal-width
    bins on [0, 1]. See LITERATURE_REVIEW.md. Lower is better; this is a
    primary objective for the research loop alongside log-loss.
    """
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin is closed on the right so p==1.0 lands somewhere
        in_bin = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if not in_bin.any():
            continue
        conf = probs[in_bin].mean()
        acc = y_true[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(acc - conf)
    return float(ece)


def _metrics(pipe, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    probs = pipe.predict_proba(X)[:, 1]
    return {
        "n":       int(len(y)),
        "logloss": float(log_loss(y, probs)),
        "ece":     expected_calibration_error(y, probs),
        "brier":   float(brier_score_loss(y, probs)),
        "auc":     float(roc_auc_score(y, probs)),
        "accuracy": float((probs.round() == y).mean()),
    }


# --------------------------------------------------------------------------- #
# Candidate pipeline construction
# --------------------------------------------------------------------------- #

def _base_estimator(model: str, C: float, n_estimators: int, max_depth: int,
                    learning_rate: float):
    if model == "logreg":
        return LogisticRegression(C=C, max_iter=1000, random_state=42, solver="lbfgs")
    if model == "gbm":
        return GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=42,
        )
    raise ValueError(f"Unknown model={model!r} (expected 'logreg' or 'gbm')")


def _calibrated(base, cal_method: str):
    if cal_method == "beta":
        return BetaCalibratedClassifier(base, cv=5)
    if cal_method in ("isotonic", "sigmoid"):
        return CalibratedClassifierCV(base, cv=5, method=cal_method)
    raise ValueError(f"Unknown cal_method={cal_method!r}")


def make_candidate_pipeline(model: str, cal_method: str, C: float,
                            keep_idx: List[int], n_estimators: int,
                            max_depth: int, learning_rate: float) -> Pipeline:
    """
    impute(15) → [mask(subset)] → scale → calibrated estimator.

    The imputer is fit on all 15 columns so the pipeline always takes 15-col
    input (shadow-compatible). The ColumnMask step is added ONLY when a real
    subset is requested; when all 15 features are kept the pipeline is pure
    sklearn (no custom class to unpickle), maximizing portability.
    """
    base = _base_estimator(model, C, n_estimators, max_depth, learning_rate)
    cal = _calibrated(base, cal_method)
    steps = [("impute", SimpleImputer(strategy="mean"))]
    if len(keep_idx) < len(AUGMENTED_FEATURES):
        steps.append(("mask", ColumnMask(keep_idx)))
    steps += [("scale", StandardScaler()), ("model", cal)]
    return Pipeline(steps)


# --------------------------------------------------------------------------- #
# Train + evaluate
# --------------------------------------------------------------------------- #

def train_candidate(
    challenger_id: str,
    hypothesis: str,
    model: str = "logreg",
    cal_method: str = "beta",
    features: Optional[List[str]] = None,
    C_grid: Optional[List[float]] = None,
    n_estimators: int = 200,
    max_depth: int = 2,
    learning_rate: float = 0.05,
    start_year: int = 2018,
    end_year: int = 2025,
    fetch_tennis_data: bool = True,
) -> Dict:
    """
    Train one challenger and write its pickle + metrics json. Returns the
    metrics dict. See module docstring for the full contract.

    `features` is a subset of AUGMENTED_FEATURES (default: all 15). For logreg
    the C grid is searched calibration-first on val; for gbm a single config is
    fit (hyperparam search can be added later).
    """
    features = features or list(AUGMENTED_FEATURES)
    unknown = [f for f in features if f not in AUGMENTED_FEATURES]
    if unknown:
        raise ValueError(f"unknown features {unknown}; this harness only varies "
                         f"over the existing 15 AUGMENTED_FEATURES (a NEW feature "
                         f"needs inference-side plumbing — see module docstring)")
    keep_idx = [AUGMENTED_FEATURES.index(f) for f in features]
    C_grid = C_grid or C_GRID

    # ── data (reuse champion pipeline; same frozen split) ──────────────────
    df_raw = load_and_normalize(start_year, end_year, fetch_tennis_data)
    df_raw = filter_irregular(df_raw)
    df = build_features(df_raw)            # full 15-feature matrix + label
    train, val, test = time_split(df)

    X_tr, y_tr = train[AUGMENTED_FEATURES].values, train["label"].values
    X_va, y_va = val[AUGMENTED_FEATURES].values,   val["label"].values
    X_te, y_te = test[AUGMENTED_FEATURES].values,  test["label"].values

    # ── calibration-first selection ───────────────────────────────────────
    grid = C_grid if model == "logreg" else [C_grid[0] if C_grid else 1.0]
    selection = []
    best = None
    for C in grid:
        pipe = make_candidate_pipeline(model, cal_method, C, keep_idx,
                                       n_estimators, max_depth, learning_rate)
        pipe.fit(X_tr, y_tr)
        ll = float(log_loss(y_va, pipe.predict_proba(X_va)[:, 1]))
        selection.append({"C": C, "val_logloss": ll})
        logger.info(f"  {model}/{cal_method} C={C:<6} val_logloss={ll:.4f}")
        if best is None or ll < best["val_logloss"]:
            best = {"C": C, "val_logloss": ll, "pipe": pipe}

    pipe = best["pipe"]
    logger.info(f"  → selected C={best['C']} (val_logloss={best['val_logloss']:.4f})")

    cand_metrics = {"val": _metrics(pipe, X_va, y_va),
                    "test": _metrics(pipe, X_te, y_te)}

    # ── champion head-to-head on the same val/test ─────────────────────────
    champion_metrics = None
    if CHAMPION_PATH.exists():
        try:
            with open(CHAMPION_PATH, "rb") as f:
                champ = pickle.load(f)
            champion_metrics = {"val": _metrics(champ, X_va, y_va),
                                "test": _metrics(champ, X_te, y_te)}
            # SHADOW-COMPATIBILITY assertion: candidate must accept the SAME
            # 15-col input the champion does, or the shadow scan can't score it.
            assert champ.predict_proba(X_te[:3]).shape == pipe.predict_proba(X_te[:3]).shape, \
                "candidate predict_proba shape != champion; not shadow-compatible"
        except Exception as e:
            logger.warning(f"could not score champion baseline: {e}")

    deltas = {}
    if champion_metrics:
        for split in ("val", "test"):
            deltas[f"{split}_logloss"] = round(
                cand_metrics[split]["logloss"] - champion_metrics[split]["logloss"], 5)
            deltas[f"{split}_ece"] = round(
                cand_metrics[split]["ece"] - champion_metrics[split]["ece"], 5)

    out = {
        "challenger_id": challenger_id,
        "hypothesis": hypothesis,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec": {
            "model": model, "cal_method": cal_method, "selected_C": best["C"],
            "features": features, "n_features": len(features),
            "n_estimators": n_estimators if model == "gbm" else None,
            "max_depth": max_depth if model == "gbm" else None,
            "learning_rate": learning_rate if model == "gbm" else None,
        },
        "split": {"train_n": int(len(train)), "val_n": int(len(val)),
                  "test_n": int(len(test))},
        "selection": selection,
        "metrics": cand_metrics,
        "champion_baseline": champion_metrics,
        "deltas": deltas,
    }

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    pkl_path = CANDIDATES_DIR / f"{challenger_id}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(pipe, f)
    with open(CANDIDATES_DIR / f"{challenger_id}.metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    logger.info(f"  saved {pkl_path}")
    if deltas:
        logger.info(f"  Δ test logloss {deltas.get('test_logloss'):+.5f}  "
                    f"Δ test ece {deltas.get('test_ece'):+.5f}  "
                    f"(negative = candidate better-calibrated)")
    return out


def register_challenger(challenger_id: str, hypothesis: str) -> None:
    """
    Promote a trained candidate to the live shadow slot by writing
    active_challenger.json. The agent calls this ONLY after the devil's-advocate
    leakage critique APPROVES. Verifies the pickle exists first.
    """
    pkl_rel = f"data/research/candidates/{challenger_id}.pkl"
    if not (_REPO / pkl_rel).exists():
        raise FileNotFoundError(f"no candidate pickle at {pkl_rel} — train it first")
    payload = {
        "challenger_id": challenger_id,
        "pickle": pkl_rel,
        "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hypothesis": hypothesis,
    }
    ACTIVE_CHALLENGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_CHALLENGER_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"registered shadow challenger {challenger_id!r} "
                f"→ {ACTIVE_CHALLENGER_PATH}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description="Train a shadow-A/B challenger model")
    p.add_argument("--id", required=True, help="unique challenger id")
    p.add_argument("--hypothesis", required=True, help="one-line research hypothesis")
    p.add_argument("--model", choices=["logreg", "gbm"], default="logreg")
    p.add_argument("--cal", dest="cal_method",
                   choices=["beta", "isotonic", "sigmoid"], default="beta")
    p.add_argument("--features", nargs="*", default=None,
                   help="subset of the 15 AUGMENTED_FEATURES (default: all)")
    p.add_argument("--C", type=float, default=None,
                   help="pin logreg C (default: grid-search calibration-first)")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--start-year", type=int, default=2018)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--no-network", action="store_true",
                   help="skip tennis-data.co.uk fetch (heuristic dates only)")
    p.add_argument("--register", action="store_true",
                   help="ALSO deploy as the active shadow challenger "
                        "(agent uses this only after devil's-advocate APPROVE)")
    args = p.parse_args()

    out = train_candidate(
        challenger_id=args.id,
        hypothesis=args.hypothesis,
        model=args.model,
        cal_method=args.cal_method,
        features=args.features,
        C_grid=[args.C] if args.C is not None else None,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        start_year=args.start_year,
        end_year=args.end_year,
        fetch_tennis_data=not args.no_network,
    )

    print("\n" + "=" * 60)
    print(f"CHALLENGER {out['challenger_id']}  ({out['spec']['model']}/"
          f"{out['spec']['cal_method']}, {out['spec']['n_features']} feats)")
    print("=" * 60)
    for split in ("val", "test"):
        m = out["metrics"][split]
        line = (f"  {split:4}  logloss {m['logloss']:.4f}  ece {m['ece']:.4f}  "
                f"brier {m['brier']:.4f}  auc {m['auc']:.4f}")
        if out["champion_baseline"]:
            c = out["champion_baseline"][split]
            line += f"   (champ ll {c['logloss']:.4f} / ece {c['ece']:.4f})"
        print(line)
    if out["deltas"]:
        print(f"\n  Δ test logloss {out['deltas']['test_logloss']:+.5f}   "
              f"Δ test ece {out['deltas']['test_ece']:+.5f}   "
              f"(negative = candidate better-calibrated)")
    print("\n  Calibration-first: judge on logloss/ece, NOT accuracy.")
    print("  Next: leakage critique, then --register to deploy as shadow challenger.")

    if args.register:
        register_challenger(args.id, args.hypothesis)


if __name__ == "__main__":
    main()
