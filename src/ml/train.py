"""
train.py

Full experiment pipeline for the tennis prediction Theo model.

WHAT THIS DOES
--------------
1. Load TML data (2018-2024, include_challenger=True) via TMLMatchLoader
2. Call normalize() to get per-match dates from tennis-data.co.uk
3. Filter out date_confidence == 'irregular_format' rows
4. Call compute_all() to engineer the 6 features
5. Build the symmetric prediction problem (random A/B flip)
6. Fixed time-based split:
       train : match_date <= 2022-12-31
       val   : 2023-01-01 <= match_date <= 2023-12-31
       test  : match_date >= 2024-01-01
7. Grid search C (regularization) on val set — separately for:
       baseline  : rank_ratio only
       augmented : rank_ratio + all 6 engineered features
8. Report accuracy, log-loss, Brier score for best-C models on val
9. Final evaluation on test set

SPLIT CONVENTION (from README)
-------------------------------
train ≤ 2022, val = 2023, test = 2024.
This is frozen. Never change it mid-experiment.

FEATURES
--------
Baseline:
    rank_ratio_a        player_a_rank / player_b_rank  (lower rank # = better)

Augmented (baseline + these):
    surface_win_rate_diff   surface_win_rate_52w_a - surface_win_rate_52w_b
    recent_form_diff        recent_form_10m_a - recent_form_10m_b
    fatigue_diff_7d         fatigue_minutes_7d_b - fatigue_minutes_7d_a
                            (positive = opponent more fatigued = good for a)
    days_rest_diff          days_since_last_match_a - days_since_last_match_b
    h2h_surface_diff        h2h_surface_advantage_a - h2h_surface_advantage_b
    fatigue_diff_14d        fatigue_minutes_14d_b - fatigue_minutes_14d_a

Missing features are mean-imputed from the training set.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Make imports work whether run as script or module
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.loaders.tml_match_loader import TMLMatchLoader
from src.ml.features.feature_engineer import compute_all, assert_feature_quality, feature_quality_report, FeatureQualityError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants — frozen for all experiments
# ============================================================================

TRAIN_END   = pd.Timestamp("2022-12-31")
VAL_START   = pd.Timestamp("2023-01-01")
VAL_END     = pd.Timestamp("2023-12-31")
TEST_START  = pd.Timestamp("2024-01-01")

# Grid search values for logistic regression C (inverse regularization strength)
C_GRID = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

BASELINE_FEATURES   = ["rank_ratio_a"]
AUGMENTED_FEATURES  = [
    "rank_ratio_a",
    # Form & fatigue
    "surface_win_rate_diff",
    "recent_form_diff",
    "fatigue_diff_21d",
    "fatigue_diff_28d",
    "days_rest_diff",
    "h2h_surface_diff",
    # Player identity (serve / return / physical)
    "ace_rate_diff",
    "df_rate_diff",
    "serve_dominance_diff",
    "return_dominance_diff",
    "first_in_pct_diff",
    "first_won_pct_diff",
    "second_won_pct_diff",
    "height_diff",
]


# ============================================================================
# Data loading
# ============================================================================

def load_and_normalize(
    start_year: int = 2018,
    end_year: int = 2024,
    fetch_tennis_data: bool = True,
) -> pd.DataFrame:
    """
    Load TML, resolve match dates, return normalized DataFrame.

    Args:
        start_year:        First year of TML data to load.
        end_year:          Last year (inclusive).
        fetch_tennis_data: If True, hit tennis-data.co.uk for exact dates.
                           Set False to skip network in tests.

    Returns:
        Normalized DF with match_date and date_confidence columns.
    """
    loader = TMLMatchLoader()
    logger.info(f"Loading TML data {start_year}–{end_year} (include_challenger=True)...")
    t0 = time.time()
    raw = loader.load(
        start_year=start_year,
        end_year=end_year,
        include_challenger=True,
    )
    logger.info(f"  {len(raw):,} raw rows loaded in {time.time()-t0:.1f}s")

    logger.info("Normalizing — resolving per-match dates via tennis-data.co.uk...")
    t0 = time.time()
    df = loader.normalize(raw, fetch_tennis_data=fetch_tennis_data)
    logger.info(f"  {len(df):,} normalized rows in {time.time()-t0:.1f}s")

    # Report date confidence breakdown
    conf_counts = df["date_confidence"].value_counts()
    for conf, n in conf_counts.items():
        logger.info(f"  date_confidence={conf!r}: {n:,} ({100*n/len(df):.1f}%)")

    return df


def filter_irregular(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where date_confidence == 'irregular_format'.

    These are Davis Cup, Tour Finals, Olympics, team events.
    The model is not calibrated for these formats; their round labels
    also don't map to a sensible heuristic offset so match_date is
    just tourney_date (useless for features).
    """
    before = len(df)
    df = df[df["date_confidence"] != "irregular_format"].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped:,} irregular_format rows ({100*dropped/before:.1f}%)")
    return df


# ============================================================================
# Feature construction
# ============================================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run compute_all() then build the symmetric (A/B) prediction problem.

    TML is winner-first by definition, which would leak the label into the
    feature matrix if we used winner/loser directly. We randomly flip ~50%
    of rows so that player_a is the winner only half the time.

    Returns a DataFrame ready for sklearn with columns:
        - All AUGMENTED_FEATURES
        - label  (1 = player_a won, 0 = player_b won)
        - match_date, tournament, surface, tourney_level  (metadata)
    """
    # The normalized DF has player_a/player_b (player_a_won=True always).
    # compute_all() expects winner_name/loser_name — rename, run, then
    # do the random A/B flip AFTER features are computed.
    df = df.rename(columns={"player_a": "winner_name", "player_b": "loser_name"})

    logger.info("Computing engineered features...")
    t0 = time.time()
    df = compute_all(df)
    logger.info(f"  Features computed in {time.time()-t0:.1f}s")

    # ── Symmetric A/B flip ──────────────────────────────────────────────────
    # Randomly assign winner→a or winner→b so the label is ~50/50.
    # This prevents the model from learning "a always wins" artifacts.
    rng = np.random.default_rng(42)
    flip = rng.random(len(df)) < 0.5   # True = flip (loser → a, winner → b)

    out = pd.DataFrame()
    out["match_date"]    = pd.to_datetime(df["match_date"])
    out["tournament"]    = df["tournament"]
    out["surface"]       = df["surface"]
    out["tourney_level"] = df["tourney_level"]
    out["label"]         = np.where(flip, 0, 1).astype(int)  # 1 = player_a won

    # Ranks (lower number = better player)
    rank_a = np.where(flip, df["loser_rank"].values,  df["winner_rank"].values).astype(float)
    rank_b = np.where(flip, df["winner_rank"].values, df["loser_rank"].values).astype(float)
    # rank_ratio_a: rank_a / rank_b. Values < 1 mean player_a is better-ranked.
    # Add small epsilon to avoid div/0 on unranked players.
    out["rank_ratio_a"] = rank_a / (rank_b + 1e-6)

    # Engineered feature diffs (a - b, or b - a for fatigue where higher = bad)
    def _feat_a(col):
        w_col = col + "_w"
        l_col = col + "_l"
        return np.where(flip, df[l_col].values, df[w_col].values).astype(float)

    def _feat_b(col):
        w_col = col + "_w"
        l_col = col + "_l"
        return np.where(flip, df[w_col].values, df[l_col].values).astype(float)

    out["surface_win_rate_diff"] = _feat_a("surface_win_rate_52w") - _feat_b("surface_win_rate_52w")
    out["recent_form_diff"]      = _feat_a("recent_form_10m")      - _feat_b("recent_form_10m")
    out["h2h_surface_diff"]      = _feat_a("h2h_surface_advantage")- _feat_b("h2h_surface_advantage")
    # Fatigue: positive diff means player_b played MORE minutes recently → edge for a
    out["fatigue_diff_21d"]      = _feat_b("fatigue_minutes_21d")   - _feat_a("fatigue_minutes_21d")
    out["fatigue_diff_28d"]      = _feat_b("fatigue_minutes_28d")  - _feat_a("fatigue_minutes_28d")
    out["days_rest_diff"]        = _feat_a("days_since_last_match")- _feat_b("days_since_last_match")

    # ── Player identity diffs ──
    # All "rate-style" features are simple a - b. Higher diff = a's stat is
    # better/larger/more dominant than b's. Sign convention is consistent
    # with surface_win_rate_diff (positive → a is better).
    out["ace_rate_diff"]         = _feat_a("ace_rate_52w")          - _feat_b("ace_rate_52w")
    # df_rate sign flipped: low DF rate is good, so positive diff = a has FEWER DFs
    out["df_rate_diff"]          = _feat_b("df_rate_52w")           - _feat_a("df_rate_52w")
    out["serve_dominance_diff"]  = _feat_a("serve_dominance_52w")   - _feat_b("serve_dominance_52w")
    # return_dominance: lower = better returner. Flip sign so positive = a returns better.
    out["return_dominance_diff"] = _feat_b("return_dominance_52w")  - _feat_a("return_dominance_52w")
    out["first_in_pct_diff"]     = _feat_a("first_in_pct_52w")      - _feat_b("first_in_pct_52w")
    out["first_won_pct_diff"]    = _feat_a("first_won_pct_52w")     - _feat_b("first_won_pct_52w")
    out["second_won_pct_diff"]   = _feat_a("second_won_pct_52w")    - _feat_b("second_won_pct_52w")
    out["height_diff"]           = _feat_a("height_cm")             - _feat_b("height_cm")

    return out


# ============================================================================
# Train / val / test split
# ============================================================================

def time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply the frozen date-based split.

    Returns (train, val, test) DataFrames.
    """
    train = df[df["match_date"] <= TRAIN_END].reset_index(drop=True)
    val   = df[(df["match_date"] >= VAL_START) & (df["match_date"] <= VAL_END)].reset_index(drop=True)
    test  = df[df["match_date"] >= TEST_START].reset_index(drop=True)
    logger.info(
        f"Split sizes — train: {len(train):,}  val: {len(val):,}  test: {len(test):,}"
    )
    return train, val, test


# ============================================================================
# Model building
# ============================================================================

def _make_pipeline(C: float) -> Pipeline:
    """
    Logistic regression pipeline: impute → scale → LR → isotonic calibration.

    Imputation: mean on train set. Scaling: standard (zero mean, unit var).
    Calibration: isotonic regression on a held-out 20% of the training fold.
    """
    lr = LogisticRegression(C=C, max_iter=1000, random_state=42, solver="lbfgs")
    cal = CalibratedClassifierCV(lr, cv=5, method="isotonic")
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale",  StandardScaler()),
        ("model",  cal),
    ])
    return pipe


def grid_search(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: List[str],
    label: str = "label",
) -> Tuple[float, Pipeline, Dict]:
    """
    Search over C_GRID; pick the C that minimises val log-loss.

    Returns (best_C, fitted_pipeline_on_full_train, results_dict).
    The pipeline returned is re-fitted on the FULL train set at best_C.
    """
    X_train = train[features].values
    y_train = train[label].values
    X_val   = val[features].values
    y_val   = val[label].values

    results = []
    for C in C_GRID:
        pipe = _make_pipeline(C)
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, probs)
        acc = (probs.round() == y_val).mean()
        results.append({"C": C, "val_logloss": ll, "val_acc": acc})
        logger.info(f"    C={C:6.3f}  val_logloss={ll:.4f}  val_acc={acc:.4f}")

    best = min(results, key=lambda r: r["val_logloss"])
    best_C = best["C"]
    logger.info(f"  → Best C = {best_C}")

    # Refit on full train at best C
    pipe = _make_pipeline(best_C)
    pipe.fit(X_train, y_train)
    return best_C, pipe, {r["C"]: r for r in results}


# ============================================================================
# Evaluation helpers
# ============================================================================

def eval_metrics(pipe: Pipeline, X: np.ndarray, y: np.ndarray, label: str) -> Dict:
    """Compute accuracy, log-loss, Brier, AUC for a fitted pipeline."""
    probs = pipe.predict_proba(X)[:, 1]
    metrics = {
        "n":        len(y),
        "accuracy": float((probs.round() == y).mean()),
        "logloss":  float(log_loss(y, probs)),
        "brier":    float(brier_score_loss(y, probs)),
        "auc":      float(roc_auc_score(y, probs)),
    }
    logger.info(
        f"{label}: n={metrics['n']:,}  acc={metrics['accuracy']:.4f}  "
        f"ll={metrics['logloss']:.4f}  brier={metrics['brier']:.4f}  "
        f"auc={metrics['auc']:.4f}"
    )
    return metrics


# ============================================================================
# Main experiment
# ============================================================================

def _format_quality_report(report: "pd.DataFrame") -> str:
    """Format a feature_quality_report DataFrame as a readable table string."""
    STATUS_ICON = {"ok": "✓", "warn": "⚠", "error": "✗"}
    lines = [
        f"  {'feature':30}  {'null_rate':>9}  {'warn@':>6}  {'max@':>5}  status",
        "  " + "─" * 65,
    ]
    for _, row in report.iterrows():
        icon = STATUS_ICON.get(row["status"], "?")
        lines.append(
            f"  {row['feature']:30}  {row['null_rate']:9.1%}  "
            f"{row['warn_above']:6.0%}  {row['structural_max']:5.0%}  "
            f"{icon} {row['status']}"
        )
    return "\n".join(lines)


def run_experiment(
    start_year: int = 2018,
    end_year: int = 2024,
    fetch_tennis_data: bool = True,
    save_model: bool = False,
) -> Dict:
    """
    End-to-end experiment. Returns results dict for downstream use.
    """
    # ── 1. Load + normalize ────────────────────────────────────────────────
    df_raw = load_and_normalize(start_year, end_year, fetch_tennis_data)

    # ── 2. Drop irregular formats ──────────────────────────────────────────
    df_raw = filter_irregular(df_raw)

    # ── 3. Engineer features ───────────────────────────────────────────────
    df = build_features(df_raw)

    # ── 4. Feature quality gate ────────────────────────────────────────────
    # Checks null rates against per-feature thresholds. Raises
    # FeatureQualityError if any feature is structurally broken (e.g. column
    # missing, join failure, minutes data absent). Logs warnings for features
    # that are sparse but within expected bounds (h2h, fatigue).
    #
    # This runs on the FULL dataset before splitting so a broken feature
    # can't silently slip into just the test set.
    logger.info("Running feature quality check...")
    quality_report = assert_feature_quality(df, context="full dataset")
    logger.info("\n" + _format_quality_report(quality_report))

    # ── 5. Split ───────────────────────────────────────────────────────────
    train, val, test = time_split(df)

    # Per-split quality checks — catches edge cases where one split has
    # dramatically different null rates (e.g. test set is a new year with
    # no historical surface data yet, or val split is very small).
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        assert_feature_quality(split_df, context=split_name)

    # Sanity check: no label imbalance (should be ~50/50 after A/B flip)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        pos_rate = split["label"].mean()
        logger.info(f"  {name} label balance: {pos_rate:.3f} (target ~0.5)")

    # ── 5. Grid search — baseline ──────────────────────────────────────────
    logger.info("\n── BASELINE (rank_ratio only) ──")
    best_C_base, pipe_base, gs_base = grid_search(
        train, val, BASELINE_FEATURES
    )

    # ── 6. Grid search — augmented ─────────────────────────────────────────
    logger.info("\n── AUGMENTED (rank + 6 features) ──")
    best_C_aug, pipe_aug, gs_aug = grid_search(
        train, val, AUGMENTED_FEATURES
    )

    # ── 7. Val metrics ─────────────────────────────────────────────────────
    logger.info("\n── VAL METRICS ──")
    val_base = eval_metrics(pipe_base, val[BASELINE_FEATURES].values,   val["label"].values, "val  baseline")
    val_aug  = eval_metrics(pipe_aug,  val[AUGMENTED_FEATURES].values,  val["label"].values, "val  augmented")

    # ── 8. Test metrics ────────────────────────────────────────────────────
    logger.info("\n── TEST METRICS (held-out 2024) ──")
    test_base = eval_metrics(pipe_base, test[BASELINE_FEATURES].values,  test["label"].values, "test baseline")
    test_aug  = eval_metrics(pipe_aug,  test[AUGMENTED_FEATURES].values, test["label"].values, "test augmented")

    results = {
        "train_n": len(train),
        "val_n":   len(val),
        "test_n":  len(test),
        "best_C_baseline":  best_C_base,
        "best_C_augmented": best_C_aug,
        "val_baseline":     val_base,
        "val_augmented":    val_aug,
        "test_baseline":    test_base,
        "test_augmented":   test_aug,
        "gs_baseline":      gs_base,
        "gs_augmented":     gs_aug,
        "pipe_baseline":    pipe_base,
        "pipe_augmented":   pipe_aug,
        "val_df":           val,
        "test_df":          test,
    }

    if save_model:
        import pickle
        out_dir = _REPO / "data" / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, pipe in [("baseline", pipe_base), ("augmented", pipe_aug)]:
            path = out_dir / f"model_{name}.pkl"
            with open(path, "wb") as f:
                pickle.dump(pipe, f)
            logger.info(f"Saved {path}")

    return results


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tennis Theo model")
    parser.add_argument("--no-network", action="store_true",
                        help="Skip tennis-data.co.uk fetch (use heuristic dates only)")
    parser.add_argument("--save-model", action="store_true",
                        help="Pickle fitted pipelines to data/processed/")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year",   type=int, default=2024)
    args = parser.parse_args()

    results = run_experiment(
        start_year=args.start_year,
        end_year=args.end_year,
        fetch_tennis_data=not args.no_network,
        save_model=args.save_model,
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  Train rows : {results['train_n']:,}")
    print(f"  Val rows   : {results['val_n']:,}")
    print(f"  Test rows  : {results['test_n']:,}")
    print()
    for split_name in ("val", "test"):
        print(f"  {split_name.upper()} RESULTS")
        for model_name in ("baseline", "augmented"):
            m = results[f"{split_name}_{model_name}"]
            print(f"    {model_name:10}  acc={m['accuracy']:.4f}  "
                  f"ll={m['logloss']:.4f}  brier={m['brier']:.4f}  "
                  f"auc={m['auc']:.4f}")
        print()
    delta_val  = results["val_augmented"]["accuracy"]  - results["val_baseline"]["accuracy"]
    delta_test = results["test_augmented"]["accuracy"] - results["test_baseline"]["accuracy"]
    print(f"  Accuracy lift (val) : {delta_val:+.4f} ({delta_val*100:+.2f}%)")
    print(f"  Accuracy lift (test): {delta_test:+.4f} ({delta_test*100:+.2f}%)")
    print()
    print("  Next: run evaluate.py for calibration table + PnL simulation")
