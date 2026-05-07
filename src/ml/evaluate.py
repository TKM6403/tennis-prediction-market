"""
evaluate.py

Post-training evaluation for the tennis Theo model.

WHAT THIS DOES
--------------
Loads a fitted pipeline + test set (from train.py results), then:

1. CALIBRATION TABLE — bins predicted probabilities into 10 buckets,
   reports actual win rate vs mean predicted probability per bucket.
   A well-calibrated model has actual ≈ predicted in every bucket.

2. PnL SIMULATION — simulates betting at 5-cent edge:
   For each match, if |theo - market_mid| >= 0.05, place a $1 bet.
   Reports total bets placed, win rate, total PnL, edge retention.

   NOTE: The PnL sim runs against our OWN Theo probabilities as a
   self-consistency check (not real Kalshi prices) until we have the
   MarketMatchJoiner plumbed in. It answers: "if we bet where our
   model is most confident and the model is calibrated, do we make
   money?" Against a random market price from [0.3, 0.7], a calibrated
   model should show positive EV.

USAGE
-----
    # After running train.py with --save-model:
    python -m src.ml.evaluate

    # Or import directly:
    from src.ml.evaluate import run_evaluation
    run_evaluation(results)   # results dict from train.run_experiment()

CALIBRATION INTERPRETATION
---------------------------
Each row of the calibration table:
    bucket        predicted prob range
    n             # matches in bucket
    mean_pred     mean predicted probability
    actual_rate   actual win rate (fraction player_a won)
    gap           actual_rate - mean_pred  (positive = under-confident)
    |gap|         absolute calibration error

A model is well-calibrated if |gap| < 0.03 in every bucket with n > 50.
Large positive gap in high-prob buckets = model too conservative.
Large negative gap = model too aggressive (overconfident).
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Calibration table
# ============================================================================

def calibration_table(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Build a calibration table with `n_bins` equal-width probability buckets.

    Args:
        probs:   Predicted probabilities for player_a winning. Shape (n,).
        labels:  Actual outcomes. 1 = player_a won. Shape (n,).
        n_bins:  Number of probability buckets. Default 10 (each 10pp wide).

    Returns:
        DataFrame with columns:
            bucket       str, e.g. "[0.4, 0.5)"
            n            int, # matches in bucket
            mean_pred    float, mean predicted prob in bucket
            actual_rate  float, actual win rate
            gap          float, actual_rate - mean_pred
            abs_gap      float, |gap|

    Dummy example:
        10 matches predicted at 0.8 → 8 actually win.
        Bucket "[0.7, 0.8)": mean_pred=0.80, actual_rate=0.80, gap=0.00.
    """
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
        n = mask.sum()
        if n == 0:
            rows.append({
                "bucket": f"[{lo:.1f}, {hi:.1f})",
                "n": 0, "mean_pred": np.nan,
                "actual_rate": np.nan, "gap": np.nan, "abs_gap": np.nan,
            })
            continue
        mean_pred   = float(probs[mask].mean())
        actual_rate = float(labels[mask].mean())
        gap         = actual_rate - mean_pred
        rows.append({
            "bucket":      f"[{lo:.1f}, {hi:.1f})",
            "n":           int(n),
            "mean_pred":   round(mean_pred, 4),
            "actual_rate": round(actual_rate, 4),
            "gap":         round(gap, 4),
            "abs_gap":     round(abs(gap), 4),
        })
    return pd.DataFrame(rows)


def print_calibration_table(table: pd.DataFrame, title: str = "") -> None:
    """Print the calibration table in a readable format."""
    if title:
        print(f"\n{title}")
        print("─" * len(title))
    header = f"  {'bucket':14}  {'n':>5}  {'mean_pred':>9}  {'actual_rate':>11}  {'gap':>7}  {'|gap|':>6}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for _, row in table.iterrows():
        if row["n"] == 0:
            continue
        flag = "  ◄ WARN" if row["abs_gap"] > 0.04 and row["n"] >= 30 else ""
        print(
            f"  {row['bucket']:14}  {row['n']:5d}  {row['mean_pred']:9.4f}  "
            f"{row['actual_rate']:11.4f}  {row['gap']:+7.4f}  {row['abs_gap']:6.4f}{flag}"
        )
    filled = table[table["n"] > 0]
    if len(filled):
        ece = (filled["abs_gap"] * filled["n"]).sum() / filled["n"].sum()
        print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")


# ============================================================================
# PnL simulation
# ============================================================================

def pnl_simulation(
    probs: np.ndarray,
    labels: np.ndarray,
    market_probs: Optional[np.ndarray] = None,
    edge_threshold: float = 0.05,
    bet_size: float = 1.0,
) -> Dict:
    """
    Simulate flat-bet PnL at a given edge threshold.

    We bet on player_a when theo - market > edge_threshold, and bet on
    player_b when market - theo > edge_threshold.

    When market_probs is None (no real Kalshi data yet), we simulate a
    market by drawing uniform noise in [-0.15, +0.15] around our Theo.
    This tests model self-consistency: a perfectly calibrated model
    with enough edge should make money even against random noise.

    Args:
        probs:            Theo predicted prob of player_a winning.
        labels:           Actual outcome (1 = player_a won).
        market_probs:     Market mid price for player_a. None = simulate.
        edge_threshold:   Minimum edge to place a bet.
        bet_size:         Flat bet size in $ per trade.

    Returns:
        Dict with: n_bets, n_bet_a, n_bet_b, win_rate, total_pnl,
                   avg_edge, edge_retention, roi
    """
    rng = np.random.default_rng(99)

    if market_probs is None:
        # Simulate market: add noise to theo so we get realistic spread of edges
        noise = rng.uniform(-0.15, 0.15, size=len(probs))
        market_probs = np.clip(probs + noise, 0.05, 0.95)

    edge_a = probs - market_probs          # positive = we like player_a
    edge_b = market_probs - probs          # positive = we like player_b

    # Bet on a when edge_a > threshold
    bet_a = edge_a > edge_threshold
    # Bet on b when edge_b > threshold
    bet_b = edge_b > edge_threshold

    n_bet_a = bet_a.sum()
    n_bet_b = bet_b.sum()
    n_bets = n_bet_a + n_bet_b

    if n_bets == 0:
        return {
            "n_bets": 0, "n_bet_a": 0, "n_bet_b": 0,
            "win_rate": np.nan, "total_pnl": 0.0,
            "avg_edge": np.nan, "edge_retention": np.nan, "roi": np.nan,
        }

    # PnL for bets on player_a:
    # If player_a wins (label=1): win = bet_size * (1/market_probs - 1)  [decimal odds minus 1]
    # If player_a loses (label=0): lose = -bet_size
    # We approximate fair decimal odds from market_probs:
    #   odds_a = 1 / market_probs   (no vig for simplicity — conservative)
    pnl_a_wins  =  bet_size * (1.0 / market_probs[bet_a] - 1.0)  # win
    pnl_a_loses = -bet_size * np.ones(bet_a.sum())                 # lose

    won_a   = labels[bet_a] == 1
    pnl_a   = np.where(won_a, pnl_a_wins, pnl_a_loses)

    # PnL for bets on player_b:
    pnl_b_wins  =  bet_size * (1.0 / (1 - market_probs[bet_b]) - 1.0)
    pnl_b_loses = -bet_size * np.ones(bet_b.sum())

    won_b   = labels[bet_b] == 0  # player_b wins when label=0
    pnl_b   = np.where(won_b, pnl_b_wins, pnl_b_loses)

    all_pnl = np.concatenate([pnl_a, pnl_b])
    wins     = np.concatenate([won_a, won_b])

    total_pnl = float(all_pnl.sum())
    win_rate  = float(wins.mean())
    all_edges = np.concatenate([edge_a[bet_a], edge_b[bet_b]])
    avg_edge  = float(all_edges.mean())
    total_wagered = bet_size * n_bets
    roi       = total_pnl / total_wagered

    # Edge retention: how much of our theoretical edge we actually captured
    # Theoretical PnL if model is perfectly calibrated = sum of edges * bet_size
    theoretical_pnl = float(all_edges.sum()) * bet_size
    edge_retention  = total_pnl / theoretical_pnl if theoretical_pnl > 0 else np.nan

    return {
        "n_bets":          int(n_bets),
        "n_bet_a":         int(n_bet_a),
        "n_bet_b":         int(n_bet_b),
        "win_rate":        win_rate,
        "total_pnl":       round(total_pnl, 2),
        "avg_edge":        round(avg_edge, 4),
        "edge_retention":  round(edge_retention, 4) if not np.isnan(edge_retention) else np.nan,
        "roi":             round(roi, 4),
    }


def print_pnl_summary(sim: Dict, title: str = "") -> None:
    if title:
        print(f"\n{title}")
        print("─" * len(title))
    print(f"  Bets placed     : {sim['n_bets']:,} ({sim['n_bet_a']} on A, {sim['n_bet_b']} on B)")
    if sim["n_bets"] == 0:
        print("  No bets placed (edge threshold not met)")
        return
    print(f"  Win rate        : {sim['win_rate']:.4f}")
    print(f"  Total PnL       : ${sim['total_pnl']:+,.2f}  (${sim['total_pnl']/sim['n_bets']:+.4f}/bet)")
    print(f"  ROI             : {sim['roi']:+.4f}  ({sim['roi']*100:+.2f}%)")
    print(f"  Avg edge taken  : {sim['avg_edge']:.4f}")
    if not np.isnan(sim["edge_retention"]):
        print(f"  Edge retention  : {sim['edge_retention']:.4f}  ({sim['edge_retention']*100:.1f}% of theo edge captured)")


# ============================================================================
# Calibration plot
# ============================================================================

def plot_calibration(
    tables: Dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot calibration curves for multiple models on the same axes.

    Args:
        tables:     {model_name: calibration_table_df}
        save_path:  If given, save figure here instead of showing.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    colors = {"baseline": "#e74c3c", "augmented": "#2ecc71"}
    for name, table in tables.items():
        t = table[table["n"] > 0].dropna()
        color = colors.get(name, "#3498db")
        ax.scatter(t["mean_pred"], t["actual_rate"],
                   label=name, color=color, s=t["n"] / t["n"].max() * 200,
                   alpha=0.8, zorder=5)
        ax.plot(t["mean_pred"], t["actual_rate"], color=color, lw=1.5, alpha=0.6)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual win rate")
    ax.set_title("Calibration curves (test set 2024)\nMarker size ∝ bucket count")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info(f"Calibration plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ============================================================================
# Main evaluation runner
# ============================================================================

def run_evaluation(results: Dict, edge_threshold: float = 0.05) -> Dict:
    """
    Run the full calibration + PnL evaluation given a results dict from
    train.run_experiment().

    Args:
        results:         Output of train.run_experiment().
        edge_threshold:  Minimum |theo - market| to place a simulated bet.

    Returns:
        Dict with calibration tables and PnL sim results for both models.
    """
    from src.ml.train import BASELINE_FEATURES, AUGMENTED_FEATURES

    test_df   = results["test_df"]
    pipe_base = results["pipe_baseline"]
    pipe_aug  = results["pipe_augmented"]

    labels = test_df["label"].values

    probs_base = pipe_base.predict_proba(test_df[BASELINE_FEATURES].values)[:, 1]
    probs_aug  = pipe_aug.predict_proba(test_df[AUGMENTED_FEATURES].values)[:, 1]

    # ── Calibration tables ─────────────────────────────────────────────────
    cal_base = calibration_table(probs_base, labels)
    cal_aug  = calibration_table(probs_aug,  labels)

    print_calibration_table(cal_base, "CALIBRATION — Baseline (rank only) — Test 2024")
    print_calibration_table(cal_aug,  "CALIBRATION — Augmented (all features) — Test 2024")

    # ── PnL simulation ─────────────────────────────────────────────────────
    rng = np.random.default_rng(42)

    # Use same simulated market prices for fair comparison
    noise = rng.uniform(-0.15, 0.15, size=len(probs_base))
    market_probs_base = np.clip(probs_base + noise, 0.05, 0.95)
    market_probs_aug  = np.clip(probs_aug  + noise, 0.05, 0.95)

    print(f"\nPnL SIMULATION — edge threshold = {edge_threshold:.2f} — $1 flat bets")
    print("NOTE: Market prices are simulated (±15% noise around Theo).")
    print("Real Kalshi prices will be plumbed in via MarketMatchJoiner.")
    print()

    sim_base = pnl_simulation(probs_base, labels, market_probs_base, edge_threshold)
    sim_aug  = pnl_simulation(probs_aug,  labels, market_probs_aug,  edge_threshold)

    print_pnl_summary(sim_base, f"PnL — Baseline — Test 2024 (n_test={len(labels):,})")
    print_pnl_summary(sim_aug,  f"PnL — Augmented — Test 2024 (n_test={len(labels):,})")

    # ── Save calibration plot ──────────────────────────────────────────────
    cal_method = results.get("cal_method", "isotonic")
    suffix = f"_{cal_method}"
    out_dir = _REPO / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_calibration(
        {"baseline": cal_base, "augmented": cal_aug},
        save_path=out_dir / f"calibration_plot{suffix}.png",
    )

    # Save calibration tables to CSV
    for name, table in [("baseline", cal_base), ("augmented", cal_aug)]:
        path = out_dir / f"calibration_{name}{suffix}.csv"
        table.to_csv(path, index=False)
        logger.info(f"Calibration table saved → {path}")

    # One-line summary of headline metrics — easy to grep across runs.
    def _summary(probs, labels):
        return {
            "log_loss": float(log_loss(labels, probs)),
            "brier":    float(brier_score_loss(labels, probs)),
        }
    print(f"\n── HEADLINE METRICS (cal_method={cal_method}) ──")
    base_h = _summary(probs_base, labels)
    aug_h  = _summary(probs_aug,  labels)
    base_ece = (cal_base.dropna()["abs_gap"] * cal_base.dropna()["n"]).sum() / cal_base.dropna()["n"].sum()
    aug_ece  = (cal_aug.dropna()["abs_gap"]  * cal_aug.dropna()["n"]).sum()  / cal_aug.dropna()["n"].sum()
    print(f"  baseline:  log_loss={base_h['log_loss']:.4f}  brier={base_h['brier']:.4f}  ECE={base_ece:.4f}")
    print(f"  augmented: log_loss={aug_h['log_loss']:.4f}  brier={aug_h['brier']:.4f}  ECE={aug_ece:.4f}")

    return {
        "cal_baseline":  cal_base,
        "cal_augmented": cal_aug,
        "pnl_baseline":  sim_base,
        "pnl_augmented": sim_aug,
    }


# ============================================================================
# Entry point — load saved models and run evaluation
# ============================================================================

if __name__ == "__main__":
    import pickle

    model_dir = _REPO / "data" / "processed"
    paths = {
        "baseline":  model_dir / "model_baseline.pkl",
        "augmented": model_dir / "model_augmented.pkl",
    }

    # Check if saved models exist; if not, re-run train
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        print(f"Saved models not found ({missing}). Running train.run_experiment()...")
        from src.ml.train import run_experiment
        results = run_experiment(save_model=True)
    else:
        print("Loading saved models from data/processed/...")
        from src.ml.train import run_experiment
        # Still need to get the test DF — re-run training with no-network if needed
        results = run_experiment(save_model=False)

    eval_results = run_evaluation(results)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("\nFiles written to data/processed/:")
    print("  calibration_baseline.csv")
    print("  calibration_augmented.csv")
    print("  calibration_plot.png")
    print("\nNext: plug in real Kalshi market prices via MarketMatchJoiner")
    print("      for a real PnL simulation against live market prices.")
