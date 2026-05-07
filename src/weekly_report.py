"""
weekly_report.py

Reads data/paper_trades/settled.csv and dropped.csv, prints a digest
of how the paper trader is doing and what's driving (or sinking) PnL.

USAGE
-----
    python -m src.weekly_report                    # all-time
    python -m src.weekly_report --since 2026-04-15 # only bets recorded on/after
    python -m src.weekly_report --save             # also write a dated .txt copy

WHAT IT REPORTS
---------------
1. HEADLINE — n_bets, win_rate, total_pnl, ROI, ECE on settled bets.
   ECE here = "are my probabilities trustworthy on the bets I actually took?"
   Different from the train-time test-set ECE, which is over all matches.

2. DROP REASONS — histogram of why scan() rejected markets. If 95% are
   `outside_tails` you may be too strict; if 95% are `wide_spread` you
   may be scanning during off-hours.

3. SIGNED FEATURE ATTRIBUTION — for each feature in AUGMENTED_FEATURES:
       signed_share_i  = shift_i / Σⱼ |shift_j|         (∈ [-1, 1])
       feature_pnl_i   = signed_share_i × bet_pnl       (per bet)
       total_attr_pnl  = Σ feature_pnl_i over all bets
       per_$_conviction = total_attr_pnl / Σ |signed_share_i|

   Interpretation: positive total_pnl and per_$ = the feature *earns*
   money when it drives bets. Negative = it's noise or counter-signal.
   This is the diagnostic that tells you which features to keep, drop,
   or rebuild for v2 of the model.

NOTE
----
Feature shifts were computed and frozen at scan time (paper_trader.py
writes them as JSON into the bet row). Retraining the model later does
NOT rewrite historical attribution — that's intentional, so the report
reflects the model that actually placed each bet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "data" / "paper_trades"


# ============================================================================
# Loaders
# ============================================================================

def load_settled(since: Optional[str] = None) -> pd.DataFrame:
    path = LOG_DIR / "settled.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp_recorded" in df.columns:
        df["timestamp_recorded"] = pd.to_datetime(df["timestamp_recorded"], errors="coerce")
        if since is not None:
            df = df[df["timestamp_recorded"] >= pd.Timestamp(since)]
    return df.reset_index(drop=True)


def load_dropped(since: Optional[str] = None) -> pd.DataFrame:
    path = LOG_DIR / "dropped.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if since is not None:
            df = df[df["timestamp"] >= pd.Timestamp(since)]
    return df.reset_index(drop=True)


# ============================================================================
# Headline metrics
# ============================================================================

def _ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Bucketed expected calibration error. Empty → NaN."""
    if len(probs) == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    weighted_gap = 0.0
    total = 0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        gap = abs(outcomes[mask].mean() - probs[mask].mean())
        weighted_gap += gap * n
        total += n
    return float(weighted_gap / total) if total > 0 else float("nan")


def headline_metrics(settled: pd.DataFrame) -> dict:
    if settled.empty:
        return {"n_bets": 0}
    n = len(settled)
    n_won = int(settled["bet_won"].sum())
    pnl = float(settled["net_pnl"].sum())
    invested = float(settled["entry_price"].sum())  # 1 contract per bet → entry_price = exposure
    roi = pnl / invested if invested > 0 else float("nan")
    ece = _ece(settled["theo_chosen"].astype(float).values,
               settled["bet_won"].astype(int).values)
    return {
        "n_bets":         n,
        "n_won":          n_won,
        "win_rate":       n_won / n,
        "total_pnl":      pnl,
        "total_invested": invested,
        "roi":            roi,
        "ece":            ece,
    }


# ============================================================================
# Signed feature attribution
# ============================================================================

def signed_attribution(settled: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-feature signed PnL attribution across all settled bets.

    For each bet:
        signed_share_i = shift_i / Σⱼ |shift_j|
        feature_pnl_i  = signed_share_i × bet_pnl

    "Toward the bet" is already encoded at scan time — paper_trader uses
    the synthetic row whose winner_name == the player we're betting on, so
    positive shift = pushes toward the bet winning.
    """
    if settled.empty or "feature_shifts_json" not in settled.columns:
        return pd.DataFrame()

    rows = []
    for _, r in settled.iterrows():
        raw = r.get("feature_shifts_json")
        if not isinstance(raw, str) or not raw:
            continue
        try:
            shifts = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            continue
        total_abs = sum(abs(v) for v in shifts.values())
        if total_abs <= 0:
            continue
        pnl = float(r["net_pnl"])
        for feat, shift in shifts.items():
            signed_share = shift / total_abs    # ∈ [-1, 1], signed toward bet
            rows.append({
                "feature":        feat,
                "abs_share":      abs(signed_share),
                "attributed_pnl": signed_share * pnl,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    agg = df.groupby("feature").agg(
        avg_abs_share=("abs_share", "mean"),
        total_abs_conviction=("abs_share", "sum"),
        total_attributed_pnl=("attributed_pnl", "sum"),
    ).reset_index()
    agg["per_$_conviction"] = np.where(
        agg["total_abs_conviction"] > 0,
        agg["total_attributed_pnl"] / agg["total_abs_conviction"],
        np.nan,
    )
    return agg.sort_values("total_attributed_pnl", ascending=False).reset_index(drop=True)


# ============================================================================
# Report rendering
# ============================================================================

def render_report(settled: pd.DataFrame, dropped: pd.DataFrame,
                  since: Optional[str] = None) -> str:
    L = []
    L.append("=" * 72)
    title = "PAPER TRADE WEEKLY REPORT"
    if since:
        title += f"  (since {since})"
    L.append(title)
    L.append(f"  generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    L.append("=" * 72)

    h = headline_metrics(settled)
    if h.get("n_bets", 0) == 0:
        L.append("\nNo settled bets in window. Run scan() and wait for matches to resolve.")
    else:
        L.append("")
        L.append("HEADLINE")
        L.append(f"  n_bets         : {h['n_bets']}")
        L.append(f"  win_rate       : {h['win_rate']:.4f}  ({h['n_won']}/{h['n_bets']})")
        L.append(f"  total_pnl      : ${h['total_pnl']:+.2f}")
        L.append(f"  total_invested : ${h['total_invested']:.2f}")
        L.append(f"  ROI            : {h['roi']:+.4f}  ({h['roi']*100:+.2f}%)")
        L.append(f"  ECE on bets    : {h['ece']:.4f}  (lower → model trustworthy on chosen edges)")

    if not dropped.empty:
        L.append("")
        L.append(f"DROP REASONS  ({len(dropped):,} markets considered & rejected)")
        counts = dropped["reason"].value_counts()
        for reason, cnt in counts.items():
            L.append(f"  {reason:25s}  {cnt:>6,}")

    attr = signed_attribution(settled)
    if not attr.empty:
        L.append("")
        L.append("SIGNED FEATURE ATTRIBUTION")
        L.append("  Per-feature share of log-odds shift TOWARD the bet × realized PnL.")
        L.append("  +total_pnl = feature earns money. -total_pnl = noise/counter-signal.")
        L.append("  per_$_conviction normalizes by how often the feature drove bets.")
        L.append("")
        L.append(f"  {'feature':30s}  {'avg_|share|':>11}  {'total_pnl':>10}  {'per_$':>8}")
        L.append("  " + "-" * 64)
        for _, r in attr.iterrows():
            L.append(
                f"  {r['feature']:30s}  "
                f"{r['avg_abs_share']:>11.3f}  "
                f"{r['total_attributed_pnl']:>+10.2f}  "
                f"{r['per_$_conviction']:>+8.3f}"
            )
        # Σ over features won't equal headline PnL with SIGNED attribution
        # because counter-signal features partially cancel. The gap is itself
        # diagnostic: large gap = features often disagreed within winning bets.
        check_sum = attr["total_attributed_pnl"].sum()
        L.append("")
        L.append(f"  Σ attributed PnL across features = ${check_sum:+.2f}")
        L.append(f"  Headline total PnL                = ${h.get('total_pnl', 0):+.2f}")
        L.append("  (gap = how often features disagreed within the same bet)")

    return "\n".join(L)


# ============================================================================
# CLI
# ============================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Weekly paper-trade digest")
    parser.add_argument("--since", type=str, default=None,
                        help="ISO date — include only bets recorded on/after.")
    parser.add_argument("--save", action="store_true",
                        help="Also write data/paper_trades/weekly_report_YYYY-MM-DD.txt")
    args = parser.parse_args(argv)

    settled = load_settled(since=args.since)
    dropped = load_dropped(since=args.since)
    text = render_report(settled, dropped, since=args.since)
    print(text)

    if args.save:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = LOG_DIR / f"weekly_report_{pd.Timestamp.now().strftime('%Y-%m-%d')}.txt"
        out.write_text(text + "\n")
        print(f"\nSaved → {out}")
    return 0


# ============================================================================
# Smoke test (synthetic data)
# ============================================================================

def _smoke_test():
    """Build a tiny fake settled.csv and verify the report renders cleanly."""
    print("=" * 60)
    print("weekly_report smoke test (3 fake bets, 2 features)")
    print("=" * 60)
    settled = pd.DataFrame([
        {
            "timestamp_recorded": "2026-04-20T08:00:00",
            "entry_price": 0.55, "theo_chosen": 0.65, "bet_won": True,
            "net_pnl": 0.42,
            "feature_shifts_json": json.dumps({"feat_A": +0.8, "feat_B": +0.2}),
        },
        {
            "timestamp_recorded": "2026-04-21T08:00:00",
            "entry_price": 0.40, "theo_chosen": 0.50, "bet_won": False,
            "net_pnl": -0.42,
            "feature_shifts_json": json.dumps({"feat_A": +0.6, "feat_B": -0.4}),
        },
        {
            "timestamp_recorded": "2026-04-22T08:00:00",
            "entry_price": 0.30, "theo_chosen": 0.45, "bet_won": True,
            "net_pnl": 0.68,
            "feature_shifts_json": json.dumps({"feat_A": +0.1, "feat_B": +0.9}),
        },
    ])
    dropped = pd.DataFrame([
        {"timestamp": "2026-04-20", "reason": "below_min_edge"},
        {"timestamp": "2026-04-20", "reason": "below_min_edge"},
        {"timestamp": "2026-04-20", "reason": "outside_tails"},
    ])
    print(render_report(settled, dropped))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--smoke-test":
        _smoke_test()
    else:
        sys.exit(main())
