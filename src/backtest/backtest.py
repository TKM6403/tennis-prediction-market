"""
End-to-end backtest runner for Kalshi vs Theo on 2026 challenger matches.

This script consolidates the fix for the tml_match_id alignment bug:
predictions are merged via composite key (date + tournament + name pair)
rather than via positional index, since compute_all() reorders rows during
feature computation.
"""
import sys; sys.path.insert(0, "/home/claude/tennis-prediction-market")
import pandas as pd, numpy as np


def _lastname(name: str) -> str:
    if not isinstance(name, str): return ""
    toks = name.strip().split()
    return toks[-1].lower() if toks else ""


def _composite_key(date, tournament, p1, p2):
    """Stable identifier for a match: (date, tournament, sorted lastname pair)."""
    a, b = sorted([_lastname(p1), _lastname(p2)])
    return (pd.Timestamp(date).date(), str(tournament).lower().strip(), a, b)


def merge_predictions(joined: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Merge model predictions onto joined Kalshi data via composite key.

    DO NOT merge on tml_match_id — compute_all() reorders rows so any
    integer ID assigned post-features won't align with a pre-features ID.
    """
    preds = preds.copy()
    preds["match_date"] = pd.to_datetime(preds["match_date"])
    preds["_key"] = preds.apply(
        lambda r: _composite_key(r["match_date"], r["tournament"],
                                  r["winner_name"], r["loser_name"]),
        axis=1,
    )
    joined = joined.copy()
    joined["_key"] = joined.apply(
        lambda r: _composite_key(r["tml_match_date"], r["tml_tournament"],
                                  r["player_a"], r["player_b"]),
        axis=1,
    )
    out = joined.drop(
        columns=["theo", "theo_winner", "winner_name", "loser_name"],
        errors="ignore",
    ).merge(
        preds[["_key", "theo_winner", "winner_name", "loser_name"]],
        on="_key", how="left",
    ).drop(columns=["_key"])

    # Convert TML-perspective theo (P winner wins) to Kalshi-perspective
    # (P player_a wins).
    out["theo"] = np.where(
        out["tml_player_a_won"], out["theo_winner"], 1 - out["theo_winner"]
    )
    return out


def kalshi_fee(price: float) -> float:
    """Taker fee per contract: 7% × p × (1 − p), charged at trade time."""
    return 0.07 * price * (1.0 - price)


def backtest(df: pd.DataFrame, threshold_func) -> pd.DataFrame:
    """
    Apply RE rule to each match. Buy YES at ask if theo - ask >= threshold.
    Buy NO at (1 - bid) if bid - theo >= threshold. Otherwise skip.

    Returns DataFrame of placed bets with PnL columns.
    """
    bets = []
    for _, row in df.iterrows():
        theo, ask, bid, result = row["theo"], row["yes_ask"], row["yes_bid"], row["resolution"]
        thresh = threshold_func(theo)
        edge_yes = theo - ask
        edge_no  = bid - theo

        if edge_yes >= thresh and edge_yes >= edge_no:
            p_paid, won = ask, (result == 1.0)
            side = "YES"
        elif edge_no >= thresh:
            p_paid, won = 1.0 - bid, (result == 0.0)
            side = "NO"
        else:
            continue

        payoff = (1.0 - p_paid) if won else -p_paid
        fee = kalshi_fee(p_paid)
        bets.append({
            "side": side, "p_paid": p_paid, "theo": theo,
            "edge": max(edge_yes, edge_no), "won": won,
            "gross": payoff, "fee": fee, "net": payoff - fee,
            "bucket": ("low" if theo < 0.30
                       else "high" if theo >= 0.70 else "mid"),
        })
    return pd.DataFrame(bets)


def report(name: str, bets: pd.DataFrame) -> None:
    if len(bets) == 0:
        print(f"\n{name}: no bets placed"); return
    n = len(bets)
    print(f"\n{name}:")
    print(f"  Bets:      {n} (YES={(bets['side']=='YES').sum()}, NO={(bets['side']=='NO').sum()})")
    print(f"  Win rate:  {bets['won'].mean():.4f}")
    print(f"  Gross:     ${bets['gross'].sum():+.2f}")
    print(f"  Fees:      ${bets['fee'].sum():.2f}")
    print(f"  Net PnL:   ${bets['net'].sum():+.2f}")
    print(f"  ROI/bet:   {bets['net'].sum()/n*100:+.2f}%")
    by_b = bets.groupby("bucket").agg(
        n=("won", "count"), wins=("won", "sum"),
        net=("net", "sum"), avg_edge=("edge", "mean"),
    )
    by_b["roi"] = by_b["net"] / by_b["n"]
    print(f"  By theo bucket:")
    print(by_b.to_string(float_format=lambda x: f"{x:.4f}"))
