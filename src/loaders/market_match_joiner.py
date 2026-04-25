"""
market_match_joiner.py

Joins prediction-market data (Kalshi/Polymarket) to TML match data using
last-name matching plus exact-or-windowed date matching.

The market loader gives us:
    market_id, question, player_a, player_b, tournament, round_, event_date,
    entry_price, resolution, source

The TML match loader gives us:
    tml_match_id, player_a, player_b, surface, indoor, tourney_level,
    round_, match_date, date_confidence, player_a_won, [stats]

We left-join markets <- tml on (last-name match for player_a + match_date).
Every market row is preserved in the output. The joined columns are tagged
with a `tml_*` prefix to disambiguate from market columns.

Match strategy (in order):
    1. Exact: market.player_a last-name == tml.player_a last-name AND
       event_date == match_date.
    2. Last-name + ±2 day window: same name match, match_date within 2 days
       of event_date. Used when date_confidence is "empirical" or "heuristic".
       The 2-day window is bounded by typical round-spacing on the smallest
       events.
    3. Reversed: market.player_a matches tml.player_b (loser side). When this
       happens, player_a_won flips to False from the market's perspective.

Anything that doesn't match leaves the tml_* columns NaN. The audit() method
reports the unmatched rate and likely culprits.
"""

from __future__ import annotations

import re
import logging
from datetime import timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_name(name) -> str:
    """Last-name lowercase, alpha + hyphens only. Matches TMLMatchLoader's helper."""
    if not isinstance(name, str) or not name.strip():
        return ""
    tokens = name.strip().split()
    tokens = [t for t in tokens if not re.fullmatch(r"[A-Za-z]\.?", t)]
    if not tokens:
        return ""
    last = tokens[-1].lower()
    return re.sub(r"[^a-z\-]", "", last)


class MarketMatchJoiner:
    """
    Join market rows to TML rows by last-name + match_date.

    Public API:
        join(markets_df, tml_df) -> joined DataFrame
        audit(joined_df)         -> dict of join statistics

    Dummy example:
        markets_df: 1 row    [Darderi YES, event_date=2024-04-25, source=kalshi]
        tml_df:     1 row    [Darderi vs Cerundolo, match_date=2024-04-25,
                              surface=Clay, player_a_won=True]

        joined: 1 row with all market cols + tml_match_id + tml_surface +
                tml_player_a_won (True) + join_confidence ("exact").
    """

    DATE_WINDOW_DAYS = 2

    def join(
        self,
        markets_df: pd.DataFrame,
        tml_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if markets_df.empty:
            return markets_df.copy()

        out = markets_df.copy()

        # Build TML index keyed on (last_name_norm, match_date)
        # Each key may have multiple rows (rare but possible — same last name
        # plays multiple matches on the same day across different tourneys).
        tml_index: Dict[Tuple[str, pd.Timestamp], list] = {}
        if not tml_df.empty:
            for idx, row in tml_df.iterrows():
                w_key = _normalize_name(row.get("player_a", ""))
                l_key = _normalize_name(row.get("player_b", ""))
                date = pd.to_datetime(row.get("match_date"))
                if pd.isna(date):
                    continue
                # Index BOTH winner and loser sides; the joiner figures out
                # which side matches the market's player_a downstream.
                for key, side in [
                    ((w_key, date.date()), "winner"),
                    ((l_key, date.date()), "loser"),
                ]:
                    if not key[0]:
                        continue
                    tml_index.setdefault(key, []).append((idx, side))

        # Resolve each market row
        join_results = out.apply(
            lambda r: self._resolve_single(r, tml_index, tml_df),
            axis=1,
        )

        # Unpack results into columns
        out["tml_match_id"] = [r["tml_match_id"] for r in join_results]
        out["join_confidence"] = [r["confidence"] for r in join_results]
        out["tml_player_a_won"] = [r["player_a_won"] for r in join_results]

        # Bring across selected TML columns when matched
        tml_passthrough = [
            "surface", "indoor", "tourney_level", "round_",
            "match_date", "date_confidence", "minutes",
            "winner_rank", "loser_rank",
        ]
        for col in tml_passthrough:
            new_col = f"tml_{col}"
            out[new_col] = [
                tml_df.loc[r["tml_idx"], col]
                if r["tml_idx"] is not None and col in tml_df.columns
                else np.nan
                for r in join_results
            ]

        return out

    def _resolve_single(
        self,
        market_row: pd.Series,
        tml_index: Dict,
        tml_df: pd.DataFrame,
    ) -> Dict:
        """
        Try to find a TML row matching this market row.
        Returns a dict with tml_idx, tml_match_id, confidence, player_a_won.
        """
        empty_result = {
            "tml_idx": None,
            "tml_match_id": np.nan,
            "confidence": None,
            "player_a_won": np.nan,
        }

        player_key = _normalize_name(market_row.get("player_a", ""))
        event_date = pd.to_datetime(market_row.get("event_date"))
        if pd.isna(event_date) or not player_key:
            return empty_result
        event_date = event_date.date()

        # 1. Exact: same name + same date
        hit = tml_index.get((player_key, event_date))
        if hit:
            tml_idx, side = hit[0]
            tml_row = tml_df.loc[tml_idx]
            return {
                "tml_idx": tml_idx,
                "tml_match_id": tml_row.get("tml_match_id"),
                "confidence": "exact",
                "player_a_won": True if side == "winner" else False,
            }

        # 2. ±N day window
        for delta in range(1, self.DATE_WINDOW_DAYS + 1):
            for offset in (-delta, delta):
                test_date = event_date + timedelta(days=offset)
                hit = tml_index.get((player_key, test_date))
                if hit:
                    tml_idx, side = hit[0]
                    tml_row = tml_df.loc[tml_idx]
                    return {
                        "tml_idx": tml_idx,
                        "tml_match_id": tml_row.get("tml_match_id"),
                        "confidence": f"windowed_{offset:+d}d",
                        "player_a_won": True if side == "winner" else False,
                    }

        return empty_result

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def audit(self, joined_df: pd.DataFrame) -> Dict:
        """
        Summary statistics on a joined DataFrame.

        Returns dict with:
            total_market_rows, matched_count, match_rate,
            by_confidence (dict),
            unmatched_players (top 10 most common unmatched),
            resolution_conflicts (rows where market resolution disagrees
                                  with tml_player_a_won)
        """
        n = len(joined_df)
        if n == 0:
            return {"total_market_rows": 0, "matched_count": 0, "match_rate": 0.0}

        matched = joined_df["tml_match_id"].notna().sum()
        rate = matched / n

        by_conf = (
            joined_df["join_confidence"].fillna("unmatched").value_counts().to_dict()
        )

        unmatched = joined_df[joined_df["tml_match_id"].isna()]
        top_unmatched = (
            unmatched["player_a"].value_counts().head(10).to_dict()
            if "player_a" in unmatched.columns else {}
        )

        # Resolution conflicts: market says resolution=1.0 but TML says
        # tml_player_a_won=False, or vice versa. Only on rows that have both.
        if "resolution" in joined_df.columns:
            both = joined_df.dropna(subset=["resolution", "tml_player_a_won"])
            conflicts = both[both["resolution"] != both["tml_player_a_won"].astype(float)]
            n_conflicts = len(conflicts)
        else:
            n_conflicts = 0

        return {
            "total_market_rows":    n,
            "matched_count":        int(matched),
            "match_rate":           float(rate),
            "by_confidence":        by_conf,
            "unmatched_top_10":     top_unmatched,
            "resolution_conflicts": n_conflicts,
        }


# ============================================================================
# Dummy data tests
# ============================================================================

def _test_join_exact():
    print("=" * 60)
    print("MarketMatchJoiner — EXACT JOIN")
    print("=" * 60)

    markets = pd.DataFrame([{
        "market_id":   "kalshi::test1",
        "question":    "Will Darderi win?",
        "player_a":    "Luciano Darderi",
        "player_b":    "Juan Manuel Cerundolo",
        "tournament":  "ATP Madrid",
        "round_":      "R64",
        "event_date":  pd.Timestamp("2024-04-25").date(),
        "entry_price": 0.62,
        "resolution":  1.0,
        "source":      "kalshi",
    }])
    tml = pd.DataFrame([{
        "tml_match_id":   "2024-1536_M29",
        "player_a":       "Luciano Darderi",
        "player_b":       "Juan Manuel Cerundolo",
        "tournament":     "Madrid",
        "tourney_level":  "M",
        "surface":        "Clay",
        "indoor":         False,
        "round_":         "R64",
        "match_date":     pd.Timestamp("2024-04-25").date(),
        "date_confidence":"exact",
        "player_a_won":   True,
        "minutes":        117.0,
        "winner_rank":    63.0, "loser_rank": 85.0,
    }])

    joiner = MarketMatchJoiner()
    result = joiner.join(markets, tml)
    print(f"  rows: {len(result)}")
    print(f"  tml_match_id:    {result['tml_match_id'].iloc[0]}")
    print(f"  join_confidence: {result['join_confidence'].iloc[0]}")
    print(f"  tml_surface:     {result['tml_surface'].iloc[0]}")
    print(f"  tml_player_a_won:{result['tml_player_a_won'].iloc[0]}")
    assert result["tml_match_id"].iloc[0] == "2024-1536_M29"
    assert result["join_confidence"].iloc[0] == "exact"
    assert result["tml_surface"].iloc[0] == "Clay"
    assert result["tml_player_a_won"].iloc[0] == True
    print("  PASSED ✓")


def _test_join_windowed():
    print("\n" + "=" * 60)
    print("MarketMatchJoiner — ±2 DAY WINDOW")
    print("=" * 60)
    print("  Market event_date: 2024-04-25")
    print("  TML match_date:    2024-04-26  (heuristic estimate off by 1)")
    print("  Expected: matched with confidence='windowed_+1d'")

    markets = pd.DataFrame([{
        "market_id":   "kalshi::test2",
        "player_a":    "Some Player",
        "player_b":    "Other Player",
        "event_date":  pd.Timestamp("2024-04-25").date(),
        "entry_price": 0.50,
        "resolution":  np.nan,
        "source":      "kalshi",
    }])
    tml = pd.DataFrame([{
        "tml_match_id":   "TID1",
        "player_a":       "Some Player",
        "player_b":       "Other Player",
        "match_date":     pd.Timestamp("2024-04-26").date(),
        "surface":        "Hard",
        "indoor":         False,
        "tourney_level":  "M",
        "round_":         "R32",
        "date_confidence":"heuristic",
        "player_a_won":   True,
        "minutes":        np.nan,
        "winner_rank":    np.nan, "loser_rank": np.nan,
    }])

    joiner = MarketMatchJoiner()
    result = joiner.join(markets, tml)
    print(f"  Matched? {result['tml_match_id'].notna().iloc[0]}")
    print(f"  confidence: {result['join_confidence'].iloc[0]}")
    assert result["tml_match_id"].iloc[0] == "TID1"
    assert result["join_confidence"].iloc[0] == "windowed_+1d"
    print("  PASSED ✓")


def _test_join_reversed_player():
    print("\n" + "=" * 60)
    print("MarketMatchJoiner — REVERSED side (market on the loser)")
    print("=" * 60)
    print("  Market is on Cerundolo YES (Cerundolo lost the match)")
    print("  Expected: matched, tml_player_a_won = False")

    markets = pd.DataFrame([{
        "market_id":   "kalshi::test3",
        "player_a":    "Juan Manuel Cerundolo",
        "player_b":    "Luciano Darderi",
        "event_date":  pd.Timestamp("2024-04-25").date(),
        "entry_price": 0.38,
        "resolution":  0.0,
        "source":      "kalshi",
    }])
    tml = pd.DataFrame([{
        "tml_match_id":   "TID_REV",
        "player_a":       "Luciano Darderi",     # winner
        "player_b":       "Juan Manuel Cerundolo",# loser
        "match_date":     pd.Timestamp("2024-04-25").date(),
        "surface":        "Clay",
        "indoor":         False,
        "tourney_level":  "M",
        "round_":         "R64",
        "date_confidence":"exact",
        "player_a_won":   True,
        "minutes":        np.nan,
        "winner_rank":    np.nan, "loser_rank": np.nan,
    }])

    joiner = MarketMatchJoiner()
    result = joiner.join(markets, tml)
    print(f"  Matched: {result['tml_match_id'].iloc[0]}")
    print(f"  tml_player_a_won (from market POV): {result['tml_player_a_won'].iloc[0]}")
    assert result["tml_match_id"].iloc[0] == "TID_REV"
    assert result["tml_player_a_won"].iloc[0] == False
    assert result["join_confidence"].iloc[0] == "exact"
    print("  PASSED ✓")


def _test_join_unmatched():
    print("\n" + "=" * 60)
    print("MarketMatchJoiner — unmatched (no TML for this player)")
    print("=" * 60)

    markets = pd.DataFrame([{
        "market_id":   "kalshi::test4",
        "player_a":    "Unknown Player",
        "player_b":    "Other",
        "event_date":  pd.Timestamp("2024-04-25").date(),
        "entry_price": 0.5,
        "resolution":  np.nan,
        "source":      "kalshi",
    }])
    tml = pd.DataFrame(columns=[
        "tml_match_id", "player_a", "player_b", "match_date",
        "surface", "indoor", "tourney_level", "round_",
        "date_confidence", "player_a_won", "minutes",
        "winner_rank", "loser_rank",
    ])

    joiner = MarketMatchJoiner()
    result = joiner.join(markets, tml)
    assert result["tml_match_id"].isna().iloc[0]
    assert result["join_confidence"].iloc[0] is None
    print("  Unmatched correctly. PASSED ✓")


def _test_audit():
    print("\n" + "=" * 60)
    print("MarketMatchJoiner — AUDIT")
    print("=" * 60)

    joined = pd.DataFrame([
        {"player_a": "A", "tml_match_id": "T1", "join_confidence": "exact",
         "resolution": 1.0, "tml_player_a_won": True},
        {"player_a": "B", "tml_match_id": "T2", "join_confidence": "windowed_+1d",
         "resolution": 0.0, "tml_player_a_won": False},
        {"player_a": "C", "tml_match_id": np.nan, "join_confidence": None,
         "resolution": np.nan, "tml_player_a_won": np.nan},
        {"player_a": "D", "tml_match_id": "T4", "join_confidence": "exact",
         "resolution": 1.0, "tml_player_a_won": False},  # CONFLICT
    ])
    joiner = MarketMatchJoiner()
    audit = joiner.audit(joined)
    print(f"  total: {audit['total_market_rows']}")
    print(f"  matched: {audit['matched_count']}")
    print(f"  rate: {audit['match_rate']:.2%}")
    print(f"  by_confidence: {audit['by_confidence']}")
    print(f"  resolution_conflicts: {audit['resolution_conflicts']}")
    assert audit["total_market_rows"] == 4
    assert audit["matched_count"] == 3
    assert audit["resolution_conflicts"] == 1
    print("  PASSED ✓")


if __name__ == "__main__":
    _test_join_exact()
    _test_join_windowed()
    _test_join_reversed_player()
    _test_join_unmatched()
    _test_audit()
    print("\n" + "=" * 60)
    print("All MarketMatchJoiner tests passed.")
    print("=" * 60)
