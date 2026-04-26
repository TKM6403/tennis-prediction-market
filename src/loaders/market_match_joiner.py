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


def _normalize_tournament(name) -> str:
    """
    Normalize a tournament name for joining across Kalshi and TML.

    Lowercase, strip whitespace, drop punctuation. Performs three additional
    normalizations to handle naming differences between the two sources:

      1. Roman numerals → arabic. Kalshi uses "Tigre II" but TML uses
         "Tigre 2". We convert II→2, III→3, IV→4 (only seen in practice).
      2. Strip "Qualification" suffix. Kalshi treats qualifying-round
         markets as a separate tournament ("Sao Paulo Qualification") but
         TML rolls them into the main draw ("Sao Paulo") with a different
         round_ label (Q1/Q2/Q3).
      3. Strip lone trailing " 1". TML calls the first edition of a
         repeat-tournament without a number suffix ("Kigali"), Kalshi adds
         the explicit "Kigali 1". Numbers ≥2 are preserved.

    Tour-level prefixes (ATP, WTA, Challenger) are also stripped if they
    leak through — defensive, since KalshiLoader strips them upstream.
    """
    if not isinstance(name, str) or not name.strip():
        return ""
    s = name.strip().lower()

    # Strip tour/level prefixes
    for prefix in ("atp challenger ", "wta challenger ", "atp ", "wta ",
                   "challenger "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break

    # Strip "qualification" / "qualifying" suffix
    for suffix in (" qualification", " qualifying", " qualifier"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break

    # Drop punctuation
    s = re.sub(r"[^\w\s\-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Roman numeral → arabic (only standalone trailing tokens — don't
    # touch words like "Mineral I.X." if any)
    ROMAN_MAP = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5"}
    tokens = s.split()
    if tokens and tokens[-1] in ROMAN_MAP:
        tokens[-1] = ROMAN_MAP[tokens[-1]]
        s = " ".join(tokens)

    # Strip lone trailing " 1" — TML names the first edition without a number
    if s.endswith(" 1"):
        s = s[:-2]

    return s.strip() or ""


class MarketMatchJoiner:
    """
    Join market rows to TML rows by **both player names + tournament**.

    Public API:
        join(markets_df, tml_df) -> joined DataFrame
        audit(joined_df)         -> dict of join statistics

    JOIN KEY
    --------
    A composite key built from:
      - lastname(market.player_a) and lastname(market.player_b)  — sorted
      - normalized tournament name (case-insensitive, prefix-stripped)

    Sorting the player last names makes the key order-independent: a Kalshi
    market for "Cerundolo vs Darderi" matches a TML row regardless of which
    player TML calls winner_name. Once matched, we determine which side
    Kalshi's player_a maps to (winner or loser) and set tml_player_a_won
    accordingly.

    Date is used as a tiebreaker when multiple TML rows share the same key
    (same two players in the same tournament — happens with rematches in
    later rounds). We pick the TML row whose match_date is closest to
    event_date, within ±DATE_WINDOW_DAYS.

    WHY THIS BEATS LAST-NAME-ONLY MATCHING
    ---------------------------------------
    Last-name-only had ~13% resolution conflicts: a Kalshi market for
    "Denolly vs Schoolkate" on 2026-04-01 would accidentally match an
    unrelated Denolly match on a different tournament. Requiring both
    player names + tournament eliminates these false positives.

    Dummy example:
        markets_df: 1 row    [player_a="Luciano Darderi",
                              player_b="Cerundolo",
                              tournament="Madrid",
                              event_date=2026-04-25,
                              resolution=1.0,
                              source="kalshi"]
        tml_df:     1 row    [player_a="Luciano Darderi" (winner),
                              player_b="Francisco Cerundolo" (loser),
                              tournament="Madrid",
                              match_date=2026-04-25, surface="Clay"]

        joined: 1 row with all market cols + tml_match_id +
                tml_player_a_won=True (since Darderi is winner) +
                join_confidence="exact".
    """

    DATE_WINDOW_DAYS = 3

    def join(
        self,
        markets_df: pd.DataFrame,
        tml_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Match each market row to a TML row using the composite key
        (sorted player lastnames + tournament).

        Date is used as a tiebreaker only — same two players + same
        tournament uniquely identifies a match in 99%+ of cases.
        """
        if markets_df.empty:
            return markets_df.copy()

        out = markets_df.copy()

        # Build TML index keyed on (sorted_lastnames_pair, tournament_norm)
        # Each key value stores the list of (idx, winner_lastname, loser_lastname,
        # match_date) tuples — usually one, occasionally many for tournaments
        # where the same pair plays multiple rounds.
        tml_index: Dict[Tuple[Tuple[str, str], str], list] = {}
        if not tml_df.empty:
            for idx, row in tml_df.iterrows():
                w_last = _normalize_name(row.get("player_a", ""))
                l_last = _normalize_name(row.get("player_b", ""))
                tour   = _normalize_tournament(row.get("tournament", ""))
                if not w_last or not l_last or not tour:
                    continue
                pair_key = tuple(sorted([w_last, l_last]))
                key = (pair_key, tour)
                date = pd.to_datetime(row.get("match_date"), errors="coerce")
                tml_index.setdefault(key, []).append(
                    (idx, w_last, l_last, date)
                )

        # Resolve each market row
        join_results = out.apply(
            lambda r: self._resolve_single(r, tml_index, tml_df),
            axis=1,
        )

        # Unpack results into columns
        out["tml_match_id"]     = [r["tml_match_id"]    for r in join_results]
        out["join_confidence"]  = [r["confidence"]      for r in join_results]
        out["tml_player_a_won"] = [r["player_a_won"]    for r in join_results]

        # Bring across selected TML columns when matched
        tml_passthrough = [
            "surface", "indoor", "tourney_level", "round_",
            "match_date", "date_confidence", "minutes",
            "winner_rank", "loser_rank", "tournament",
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
        Find the TML row matching this market row using:
            (sorted player lastnames pair, tournament) → exact key
            then date proximity within ±DATE_WINDOW_DAYS as tiebreaker.

        Returns dict: tml_idx, tml_match_id, confidence, player_a_won.
        """
        empty_result = {
            "tml_idx":      None,
            "tml_match_id": np.nan,
            "confidence":   None,
            "player_a_won": np.nan,
        }

        a_last = _normalize_name(market_row.get("player_a", ""))
        b_last = _normalize_name(market_row.get("player_b", ""))
        tour   = _normalize_tournament(market_row.get("tournament", ""))
        event_dt = pd.to_datetime(market_row.get("event_date"), errors="coerce")

        if not a_last or not b_last or not tour or pd.isna(event_dt):
            return empty_result

        pair_key = tuple(sorted([a_last, b_last]))
        candidates = tml_index.get((pair_key, tour))
        if not candidates:
            return empty_result

        # Among candidates, pick the one whose match_date is closest to event_date
        # (within DATE_WINDOW_DAYS days). If multiple candidates exist (rare —
        # same two players in same tournament across multiple rounds), the
        # closest-date heuristic picks the right one.
        event_date = event_dt.date()
        best = None
        best_delta = None
        for tml_idx, w_last, l_last, m_date in candidates:
            if pd.isna(m_date):
                continue
            delta = abs((m_date.date() - event_date).days)
            if delta > self.DATE_WINDOW_DAYS:
                continue
            if best is None or delta < best_delta:
                best = (tml_idx, w_last, l_last, m_date)
                best_delta = delta

        if best is None:
            # We have a player+tournament match, but no row within date window.
            # Could be a rescheduled or canceled event. Mark as unmatched.
            return {
                **empty_result,
                "confidence": "no_date_match",
            }

        tml_idx, w_last, l_last, m_date = best
        # Determine if Kalshi's player_a is the winner side (TML's player_a)
        player_a_won = (a_last == w_last)
        confidence = "exact" if best_delta == 0 else f"windowed_{best_delta}d"

        return {
            "tml_idx":      tml_idx,
            "tml_match_id": tml_df.loc[tml_idx].get("tml_match_id"),
            "confidence":   confidence,
            "player_a_won": player_a_won,
        }

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
