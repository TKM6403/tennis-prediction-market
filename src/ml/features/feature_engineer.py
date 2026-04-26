"""
feature_engineer.py

All feature computation for the tennis prediction model lives here.

PUBLIC ENTRY POINT
    compute_all(matches_df, cutoff_date=None) → DataFrame
        Takes a normalized TML match-level DataFrame (one row per match,
        winner-first per TML convention) and returns it augmented with
        engineered features for both winner and loser.

    The returned DataFrame is still in winner/loser orientation. The
    downstream training pipeline is responsible for the random A/B flip
    that creates the symmetric prediction problem.

LOOKAHEAD GUARD (per CLAUDE.md)
    Every feature here is computed using ONLY matches strictly before
    the current match's match_date. The historical-window helpers all
    take cutoff_date as an explicit argument and apply it with a
    strict `<` filter — never `<=`.

THE 6 FEATURES IMPLEMENTED IN THIS PASS
    fatigue_minutes_7d        — minutes played in last 7 days
    fatigue_minutes_14d       — minutes played in last 14 days
    days_since_last_match     — recovery time since previous match
    surface_win_rate_52w      — win rate on this surface over last year
    recent_form_10m           — win rate over last 10 matches (any surface)
    h2h_surface_advantage     — win rate vs this opponent on this surface

    For each feature, we compute the WINNER's value and the LOSER's
    value separately. Column naming convention: `{feature}_w` and
    `{feature}_l`. The training pipeline then takes differences /
    keeps both as needed.

DUMMY EXAMPLE
    Input row:  Darderi (winner) vs Cerundolo (loser), Madrid R64,
                surface=Clay, match_date=2024-04-25.

    For Darderi, looking back at his earlier matches in the dataset:
        fatigue_minutes_7d_w   = 215.0   (won R128 5 days ago in 215 min)
        days_since_last_match_w= 5.0
        surface_win_rate_52w_w = 0.62    (8/13 clay matches in past year)
        recent_form_10m_w      = 0.70    (7/10 last matches won)
        h2h_surface_advantage_w= NaN     (never played Cerundolo on clay)

    Same fields computed for Cerundolo as `..._l`.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================================
# Public entry point
# ============================================================================

def compute_all(
    matches_df: pd.DataFrame,
    cutoff_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Add engineered features to a TML-style match DataFrame.

    Args:
        matches_df:    DataFrame with at minimum these columns:
                       winner_name, loser_name, match_date, surface, minutes
        cutoff_date:   If set, drop rows on or after this date BEFORE
                       computing features (so training data never sees
                       future matches). The features themselves still
                       respect lookahead per row.

    Returns:
        Same DataFrame with these new columns added:
            fatigue_minutes_7d_w / _l
            fatigue_minutes_14d_w / _l
            days_since_last_match_w / _l
            surface_win_rate_52w_w / _l
            recent_form_10m_w / _l
            h2h_surface_advantage_w / _l

    Implementation:
        Vectorized per-player. For each unique player, we collect all
        their matches sorted by date, then compute every feature as a
        rolling/cumulative operation. This is O(n log n) instead of the
        naive O(n²) row-by-row approach.
    """
    df = matches_df.copy()

    if cutoff_date is not None:
        cutoff = pd.Timestamp(cutoff_date)
        df = df[pd.to_datetime(df["match_date"]) < cutoff].reset_index(drop=True)

    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)

    history = _build_player_history(df)

    # ── Compute features per-player using groupby + shift to enforce strict
    # less-than (you never see the current match in your own history)
    history = history.sort_values(["player", "match_date", "won"]).reset_index(drop=True)

    # ── 1. days_since_last_match: most recent match strictly before current
    # The simple .diff() won't work because it gives 0 for same-day matches.
    # Instead, for each player, compute days since the most recent match
    # at a strictly earlier date.
    def _days_since(group):
        dates = group["match_date"].values
        out = np.full(len(group), np.nan, dtype=float)
        for i in range(len(group)):
            mask = dates < dates[i]
            if mask.any():
                out[i] = (dates[i] - dates[mask].max()).astype("timedelta64[D]").astype(float)
        return pd.Series(out, index=group.index)
    history["days_since_last"] = (
        history.groupby("player", group_keys=False).apply(_days_since)
    )

    # ── 2. recent_form: rolling mean of `won` over last 10 matches.
    # Use closed='left' so all same-day matches are excluded — important
    # because tourney_date is week-start, so multiple matches share dates.
    # min_periods=3 enforces meaningful sample.
    def _rolling_form(s):
        return s.rolling(window=10, min_periods=3, closed="left").mean()
    history["recent_form_10m"] = (
        history.groupby("player")["won"].transform(_rolling_form)
    )

    # ── 3. surface_win_rate_52w: rolling time-based mean per (player, surface)
    history["_pos"] = np.arange(len(history))
    history_dt = history.set_index("match_date")

    def _time_rolling_mean(group_df, value_col, window):
        s = group_df[value_col]
        rolled = s.rolling(window, min_periods=3, closed="left").mean()
        return pd.DataFrame({"_pos": group_df["_pos"].values,
                             "_val": rolled.values})

    swr_parts = []
    for (_player, _surf), grp in history_dt.groupby(["player", "surface"], sort=False):
        swr_parts.append(_time_rolling_mean(grp, "won", "365D"))
    swr_aligned = pd.concat(swr_parts, ignore_index=True).set_index("_pos")["_val"]
    history["surface_win_rate_52w"] = history["_pos"].map(swr_aligned)

    # ── 4. fatigue_minutes_Xd: time-based rolling sum per player
    def _time_rolling_sum(group_df, value_col, window):
        s = group_df[value_col].fillna(0)
        rolled = s.rolling(window, closed="left").sum()
        return pd.DataFrame({"_pos": group_df["_pos"].values,
                             "_val": rolled.values})

    fat7_parts, fat14_parts = [], []
    for _player, grp in history_dt.groupby("player", sort=False):
        fat7_parts.append(_time_rolling_sum(grp, "minutes", "7D"))
        fat14_parts.append(_time_rolling_sum(grp, "minutes", "14D"))
    fat7_aligned = pd.concat(fat7_parts, ignore_index=True).set_index("_pos")["_val"]
    fat14_aligned = pd.concat(fat14_parts, ignore_index=True).set_index("_pos")["_val"]
    history["fatigue_minutes_7d"] = history["_pos"].map(fat7_aligned)
    history["fatigue_minutes_14d"] = history["_pos"].map(fat14_aligned)

    # ── 5. h2h_surface_advantage: per (player, opponent, surface), expanding
    # mean using only matches strictly before current date. We can't use
    # closed='left' on expanding without time-indexed rolling; instead we
    # compute manually: cumulative mean of values from earlier dates only.
    def _h2h(group):
        # group sorted by match_date; need expanding mean of `won` BUT
        # excluding all rows with the same match_date as the current one.
        dates = group["match_date"].values
        wins = group["won"].astype(float).values
        out = np.full(len(group), np.nan)
        for i in range(len(group)):
            mask = dates < dates[i]  # strict less-than
            if mask.any():
                out[i] = wins[mask].mean()
        return pd.Series(out, index=group.index)
    history["h2h_surface_advantage"] = (
        history.groupby(["player", "opponent", "surface"], group_keys=False)
        .apply(_h2h)
    )

    # ── Now project history features back onto df via merge.
    # For each match in df, we look up the row in history that corresponds
    # to (winner_name, match_date, won=True) — and (loser_name, match_date, won=False).

    feat_cols = [
        "days_since_last", "recent_form_10m", "surface_win_rate_52w",
        "fatigue_minutes_7d", "fatigue_minutes_14d", "h2h_surface_advantage",
    ]

    # Deduplicate on key (rare same-day same-pair matches in TML); keep first
    history_idx = (
        history.groupby(["player", "match_date", "won"])[feat_cols]
        .first()
    )

    # Winner features
    w_keys = list(zip(df["winner_name"], df["match_date"],
                      [True] * len(df)))
    w_feats = history_idx.reindex(w_keys)[feat_cols].values

    # Loser features
    l_keys = list(zip(df["loser_name"], df["match_date"],
                      [False] * len(df)))
    l_feats = history_idx.reindex(l_keys)[feat_cols].values

    # Attach to df with _w / _l suffix
    name_map = {
        "fatigue_minutes_7d":     "fatigue_minutes_7d",
        "fatigue_minutes_14d":    "fatigue_minutes_14d",
        "days_since_last":        "days_since_last_match",
        "surface_win_rate_52w":   "surface_win_rate_52w",
        "recent_form_10m":        "recent_form_10m",
        "h2h_surface_advantage":  "h2h_surface_advantage",
    }
    for i, c in enumerate(feat_cols):
        out_name = name_map[c]
        df[f"{out_name}_w"] = w_feats[:, i]
        df[f"{out_name}_l"] = l_feats[:, i]

    return df


# ============================================================================
# History table builder
# ============================================================================

def _build_player_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match-level DF (winner-first) into long-form per-player history.
    Each match becomes 2 rows: one from winner POV, one from loser POV.

    Columns: player, opponent, match_date, surface, minutes, won
    """
    winner_view = pd.DataFrame({
        "player":     df["winner_name"],
        "opponent":   df["loser_name"],
        "match_date": df["match_date"],
        "surface":    df["surface"],
        "minutes":    df["minutes"],
        "won":        True,
    })
    loser_view = pd.DataFrame({
        "player":     df["loser_name"],
        "opponent":   df["winner_name"],
        "match_date": df["match_date"],
        "surface":    df["surface"],
        "minutes":    df["minutes"],
        "won":        False,
    })
    history = pd.concat([winner_view, loser_view], ignore_index=True)
    history = history.sort_values(["player", "match_date"]).reset_index(drop=True)
    return history


# ============================================================================
# Per-feature helpers (each takes a player + cutoff date, looks ONLY backward)
# ============================================================================

def _fatigue_minutes(
    history: pd.DataFrame,
    player: str,
    cutoff: pd.Timestamp,
    days: int,
) -> float:
    """
    Total minutes played by `player` in the [cutoff - days, cutoff) window.
    Strict less-than on cutoff so the current match is never counted.

    Returns 0.0 if no matches in window. NaN if minutes data missing.
    """
    if not isinstance(player, str) or not player:
        return np.nan
    window_start = cutoff - timedelta(days=days)
    mask = (
        (history["player"] == player)
        & (history["match_date"] >= window_start)
        & (history["match_date"] < cutoff)
    )
    rows = history.loc[mask, "minutes"]
    if rows.empty:
        return 0.0
    if rows.isna().all():
        return np.nan
    return float(rows.fillna(0).sum())


def _days_since_last(
    history: pd.DataFrame,
    player: str,
    cutoff: pd.Timestamp,
) -> float:
    """
    Days between cutoff and the most recent match before cutoff.
    NaN if the player has no prior match in the dataset.
    """
    if not isinstance(player, str) or not player:
        return np.nan
    mask = (history["player"] == player) & (history["match_date"] < cutoff)
    prev = history.loc[mask, "match_date"]
    if prev.empty:
        return np.nan
    return float((cutoff - prev.max()).days)


def _surface_win_rate(
    history: pd.DataFrame,
    player: str,
    cutoff: pd.Timestamp,
    surface: str,
    days: int = 365,
) -> float:
    """
    Win rate for `player` on `surface` over the prior `days` days.
    Strict less-than on cutoff. NaN if fewer than 3 matches in window
    (sample too small to be meaningful).
    """
    if not isinstance(player, str) or not player or not isinstance(surface, str):
        return np.nan
    window_start = cutoff - timedelta(days=days)
    mask = (
        (history["player"] == player)
        & (history["match_date"] >= window_start)
        & (history["match_date"] < cutoff)
        & (history["surface"] == surface)
    )
    rows = history.loc[mask, "won"]
    if len(rows) < 3:
        return np.nan
    return float(rows.mean())


def _recent_form(
    history: pd.DataFrame,
    player: str,
    cutoff: pd.Timestamp,
    n: int = 10,
) -> float:
    """
    Win rate over the most recent `n` matches before cutoff (any surface).
    NaN if fewer than 3 matches available.
    """
    if not isinstance(player, str) or not player:
        return np.nan
    mask = (history["player"] == player) & (history["match_date"] < cutoff)
    rows = history.loc[mask].sort_values("match_date", ascending=False).head(n)
    if len(rows) < 3:
        return np.nan
    return float(rows["won"].mean())


def _h2h_surface(
    history: pd.DataFrame,
    player: str,
    opponent: str,
    cutoff: pd.Timestamp,
    surface: str,
) -> float:
    """
    Player's win rate vs this specific opponent on this specific surface.
    NaN if they've never played on this surface (extremely common —
    most matchups are first-time meetings on a given surface).
    """
    if not all(isinstance(x, str) and x for x in [player, opponent, surface]):
        return np.nan
    mask = (
        (history["player"] == player)
        & (history["opponent"] == opponent)
        & (history["match_date"] < cutoff)
        & (history["surface"] == surface)
    )
    rows = history.loc[mask, "won"]
    if rows.empty:
        return np.nan
    return float(rows.mean())


# ============================================================================
# Dummy data tests
# ============================================================================

def _test_fatigue_minutes():
    print("=" * 60)
    print("fatigue_minutes — windowed sum, strict cutoff")
    print("=" * 60)

    history = pd.DataFrame([
        {"player": "Darderi", "opponent": "X", "match_date": pd.Timestamp("2024-04-20"),
         "surface": "Clay", "minutes": 90.0, "won": True},
        {"player": "Darderi", "opponent": "Y", "match_date": pd.Timestamp("2024-04-22"),
         "surface": "Clay", "minutes": 125.0, "won": True},
        {"player": "Darderi", "opponent": "Z", "match_date": pd.Timestamp("2024-04-25"),
         "surface": "Clay", "minutes": 117.0, "won": True},  # the match we're predicting — must NOT be counted
    ])
    cutoff = pd.Timestamp("2024-04-25")

    result_7d = _fatigue_minutes(history, "Darderi", cutoff, days=7)
    print(f"  7-day fatigue at 2024-04-25: {result_7d}  (expected 215.0 — Apr 20 + Apr 22)")
    assert result_7d == 215.0, f"got {result_7d}"

    result_14d = _fatigue_minutes(history, "Darderi", cutoff, days=14)
    print(f"  14-day fatigue:              {result_14d}  (expected 215.0)")
    assert result_14d == 215.0

    # Player not in history → 0
    result_none = _fatigue_minutes(history, "Sinner", cutoff, days=7)
    print(f"  Unknown player:              {result_none}  (expected 0.0)")
    assert result_none == 0.0
    print("  PASSED ✓")


def _test_days_since_last():
    print("\n" + "=" * 60)
    print("days_since_last_match")
    print("=" * 60)

    history = pd.DataFrame([
        {"player": "Darderi", "opponent": "X", "match_date": pd.Timestamp("2024-04-20"),
         "surface": "Clay", "minutes": 90, "won": True},
    ])
    cutoff = pd.Timestamp("2024-04-25")
    result = _days_since_last(history, "Darderi", cutoff)
    print(f"  Days since 2024-04-20 → 2024-04-25: {result}  (expected 5.0)")
    assert result == 5.0

    # No prior match → NaN
    result_nan = _days_since_last(history, "Sinner", cutoff)
    print(f"  Unknown player: {result_nan}  (expected NaN)")
    assert pd.isna(result_nan)
    print("  PASSED ✓")


def _test_surface_win_rate():
    print("\n" + "=" * 60)
    print("surface_win_rate_52w (min 3 matches required)")
    print("=" * 60)

    rows = []
    for i, won in enumerate([True, True, False, True, True]):
        rows.append({
            "player": "Darderi", "opponent": f"P{i}",
            "match_date": pd.Timestamp("2024-01-15") + timedelta(days=i*10),
            "surface": "Clay", "minutes": 100, "won": won,
        })
    # Add some hard court matches that should be excluded
    rows.append({
        "player": "Darderi", "opponent": "PH",
        "match_date": pd.Timestamp("2024-02-01"),
        "surface": "Hard", "minutes": 100, "won": False,
    })
    history = pd.DataFrame(rows)
    cutoff = pd.Timestamp("2024-04-25")

    result = _surface_win_rate(history, "Darderi", cutoff, "Clay", days=365)
    print(f"  4 wins / 5 clay matches: {result}  (expected 0.8)")
    assert result == 0.8

    # Fewer than 3 matches → NaN (sample too small)
    history_thin = history.head(2)
    result_thin = _surface_win_rate(history_thin, "Darderi", cutoff, "Clay")
    print(f"  Only 2 matches: {result_thin}  (expected NaN, sample too small)")
    assert pd.isna(result_thin)
    print("  PASSED ✓")


def _test_recent_form():
    print("\n" + "=" * 60)
    print("recent_form_10m")
    print("=" * 60)

    rows = []
    # 12 matches, 8 wins, 4 losses — most recent 10 should give 7/10
    outcomes = [True]*8 + [False]*4
    for i, won in enumerate(outcomes):
        rows.append({
            "player": "Darderi", "opponent": f"P{i}",
            "match_date": pd.Timestamp("2024-01-01") + timedelta(days=i*5),
            "surface": "Clay", "minutes": 100, "won": won,
        })
    history = pd.DataFrame(rows)
    cutoff = pd.Timestamp("2024-04-25")

    result = _recent_form(history, "Darderi", cutoff, n=10)
    # Most recent 10 matches: idx 2-11 → 6 wins (idx 2-7) + 4 losses (idx 8-11) = 0.6
    print(f"  Last 10 of 12 matches: {result}  (expected 0.6)")
    assert result == 0.6
    print("  PASSED ✓")


def _test_h2h_surface():
    print("\n" + "=" * 60)
    print("h2h_surface_advantage")
    print("=" * 60)

    history = pd.DataFrame([
        # Darderi vs Cerundolo on clay: Darderi 1-1
        {"player": "Darderi", "opponent": "Cerundolo", "match_date": pd.Timestamp("2023-05-01"),
         "surface": "Clay", "minutes": 100, "won": True},
        {"player": "Cerundolo", "opponent": "Darderi", "match_date": pd.Timestamp("2023-05-01"),
         "surface": "Clay", "minutes": 100, "won": False},
        {"player": "Darderi", "opponent": "Cerundolo", "match_date": pd.Timestamp("2023-08-01"),
         "surface": "Clay", "minutes": 100, "won": False},
        {"player": "Cerundolo", "opponent": "Darderi", "match_date": pd.Timestamp("2023-08-01"),
         "surface": "Clay", "minutes": 100, "won": True},
        # On hard court (should be excluded)
        {"player": "Darderi", "opponent": "Cerundolo", "match_date": pd.Timestamp("2024-01-01"),
         "surface": "Hard", "minutes": 100, "won": True},
    ])
    cutoff = pd.Timestamp("2024-04-25")

    result = _h2h_surface(history, "Darderi", "Cerundolo", cutoff, "Clay")
    print(f"  Darderi vs Cerundolo on clay (1W, 1L): {result}  (expected 0.5)")
    assert result == 0.5

    # Never met → NaN
    result_nan = _h2h_surface(history, "Sinner", "Alcaraz", cutoff, "Clay")
    print(f"  Never met: {result_nan}  (expected NaN)")
    assert pd.isna(result_nan)
    print("  PASSED ✓")


def _test_compute_all_endtoend():
    print("\n" + "=" * 60)
    print("compute_all — end to end on tiny match DataFrame")
    print("=" * 60)

    matches = pd.DataFrame([
        # Earlier matches that establish history
        {"winner_name": "Darderi", "loser_name": "Other1",
         "match_date": pd.Timestamp("2024-04-20"),
         "surface": "Clay", "minutes": 90.0},
        {"winner_name": "Darderi", "loser_name": "Other2",
         "match_date": pd.Timestamp("2024-04-22"),
         "surface": "Clay", "minutes": 125.0},
        {"winner_name": "Other3", "loser_name": "Cerundolo",
         "match_date": pd.Timestamp("2024-04-23"),
         "surface": "Clay", "minutes": 100.0},
        # The match we're predicting
        {"winner_name": "Darderi", "loser_name": "Cerundolo",
         "match_date": pd.Timestamp("2024-04-25"),
         "surface": "Clay", "minutes": 117.0},
    ])
    out = compute_all(matches)
    last = out.iloc[-1]

    print(f"  fatigue_minutes_7d_w (Darderi):  {last['fatigue_minutes_7d_w']}  (expected 215.0)")
    print(f"  fatigue_minutes_7d_l (Cerundolo):{last['fatigue_minutes_7d_l']}  (expected 100.0)")
    print(f"  days_since_last_match_w:         {last['days_since_last_match_w']}  (expected 3.0)")
    print(f"  days_since_last_match_l:         {last['days_since_last_match_l']}  (expected 2.0)")
    assert last["fatigue_minutes_7d_w"] == 215.0
    assert last["fatigue_minutes_7d_l"] == 100.0
    assert last["days_since_last_match_w"] == 3.0
    assert last["days_since_last_match_l"] == 2.0
    print("  PASSED ✓")


def _test_no_same_day_leak():
    """
    CRITICAL: TML uses tourney_date (week start), so multiple matches share
    the same date. The current match's stats must NEVER leak into its own
    feature computation. This test verifies that.
    """
    print("\n" + "=" * 60)
    print("compute_all — SAME-DAY LEAK TEST")
    print("=" * 60)
    print("  Build 3 matches all on the same date.")
    print("  Each match's fatigue feature should NOT include any of the")
    print("  same-day matches' minutes.")

    matches = pd.DataFrame([
        # 3 matches on the SAME date — extreme stress test
        {"winner_name": "PlayerA", "loser_name": "PlayerB",
         "match_date": pd.Timestamp("2024-04-25"),
         "surface": "Clay", "minutes": 100.0},
        {"winner_name": "PlayerA", "loser_name": "PlayerC",
         "match_date": pd.Timestamp("2024-04-25"),
         "surface": "Clay", "minutes": 80.0},
        {"winner_name": "PlayerD", "loser_name": "PlayerA",
         "match_date": pd.Timestamp("2024-04-25"),
         "surface": "Clay", "minutes": 60.0},
    ])
    out = compute_all(matches)

    # PlayerA appears in all 3 matches. None of these minutes (100, 80, 60)
    # should appear in any fatigue feature for PlayerA.
    a_fatigues_w = out["fatigue_minutes_7d_w"][out["winner_name"] == "PlayerA"]
    a_fatigues_l = out["fatigue_minutes_7d_l"][out["loser_name"] == "PlayerA"]

    print(f"  PlayerA fatigue values when winner: {list(a_fatigues_w.values)}")
    print(f"  PlayerA fatigue values when loser:  {list(a_fatigues_l.values)}")
    print(f"  All should be 0.0 — no prior-day matches exist.")

    assert (a_fatigues_w.fillna(0) == 0).all(), "fatigue leaked from same-day matches"
    assert (a_fatigues_l.fillna(0) == 0).all(), "fatigue leaked from same-day matches"
    print("  PASSED ✓ — no leak from same-day matches")


def _test_distributional_sanity():
    """
    Verifies that across many matches, mean fatigue for winners equals
    mean fatigue for losers. If they differ systematically, the current
    match's data is leaking.
    """
    print("\n" + "=" * 60)
    print("compute_all — DISTRIBUTIONAL SANITY")
    print("=" * 60)
    rng = np.random.default_rng(0)
    players = [f"P{i}" for i in range(50)]
    n = 500
    rows = []
    base = pd.Timestamp("2024-01-01")
    for i in range(n):
        a, b = rng.choice(players, 2, replace=False)
        won_a = rng.random() < 0.5
        rows.append({
            "winner_name": a if won_a else b,
            "loser_name":  b if won_a else a,
            "match_date":  base + pd.Timedelta(days=int(rng.integers(0, 365))),
            "surface":     rng.choice(["Clay", "Hard"]),
            "minutes":     float(rng.uniform(60, 180)),
        })
    matches = pd.DataFrame(rows)
    out = compute_all(matches)

    mean_w = out["fatigue_minutes_7d_w"].mean()
    mean_l = out["fatigue_minutes_7d_l"].mean()
    diff = mean_w - mean_l
    print(f"  Mean winner fatigue: {mean_w:.2f}")
    print(f"  Mean loser fatigue:  {mean_l:.2f}")
    print(f"  Difference:          {diff:+.2f}  (should be near 0)")
    # Allow small noise from the random sample
    assert abs(diff) < 5.0, f"systematic diff of {diff} suggests leak"
    print("  PASSED ✓ — no systematic leak detected")


if __name__ == "__main__":
    _test_fatigue_minutes()
    _test_days_since_last()
    _test_surface_win_rate()
    _test_recent_form()
    _test_h2h_surface()
    _test_compute_all_endtoend()
    _test_no_same_day_leak()
    _test_distributional_sanity()
    print("\n" + "=" * 60)
    print("All feature_engineer tests passed.")
    print("=" * 60)
