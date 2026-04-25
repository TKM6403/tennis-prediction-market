"""
TML (TennisMyLife) data loader.

Pulls ATP tour and challenger CSVs from stats.tennismylife.org,
caches them locally in data/raw/tml/, and returns clean DataFrames.

Key design decisions:
- Cache raw CSVs to disk so we don't re-download on every run
- Force cutoff_date on all queries so features can never leak future data
- Challenger data included — different competitive context but same schema
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Where to cache raw CSVs
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "tml"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://stats.tennismylife.org/data"

# Tourney level codes in TML data
LEVEL_MAP = {
    "G": "grand_slam",
    "M": "masters",
    "500": "atp500",
    "250": "atp250",
    "A": "other_atp",
    "D": "davis_cup",
    "F": "atp_finals",
}

SURFACES = {"Hard", "Clay", "Grass", "Carpet"}


def _cache_path(year: int, challenger: bool = False) -> Path:
    suffix = "_challenger" if challenger else ""
    return RAW_DIR / f"{year}{suffix}.csv"


def _download(year: int, challenger: bool = False) -> Path:
    """Download a single year CSV and cache it. Re-downloads current year always."""
    suffix = "_challenger" if challenger else ""
    url = f"{BASE_URL}/{year}{suffix}.csv"
    path = _cache_path(year, challenger)

    current_year = datetime.now().year
    # Always re-download current year (data updates daily)
    # For past years, use cache if it exists
    if path.exists() and year < current_year:
        logger.info(f"Cache hit: {path.name}")
        return path

    logger.info(f"Downloading {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def _load_year(year: int, challenger: bool = False) -> pd.DataFrame:
    path = _download(year, challenger)
    df = pd.read_csv(path, low_memory=False)

    # Normalize tourney_date to proper datetime
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")

    # Tag challenger vs tour
    df["is_challenger"] = challenger

    # Numeric coercions for key fields
    for col in ["winner_rank", "loser_rank", "winner_age", "loser_age",
                "winner_ht", "loser_ht", "minutes",
                "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
                "w_SvGms", "w_bpSaved", "w_bpFaced",
                "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
                "l_SvGms", "l_bpSaved", "l_bpFaced"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_matches(
    start_year: int = 2018,
    end_year: int = None,
    include_challenger: bool = False,
    surfaces: list = None,
    levels: list = None,
    cutoff_date: str = None,
) -> pd.DataFrame:
    """
    Load TML match data across a year range.

    Args:
        start_year:          First year to load (default 2018)
        end_year:            Last year to load (default: current year)
        include_challenger:  Include ATP Challenger Tour matches
        surfaces:            Filter to surfaces e.g. ['Grass', 'Hard']
        levels:              Filter to tourney levels e.g. ['G', 'M']
        cutoff_date:         CRITICAL — if set, drops all matches on or after
                             this date. Use this when building features for a
                             specific match to prevent lookahead bias.
                             Format: 'YYYY-MM-DD'

    Returns:
        pd.DataFrame with one row per match
    """
    if end_year is None:
        end_year = datetime.now().year

    frames = []
    for year in range(start_year, end_year + 1):
        try:
            frames.append(_load_year(year, challenger=False))
            if include_challenger:
                frames.append(_load_year(year, challenger=True))
        except requests.HTTPError as e:
            logger.warning(f"Could not load year {year}: {e}")

    if not frames:
        raise ValueError("No data loaded. Check year range.")

    df = pd.concat(frames, ignore_index=True)

    # Apply cutoff — this is the lookahead bias guard
    if cutoff_date is not None:
        cutoff = pd.Timestamp(cutoff_date)
        before = len(df)
        df = df[df["tourney_date"] < cutoff]
        logger.info(f"Cutoff {cutoff_date}: dropped {before - len(df)} rows after cutoff")

    # Surface filter
    if surfaces:
        df = df[df["surface"].isin(surfaces)]

    # Level filter
    if levels:
        df = df[df["tourney_level"].isin(levels)]

    df = df.sort_values("tourney_date").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} matches ({start_year}–{end_year})")
    return df


def get_player_history(
    player_name: str,
    df: pd.DataFrame,
    cutoff_date: str,
    surface: str = None,
    last_n_days: int = 365,
) -> pd.DataFrame:
    """
    Return all matches for a player strictly before cutoff_date.

    This is the core building block for feature engineering.
    Every feature we compute calls this first.

    Args:
        player_name:   Exact name as it appears in TML data
        df:            Full match DataFrame (already loaded)
        cutoff_date:   Only return matches strictly before this date
        surface:       Optional surface filter
        last_n_days:   Rolling window in days (default 365 = last 52 weeks)

    Returns:
        DataFrame of that player's matches as winner or loser, with a
        'won' column (1 if they won, 0 if they lost) and 'player_rank',
        'player_ht', serve stats normalized to player perspective.
    """
    cutoff = pd.Timestamp(cutoff_date)
    since = cutoff - timedelta(days=last_n_days)

    # Matches where player won
    won = df[
        (df["winner_name"] == player_name) &
        (df["tourney_date"] < cutoff) &
        (df["tourney_date"] >= since)
    ].copy()
    won["won"] = 1
    won["player_rank"] = won["winner_rank"]
    won["opp_rank"] = won["loser_rank"]
    won["player_age"] = won["winner_age"]
    won["svpt"] = won["w_svpt"]
    won["first_in"] = won["w_1stIn"]
    won["first_won"] = won["w_1stWon"]
    won["second_won"] = won["w_2ndWon"]
    won["aces"] = won["w_ace"]
    won["dfs"] = won["w_df"]
    won["bp_saved"] = won["w_bpSaved"]
    won["bp_faced"] = won["w_bpFaced"]

    # Matches where player lost
    lost = df[
        (df["loser_name"] == player_name) &
        (df["tourney_date"] < cutoff) &
        (df["tourney_date"] >= since)
    ].copy()
    lost["won"] = 0
    lost["player_rank"] = lost["loser_rank"]
    lost["opp_rank"] = lost["winner_rank"]
    lost["player_age"] = lost["loser_age"]
    lost["svpt"] = lost["l_svpt"]
    lost["first_in"] = lost["l_1stIn"]
    lost["first_won"] = lost["l_1stWon"]
    lost["second_won"] = lost["l_2ndWon"]
    lost["aces"] = lost["l_ace"]
    lost["dfs"] = lost["l_df"]
    lost["bp_saved"] = lost["l_bpSaved"]
    lost["bp_faced"] = lost["l_bpFaced"]

    history = pd.concat([won, lost], ignore_index=True)
    history = history.sort_values("tourney_date")

    if surface:
        history = history[history["surface"] == surface]

    return history


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Smoke test
    df = load_matches(start_year=2023, end_year=2025, levels=["G", "M"])
    print(f"Loaded {len(df)} slam + masters matches")
    print(df[["tourney_date", "tourney_name", "surface", "winner_name",
              "loser_name", "winner_rank", "loser_rank"]].head(10).to_string())

    print("\nPlayer history test — Sinner on clay before Wimbledon 2024:")
    sinner = get_player_history(
        "Jannik Sinner", df,
        cutoff_date="2024-07-01",
        surface="Clay",
        last_n_days=365
    )
    print(f"Matches found: {len(sinner)}")
    print(f"Win rate: {sinner['won'].mean():.2%}")
