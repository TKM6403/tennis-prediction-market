"""
tml_match_loader.py

Match-level loader that combines:
    1. Sackmann TML (full match stats, winner/loser perspective)
    2. tennis-data.co.uk (per-match calendar dates)

Why:
    TML is the gold standard for stats but only has tourney_date — the week
    the tournament started, not the actual match date. tennis-data.co.uk has
    per-match dates but limited stats.

    For our prediction market workflow we need the EXACT match date so we
    can join to Kalshi / Polymarket markets without a fuzzy date window
    that could span multiple matches by the same player.

Resolution strategy (per TML row):
    1. EXACT: look up (last_name_winner, last_name_loser, year, tournament)
       in tennis-data → get the actual match date.
    2. EMPIRICAL: if exact lookup misses but we have OTHER tennis-data rows
       for the same (level, round_), use the median day-offset from
       tourney_date.
    3. HEURISTIC: hard-coded round → day-offset table for tournaments we
       have no tennis-data for at all. Coarse but bounded error.

The result: every TML match gets a `match_date` and a `date_confidence`
column ("exact" / "empirical" / "heuristic"). Downstream (the
MarketMatchJoiner) can prefer high-confidence rows when joining.

----------------------------------------------------------------------------
LOOKAHEAD GUARD
----------------------------------------------------------------------------

Per CLAUDE.md, load() accepts cutoff_date and drops rows on or after that
date. The cutoff is applied to MATCH_DATE (the resolved per-match date),
not tourney_date — otherwise an early-round Madrid match on day 1 might
leak into a feature window for a later-round match in the same tournament.
"""

from __future__ import annotations

from pathlib import Path
from datetime import timedelta
import logging
import re
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

try:
    from src.loaders.tml_loader import load_matches as _load_tml_matches
except ImportError:
    # Fallback for direct script execution
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.loaders.tml_loader import load_matches as _load_tml_matches

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# tennis-data.co.uk URL slugs for tournaments we expect to encounter.
# Maps a normalized TML tourney_name -> tennis-data slug.
# The keys are stripped/lowercased; the values are exact path components in
# http://www.tennis-data.co.uk/{year}/{slug}.csv
#
# Coverage focus: all Grand Slams + all Masters 1000 + the ATP 500s most
# likely to show up on Kalshi / Polymarket (high-volume tournaments).
# Smaller 250s and challengers will fall through to empirical/heuristic
# resolution. That's fine — the tournaments with the most market volume
# are the ones we have exact dates for.
TENNIS_DATA_SLUGS: Dict[str, str] = {
    # Grand Slams
    "australian open":         "ausopen",
    "us open":                 "usopen",
    "wimbledon":               "wimbledon",
    "roland garros":           "frenchopen",
    "french open":             "frenchopen",
    # Masters 1000
    "indian wells masters":    "indianwells",
    "indian wells":            "indianwells",
    "miami masters":           "miami",
    "miami open":              "miami",
    "miami":                   "miami",
    "monte carlo masters":     "montecarlo",
    "monte carlo":             "montecarlo",
    "madrid masters":          "madrid",
    "madrid":                  "madrid",
    "mutua madrid open":       "madrid",
    "rome masters":            "rome",
    "rome":                    "rome",
    "italian open":            "rome",
    "canada masters":          "montreal",  # alternates Toronto/Montreal
    "toronto":                 "montreal",
    "montreal":                "montreal",
    "cincinnati masters":      "cincinnati",
    "cincinnati":              "cincinnati",
    "shanghai masters":        "shanghai",
    "shanghai":                "shanghai",
    "paris masters":           "paris",
    "paris":                   "paris",
    # Selected 500s
    "rotterdam":               "rotterdam",
    "dubai":                   "dubai",
    "barcelona":               "barcelona",
    "hamburg":                 "hamburg",
    "queens club":             "queensclub",
    "halle":                   "halle",
    "washington":              "washington",
    "beijing":                 "beijing",
    "tokyo":                   "tokyo",
    "vienna":                  "vienna",
    "basel":                   "basel",
    # Selected 250s
    "doha":                    "doha",
    "adelaide":                "adelaide1",
    "auckland":                "auckland",
    "brisbane":                "brisbane",
    # Additional 250s confirmed via URL probe
    "montpellier":             "montpellier",
    "buenos aires":            "buenosaires",
    "delray beach":            "delraybeach",
    "marseille":               "marseille",
    "kitzbuhel":               "kitzbuhel",
    "antwerp":                 "antwerp",
    "sofia":                   "sofia",
    "estoril":                 "estoril",
    "munich":                  "munich",
    "lyon":                    "lyon",
    "geneva":                  "geneva",
    "stuttgart":               "stuttgart",
    "eastbourne":              "eastbourne",
    "bastad":                  "bastad",
    "umag":                    "umag",
    "newport":                 "newport",
    "atlanta":                 "atlanta",
    "gstaad":                  "gstaad",
    "metz":                    "metz",
    "stockholm":               "stockholm",
    "pune":                    "pune",
    # Second tier — confirmed via URL probe
    "cordoba":                 "cordoba",
    "houston":                 "houston",
    "marrakech":               "marrakech",
    "'s-hertogenbosch":        "shertogenbosch",
    "s-hertogenbosch":         "shertogenbosch",
    "hertogenbosch":           "shertogenbosch",
    "winston-salem":           "winstonsalem",
    "winston salem":           "winstonsalem",
    "santiago":                "santiago",
    "chengdu":                 "chengdu",
    "los cabos":               "loscabos",
    "mallorca":                "mallorca",
    "dallas":                  "dallas",
    # Not on tennis-data.co.uk — Sydney, Antalya, St. Petersburg, Moscow,
    # Belgrade will remain on heuristic fallback permanently.
}

# Hard-coded round-to-day-offset fallback when we have NO tennis-data for
# a tournament. Conservative midpoints; expected error ±1-2 days.
#
# Keys: (tourney_level, round_). tourney_level uses TML conventions:
#   "G"   = Grand Slam (14 day event, 128 draw)
#   "M"   = Masters 1000 (10-12 days, 96 draw)
#   "500" = ATP 500 (~1 week, 32-48 draw)
#   "250" = ATP 250 (~1 week, 28-32 draw)
HEURISTIC_ROUND_OFFSETS: Dict[Tuple[str, str], int] = {
    # Grand Slam
    ("G", "R128"):  1,
    ("G", "R64"):   3,
    ("G", "R32"):   5,
    ("G", "R16"):   7,
    ("G", "QF"):    9,
    ("G", "SF"):   11,
    ("G", "F"):    13,
    # Masters 1000
    ("M", "R128"): 1,
    ("M", "R64"):  2,
    ("M", "R32"):  4,
    ("M", "R16"):  6,
    ("M", "QF"):   7,
    ("M", "SF"):   9,
    ("M", "F"):   10,
    # 500 (1 week, 32 draw typical)
    ("500", "R32"): 1,
    ("500", "R16"): 3,
    ("500", "QF"):  4,
    ("500", "SF"):  5,
    ("500", "F"):   6,
    # 250 (1 week, 28-32 draw typical)
    ("250", "R32"): 1,
    ("250", "R16"): 3,
    ("250", "QF"):  4,
    ("250", "SF"):  5,
    ("250", "F"):   6,
    # ATP-level "A" (mixed, treated as 250 by default)
    ("A", "R32"): 1,
    ("A", "R16"): 3,
    ("A", "QF"):  4,
    ("A", "SF"):  5,
    ("A", "F"):   6,
}

# Map tennis-data.co.uk Round strings to TML round_ codes
_TD_ROUND_MAP = {
    "1st round":      "R64",   # ambiguous — depends on draw size, refined below
    "2nd round":      "R32",
    "3rd round":      "R16",
    "4th round":      "R16",   # only Slams have 4th round, gets remapped
    "round of 32":    "R32",
    "round of 16":    "R16",
    "quarterfinals":  "QF",
    "semifinals":     "SF",
    "the final":      "F",
}

# Map TML tourney_level to "G/M/500/250/A" buckets used in heuristic table.
# TML levels we see: 'G' (slam), 'M' (Masters 1000), 'A' (ATP), '250', '500',
# 'F' (tour finals), 'D' (Davis Cup), 'C' (Challenger).
def _level_bucket(tml_level: str) -> str:
    if tml_level in ("G", "M", "500", "250", "A"):
        return tml_level
    return "A"  # safe default


# Tournaments with non-standard scheduling where round-offset heuristics are
# meaningless and match dates cannot be reliably inferred.
#
# BETTING SAFETY: these are also excluded from inference — attempting to run
# prediction on a match from one of these events will raise IrregularFormatError.
# This prevents accidental bets on markets where our model has no calibrated
# prior.
#
# Davis Cup (D):    Team event, captain selections, surface chosen per tie,
#                   scheduling across multiple days with no fixed round pattern.
# Tour Finals (F):  Round-robin then knockout. Round labels (RR, SF, F) don't
#                   map to fixed offsets from tourney_date.
# Olympics (O):     4-year irregular cycle, compressed schedule with different
#                   seeding rules and no ranking implications.
IRREGULAR_FORMAT_LEVELS = frozenset({"D", "F", "O"})

IRREGULAR_FORMAT_TOURNAMENTS = frozenset({
    "atp cup",
    "united cup",
    "laver cup",
    "hopman cup",
    "world team cup",
})


class IrregularFormatError(ValueError):
    """
    Raised when inference is attempted on a tournament type that has been
    explicitly excluded from the model due to non-standard scheduling,
    team-based format, or other structural incompatibilities.

    This is a hard safety guard — do not catch and suppress this error.
    If you are trying to place a bet and see this, stop.

    Tournament types that trigger this:
        Davis Cup, Tour Finals, Olympics, ATP Cup, United Cup, Laver Cup.

    Why: our model is calibrated on standard ATP tour matches. These events
    have different formats, scheduling, surface selection, and player
    incentives. The model's probability estimates are not valid here.
    """
    pass


# ============================================================================
# Helpers
# ============================================================================

def _normalize_name(name) -> str:
    """
    Normalize a player name to a join key.

    Strategy: take last whitespace-delimited token, lowercase, strip
    punctuation. This handles:
      "Luciano Darderi"  -> "darderi"
      "Darderi L."       -> "darderi"      (tennis-data style)
      "De Minaur A."     -> "minaur"       (imperfect on multi-word names —
                                            documented limitation, last-name
                                            matching only as you requested)
    """
    if not isinstance(name, str) or not name.strip():
        return ""
    # Tennis-data format is "Lastname F." -> drop the initial-period token
    tokens = name.strip().split()
    # Drop anything that looks like an initial: single letter optionally with .
    tokens = [t for t in tokens if not re.fullmatch(r"[A-Za-z]\.?", t)]
    if not tokens:
        return ""
    last = tokens[-1].lower()
    last = re.sub(r"[^a-z\-]", "", last)
    return last


def _normalize_tournament(name) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def _parse_td_date(s) -> Optional[pd.Timestamp]:
    """
    Parse tennis-data.co.uk date. The site uses two formats inconsistently:
        'dd/mm/yyyy'  (4-digit year, used in 2019+ files)
        'dd/mm/yy'    (2-digit year, used in 2018 and earlier files)

    Returns NaT if neither format matches.
    """
    if not isinstance(s, str):
        return None
    # Try 4-digit year first (most common)
    parsed = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    if pd.notna(parsed):
        return parsed
    # Fall back to 2-digit year
    parsed = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    if pd.notna(parsed):
        return parsed
    return None


# ============================================================================
# TMLMatchLoader
# ============================================================================

class TMLMatchLoader:
    """
    Load TML match data and resolve per-match calendar dates by joining to
    tennis-data.co.uk where possible, falling back to round-offset
    heuristics elsewhere.

    Public API:
        load(start_year, end_year, cutoff_date) -> raw TML DataFrame
        normalize(raw, fetch_tennis_data=True) -> schema with match_date
        feature_engineer(df) -> stub for now

    All historical data filtered by cutoff_date applies the cutoff to the
    RESOLVED match_date, not the tourney_date.

    Dummy example (normalize):
        Raw TML row:
            tourney_name:    "Madrid"
            tourney_level:   "M"
            tourney_date:    Timestamp("2024-04-22")
            round:           "R64"
            winner_name:     "Luciano Darderi"
            loser_name:      "Juan Manuel Cerundolo"

        After normalize() with tennis-data fetched:
            tml_match_id:       "2024-339_M1"
            player_a:           "Luciano Darderi"
            player_b:           "Juan Manuel Cerundolo"
            tournament:         "Madrid"
            tourney_level:      "M"
            surface:            "Clay"
            indoor:             False
            round_:             "R64"
            tourney_date:       date(2024, 4, 22)
            match_date:         date(2024, 4, 25)        ← from tennis-data
            date_confidence:    "exact"
            player_a_won:       True
            minutes:            117.0
            ... [all serve/return stats] ...
    """

    TENNIS_DATA_BASE = "http://www.tennis-data.co.uk"

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            repo_root = Path(__file__).resolve().parents[2]
            self.cache_dir = repo_root / "data" / "raw" / "tennis_data"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        start_year: int = 2018,
        end_year: Optional[int] = None,
        include_challenger: bool = False,
        cutoff_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Pull raw TML match data, unfiltered by cutoff.

        cutoff_date is intentionally NOT applied here — the underlying TML
        data only knows tourney_date. Cutoff is applied to match_date AFTER
        normalize() resolves per-match dates.

        Args:
            start_year:          First year of TML data.
            end_year:            Last year (defaults to current year).
            include_challenger:  Include challenger tour matches.
            cutoff_date:         Stored on the loader; applied in normalize().

        Returns:
            Raw TML DataFrame (Sackmann column names).
        """
        self._cutoff_date = (
            pd.Timestamp(cutoff_date) if cutoff_date else None
        )
        df = _load_tml_matches(
            start_year=start_year,
            end_year=end_year,
            include_challenger=include_challenger,
        )
        return df

    def normalize(
        self,
        raw: pd.DataFrame,
        fetch_tennis_data: bool = True,
    ) -> pd.DataFrame:
        """
        Map TML columns to our schema and resolve match_date for every row.

        Args:
            raw:                  DataFrame from load().
            fetch_tennis_data:    If True, fetch tennis-data.co.uk CSVs to
                                  build the exact_lookup. If False, every
                                  row falls back to heuristic resolution.

        Returns:
            Normalized DataFrame with match_date + date_confidence columns.
        """
        if raw.empty:
            return pd.DataFrame()

        df = raw.copy()
        out = pd.DataFrame(index=df.index)

        # tml_match_id: tourney_id + match_num for traceability
        out["tml_match_id"] = (
            df["tourney_id"].astype(str) + "_M" + df["match_num"].astype(str)
        )

        # Player columns — keep TML "winner/loser" naming as a/b; we record
        # which one actually won so the joiner can reconcile against market
        # resolution.
        out["player_a"] = df["winner_name"]
        out["player_b"] = df["loser_name"]
        out["player_a_won"] = True  # winner_name always wins by definition

        # Tournament context
        out["tournament"] = df["tourney_name"]
        out["tourney_level"] = df["tourney_level"].astype(str)
        out["surface"] = df["surface"]
        out["indoor"] = df["indoor"].map({"I": True, "O": False})
        out["round_"] = df["round"]
        out["tourney_date"] = pd.to_datetime(df["tourney_date"]).dt.date

        # Resolve match_date
        if fetch_tennis_data:
            years = sorted(set(pd.to_datetime(df["tourney_date"]).dt.year))
            tennis_data_df = self._fetch_tennis_data(years)
            exact_lookup = self._build_exact_lookup(tennis_data_df)
            empirical_offsets = self._build_empirical_offsets(tennis_data_df)
        else:
            exact_lookup = {}
            empirical_offsets = {}

        resolutions = df.apply(
            lambda r: self._resolve_match_date(
                r, exact_lookup, empirical_offsets
            ),
            axis=1,
        )
        out["match_date"] = [r[0] for r in resolutions]
        out["date_confidence"] = [r[1] for r in resolutions]

        # Pass through all the stats we want available downstream
        passthrough = [
            "minutes",
            "winner_rank", "loser_rank",
            "winner_rank_points", "loser_rank_points",
            "winner_age", "loser_age",
            "winner_hand", "loser_hand",
            "winner_ht", "loser_ht",
            "winner_ioc", "loser_ioc",
            "score", "best_of",
            "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
            "w_SvGms", "w_bpSaved", "w_bpFaced",
            "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
            "l_SvGms", "l_bpSaved", "l_bpFaced",
        ]
        for c in passthrough:
            if c in df.columns:
                out[c] = df[c].values

        # Apply cutoff to match_date if set
        cutoff = getattr(self, "_cutoff_date", None)
        if cutoff is not None:
            mask = pd.to_datetime(out["match_date"]) < cutoff
            before = len(out)
            out = out[mask].reset_index(drop=True)
            logger.info(
                f"Cutoff {cutoff.date()}: dropped {before - len(out)} rows "
                f"(match_date >= cutoff)"
            )

        return out

    def feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features. Delegates to src/ml/features/feature_engineer.

        Returns df with new feature columns appended.
        """
        from src.ml.features.feature_engineer import compute_all
        return compute_all(df)

    @staticmethod
    def check_inference_safe(
        tournament: str,
        tourney_level: str,
        raise_on_unsafe: bool = True,
    ) -> bool:
        """
        Check whether it is safe to run model inference for a given match.

        Call this before computing features or generating a probability
        estimate for any live match. If the tournament is in the excluded
        set, inference is not valid and you should not place a bet.

        Args:
            tournament:       Tournament name (e.g. "Davis Cup", "ATP Cup").
            tourney_level:    TML-style level code (e.g. "D", "F", "O", "G").
            raise_on_unsafe:  If True (default), raise IrregularFormatError
                              when the match is unsafe. If False, return bool.

        Returns:
            True if safe to proceed. Never returns False — raises instead
            unless raise_on_unsafe=False.

        Raises:
            IrregularFormatError: if the tournament format makes model
                                  inference unreliable.

        Example:
            # Before placing a bet
            TMLMatchLoader.check_inference_safe(
                tournament="Davis Cup",
                tourney_level="D",
            )
            # → raises IrregularFormatError — do not proceed
        """
        tournament_key = _normalize_tournament(tournament)
        is_irregular = (
            tourney_level in IRREGULAR_FORMAT_LEVELS
            or tournament_key in IRREGULAR_FORMAT_TOURNAMENTS
        )
        if is_irregular:
            msg = (
                f"BETTING BLOCKED: '{tournament}' (level='{tourney_level}') is an "
                f"irregular-format tournament. The model is not calibrated for this "
                f"event type and probability estimates are not valid.\n"
                f"Blocked levels:      {sorted(IRREGULAR_FORMAT_LEVELS)}\n"
                f"Blocked tournaments: {sorted(IRREGULAR_FORMAT_TOURNAMENTS)}"
            )
            if raise_on_unsafe:
                raise IrregularFormatError(msg)
            return False
        return True

    # ------------------------------------------------------------------
    # Tennis-data.co.uk fetching
    # ------------------------------------------------------------------

    def _fetch_tennis_data(self, years) -> pd.DataFrame:
        """
        Pull all known tournament CSVs for the requested years.

        Caches each (year, slug) to disk. 404s are tolerated silently —
        not every slug exists every year.
        """
        import requests

        slugs = sorted(set(TENNIS_DATA_SLUGS.values()))
        all_rows = []

        for year in years:
            year = int(year)
            for slug in slugs:
                cache_path = self.cache_dir / f"{year}_{slug}.csv"
                if cache_path.exists():
                    try:
                        df = pd.read_csv(cache_path)
                        df["_td_year"] = year
                        df["_td_slug"] = slug
                        all_rows.append(df)
                        continue
                    except Exception:
                        pass

                url = f"{self.TENNIS_DATA_BASE}/{year}/{slug}.csv"
                try:
                    resp = requests.get(url, timeout=20)
                    if resp.status_code == 404:
                        continue
                    resp.raise_for_status()
                    cache_path.write_bytes(resp.content)
                    from io import BytesIO
                    df = pd.read_csv(BytesIO(resp.content))
                    df["_td_year"] = year
                    df["_td_slug"] = slug
                    all_rows.append(df)
                except Exception as e:
                    logger.warning(f"tennis-data fetch failed {year}/{slug}: {e}")

        if not all_rows:
            return pd.DataFrame()
        return pd.concat(all_rows, ignore_index=True)

    # ------------------------------------------------------------------
    # Lookup table builders
    # ------------------------------------------------------------------

    def _build_exact_lookup(
        self, td_df: pd.DataFrame
    ) -> Dict[Tuple[str, str, int, str], pd.Timestamp]:
        """
        Build a dict keyed on (winner_lastname, loser_lastname, year, slug)
        → exact match date.

        We key on slug rather than tournament name because tennis-data and
        TML use different names for the same event ("Mutua Madrid Open" vs
        "Madrid"). Slug is canonical.
        """
        if td_df.empty:
            return {}
        lookup = {}
        for _, row in td_df.iterrows():
            winner_key = _normalize_name(row.get("Winner", ""))
            loser_key = _normalize_name(row.get("Loser", ""))
            if not winner_key or not loser_key:
                continue
            date = _parse_td_date(row.get("Date"))
            if date is None or pd.isna(date):
                continue
            year = int(row["_td_year"])
            slug = row["_td_slug"]
            lookup[(winner_key, loser_key, year, slug)] = date
        return lookup

    def _build_empirical_offsets(
        self, td_df: pd.DataFrame
    ) -> Dict[Tuple[str, str], int]:
        """
        Build dict keyed on (level_bucket, round_code) → median day offset
        from each tournament's start date.

        Useful when we have tennis-data for some matches in a tournament
        but not the specific match we're resolving.
        """
        if td_df.empty:
            return {}

        # Need tourney start = min(Date) per (year, slug)
        td = td_df.copy()
        td["_date"] = td["Date"].apply(_parse_td_date)
        td = td.dropna(subset=["_date"])
        if td.empty:
            return {}

        starts = td.groupby(["_td_year", "_td_slug"])["_date"].min().reset_index()
        starts = starts.rename(columns={"_date": "_start"})
        td = td.merge(starts, on=["_td_year", "_td_slug"])
        td["_offset"] = (td["_date"] - td["_start"]).dt.days

        # Map tennis-data Round strings to our codes; refine 1st Round by
        # draw size. We don't have draw size directly per row; use Series
        # field as a proxy.
        def _td_round_to_code(row) -> Optional[str]:
            r = str(row.get("Round", "")).strip().lower()
            series = str(row.get("Series", "")).strip().lower()
            # Slam: 1st Round = R128, 2nd = R64, 3rd = R32, 4th = R16
            if "grand slam" in series:
                slam_map = {
                    "1st round": "R128", "2nd round": "R64",
                    "3rd round": "R32",  "4th round": "R16",
                    "quarterfinals": "QF", "semifinals": "SF",
                    "the final": "F",
                }
                return slam_map.get(r)
            # Masters 1000: 1st Round = R64 (some have R128 in 96-draws)
            if "masters" in series:
                m_map = {
                    "1st round": "R64", "2nd round": "R32",
                    "3rd round": "R16", "quarterfinals": "QF",
                    "semifinals": "SF", "the final": "F",
                }
                return m_map.get(r)
            # Default 250/500: 1st = R32
            return _TD_ROUND_MAP.get(r)

        td["_round_code"] = td.apply(_td_round_to_code, axis=1)

        # Bucket by Series → level
        def _series_to_level(s):
            s = str(s).strip().lower()
            if "grand slam" in s:
                return "G"
            if "masters" in s:
                return "M"
            if "500" in s:
                return "500"
            if "250" in s:
                return "250"
            return "A"

        td["_level"] = td["Series"].apply(_series_to_level)

        td = td.dropna(subset=["_round_code"])
        offsets: Dict[Tuple[str, str], int] = {}
        for (lvl, rd), group in td.groupby(["_level", "_round_code"]):
            offsets[(lvl, rd)] = int(group["_offset"].median())
        return offsets

    # ------------------------------------------------------------------
    # Per-row resolver
    # ------------------------------------------------------------------

    def _resolve_match_date(
        self,
        tml_row: pd.Series,
        exact_lookup: Dict,
        empirical_offsets: Dict,
    ) -> Tuple[Optional[pd.Timestamp], str]:
        """
        Resolve a single TML row's match_date.

        Returns (timestamp.date(), confidence) where confidence is one of:
          "exact"      — direct hit in tennis-data
          "empirical"  — inferred from tennis-data offsets for same level/round
          "heuristic"  — hard-coded round-offset fallback
        """
        tourney_date = pd.to_datetime(tml_row["tourney_date"])
        year = tourney_date.year
        winner_key = _normalize_name(tml_row.get("winner_name", ""))
        loser_key = _normalize_name(tml_row.get("loser_name", ""))
        tournament_key = _normalize_tournament(tml_row.get("tourney_name", ""))
        slug = TENNIS_DATA_SLUGS.get(tournament_key)

        # 1. Exact lookup
        if slug and winner_key and loser_key:
            key1 = (winner_key, loser_key, year, slug)
            key2 = (loser_key, winner_key, year, slug)  # tennis-data records winner first; fallback if order differs
            if key1 in exact_lookup:
                return (exact_lookup[key1].date(), "exact")
            if key2 in exact_lookup:
                return (exact_lookup[key2].date(), "exact")

        # 2. Empirical from tennis-data offsets
        level = _level_bucket(tml_row.get("tourney_level", "A"))
        round_ = str(tml_row.get("round", ""))
        offset = empirical_offsets.get((level, round_))
        if offset is not None:
            return ((tourney_date + timedelta(days=offset)).date(), "empirical")

        # 3. Check for irregular format BEFORE falling to heuristic.
        # These tournaments have non-standard scheduling — heuristic offsets
        # would be wrong by multiple days and the model is not calibrated
        # for these event types regardless.
        raw_level = str(tml_row.get("tourney_level", ""))
        if raw_level in IRREGULAR_FORMAT_LEVELS or tournament_key in IRREGULAR_FORMAT_TOURNAMENTS:
            return (tourney_date.date(), "irregular_format")

        # 4. Heuristic table
        offset = HEURISTIC_ROUND_OFFSETS.get((level, round_))
        if offset is not None:
            return ((tourney_date + timedelta(days=offset)).date(), "heuristic")

        # 5. Last resort: tourney_date itself
        return (tourney_date.date(), "heuristic")


# ============================================================================
# Dummy data tests
# ============================================================================

def _test_normalize_name():
    print("=" * 60)
    print("normalize_name TESTS")
    print("=" * 60)

    cases = [
        ("Luciano Darderi", "darderi"),
        ("Darderi L.", "darderi"),
        ("De Minaur A.", "minaur"),  # imperfect, documented
        ("Juan Manuel Cerundolo", "cerundolo"),
        ("", ""),
        (None, ""),
    ]
    for inp, expected in cases:
        got = _normalize_name(inp)
        ok = "✓" if got == expected else "✗"
        print(f"  {ok}  _normalize_name({inp!r:35}) = {got!r:15} (expected {expected!r})")
        assert got == expected
    print("  PASSED")


def _test_resolve_match_date_exact():
    print("\n" + "=" * 60)
    print("resolve_match_date — EXACT path")
    print("=" * 60)

    loader = TMLMatchLoader()
    exact_lookup = {
        ("darderi", "cerundolo", 2024, "madrid"): pd.Timestamp("2024-04-25"),
    }
    row = pd.Series({
        "winner_name":    "Luciano Darderi",
        "loser_name":     "Juan Manuel Cerundolo",
        "tourney_name":   "Madrid",
        "tourney_level":  "M",
        "tourney_date":   pd.Timestamp("2024-04-22"),
        "round":          "R64",
    })
    date, conf = loader._resolve_match_date(row, exact_lookup, {})
    print(f"  Input:  Darderi vs Cerundolo, Madrid, 2024-04-22 week")
    print(f"  Output: {date}, confidence={conf!r}")
    assert date == pd.Timestamp("2024-04-25").date()
    assert conf == "exact"
    print("  PASSED ✓")


def _test_resolve_match_date_empirical():
    print("\n" + "=" * 60)
    print("resolve_match_date — EMPIRICAL path (tennis-data has offsets but")
    print("                                     not this exact pair)")
    print("=" * 60)

    loader = TMLMatchLoader()
    empirical = {("M", "R32"): 4}  # learned: Masters R32 happens day 4
    row = pd.Series({
        "winner_name":    "Some Player",
        "loser_name":     "Other Player",
        "tourney_name":   "Madrid",
        "tourney_level":  "M",
        "tourney_date":   pd.Timestamp("2024-04-22"),
        "round":          "R32",
    })
    date, conf = loader._resolve_match_date(row, {}, empirical)
    print(f"  Tourney start Mon 2024-04-22, R32 expected ~+4 days")
    print(f"  Output: {date}, confidence={conf!r}")
    assert date == pd.Timestamp("2024-04-26").date()
    assert conf == "empirical"
    print("  PASSED ✓")


def _test_resolve_match_date_heuristic():
    print("\n" + "=" * 60)
    print("resolve_match_date — HEURISTIC fallback")
    print("=" * 60)

    loader = TMLMatchLoader()
    row = pd.Series({
        "winner_name":    "Player A",
        "loser_name":     "Player B",
        "tourney_name":   "Some Obscure 250",
        "tourney_level":  "250",
        "tourney_date":   pd.Timestamp("2024-06-03"),
        "round":          "QF",
    })
    date, conf = loader._resolve_match_date(row, {}, {})
    expected = pd.Timestamp("2024-06-03") + timedelta(days=4)  # 250 QF = +4
    print(f"  ATP 250 QF, no tennis-data → heuristic offset +4 days")
    print(f"  Output: {date}, confidence={conf!r}")
    assert date == expected.date()
    assert conf == "heuristic"
    print("  PASSED ✓")


def _test_normalize_dummy_tml():
    print("\n" + "=" * 60)
    print("normalize() with dummy TML data and pre-built lookup tables")
    print("=" * 60)

    loader = TMLMatchLoader()
    raw = pd.DataFrame([
        {
            "tourney_id":     "2024-1536",
            "match_num":      29,
            "tourney_name":   "Madrid",
            "tourney_level":  "M",
            "tourney_date":   pd.Timestamp("2024-04-22"),
            "surface":        "Clay",
            "indoor":         "O",
            "round":          "R64",
            "winner_name":    "Luciano Darderi",
            "loser_name":     "Juan Manuel Cerundolo",
            "minutes":        117.0,
            "winner_rank":    63.0,
            "loser_rank":     85.0,
            "winner_rank_points": 920.0,
            "loser_rank_points":  720.0,
            "score":          "6-3 6-2",
            "best_of":        3,
            "winner_age":     22.1, "loser_age": 23.4,
            "winner_hand":    "R", "loser_hand": "R",
            "winner_ht":      183.0, "loser_ht": 178.0,
            "winner_ioc":     "ITA", "loser_ioc": "ARG",
            "w_ace": 5, "w_df": 2, "w_svpt": 60, "w_1stIn": 38,
            "w_1stWon": 30, "w_2ndWon": 15, "w_SvGms": 10,
            "w_bpSaved": 1, "w_bpFaced": 2,
            "l_ace": 3, "l_df": 4, "l_svpt": 60, "l_1stIn": 35,
            "l_1stWon": 22, "l_2ndWon": 12, "l_SvGms": 10,
            "l_bpSaved": 2, "l_bpFaced": 5,
        },
    ])

    # Pre-stub the lookups by patching fetch_tennis_data to return empty
    # (forces heuristic path so we don't hit network in tests)
    out = loader.normalize(raw, fetch_tennis_data=False)

    print("Output schema:")
    for c in out.columns:
        v = out[c].iloc[0]
        print(f"  {c:25} = {v!r}")

    assert out["player_a"].iloc[0] == "Luciano Darderi"
    assert out["player_b"].iloc[0] == "Juan Manuel Cerundolo"
    assert out["surface"].iloc[0] == "Clay"
    assert out["indoor"].iloc[0] == False
    assert out["round_"].iloc[0] == "R64"
    assert out["player_a_won"].iloc[0] == True
    # M + R64 heuristic offset = 2 days
    assert out["match_date"].iloc[0] == (
        pd.Timestamp("2024-04-22") + timedelta(days=2)
    ).date()
    assert out["date_confidence"].iloc[0] == "heuristic"
    print("\n  PASSED ✓")


def _test_cutoff_applied_to_match_date():
    print("\n" + "=" * 60)
    print("cutoff filter applied to match_date (not tourney_date)")
    print("=" * 60)

    loader = TMLMatchLoader()
    raw = pd.DataFrame([
        {  # Madrid R128 = +1 (Apr 23) — kept (cutoff Apr 24)
            "tourney_id": "2024-1536", "match_num": 1,
            "tourney_name": "Madrid", "tourney_level": "M",
            "tourney_date": pd.Timestamp("2024-04-22"),
            "surface": "Clay", "indoor": "O", "round": "R128",
            "winner_name": "A", "loser_name": "B",
        },
        {  # Madrid R32 = +4 (Apr 26) — dropped
            "tourney_id": "2024-1536", "match_num": 50,
            "tourney_name": "Madrid", "tourney_level": "M",
            "tourney_date": pd.Timestamp("2024-04-22"),
            "surface": "Clay", "indoor": "O", "round": "R32",
            "winner_name": "C", "loser_name": "D",
        },
    ])

    loader._cutoff_date = pd.Timestamp("2024-04-24")
    out = loader.normalize(raw, fetch_tennis_data=False)
    print(f"  Rows remaining after cutoff 2024-04-24: {len(out)} (expected 1)")
    print(f"  Surviving match_date: {out['match_date'].iloc[0]}")
    assert len(out) == 1
    assert out["match_date"].iloc[0] == pd.Timestamp("2024-04-23").date()
    print("  PASSED ✓")


def _test_irregular_format_flagged():
    print("\n" + "=" * 60)
    print("irregular_format — flagged in normalize()")
    print("=" * 60)

    loader = TMLMatchLoader()
    raw = pd.DataFrame([
        # Davis Cup match — should get irregular_format
        {
            "tourney_id": "2024-D001", "match_num": 1,
            "tourney_name": "Davis Cup", "tourney_level": "D",
            "tourney_date": pd.Timestamp("2024-09-10"),
            "surface": "Hard", "indoor": "I", "round": "RR",
            "winner_name": "Player A", "loser_name": "Player B",
        },
        # ATP Cup — should get irregular_format
        {
            "tourney_id": "2024-AC01", "match_num": 1,
            "tourney_name": "ATP Cup", "tourney_level": "A",
            "tourney_date": pd.Timestamp("2024-01-01"),
            "surface": "Hard", "indoor": "I", "round": "RR",
            "winner_name": "Player C", "loser_name": "Player D",
        },
        # Tour Finals — should get irregular_format
        {
            "tourney_id": "2024-F001", "match_num": 1,
            "tourney_name": "Tour Finals", "tourney_level": "F",
            "tourney_date": pd.Timestamp("2024-11-10"),
            "surface": "Hard", "indoor": "I", "round": "RR",
            "winner_name": "Player E", "loser_name": "Player F",
        },
        # Normal match — should get heuristic (not in tennis-data)
        {
            "tourney_id": "2024-M001", "match_num": 1,
            "tourney_name": "Some 250", "tourney_level": "250",
            "tourney_date": pd.Timestamp("2024-03-04"),
            "surface": "Hard", "indoor": "O", "round": "R32",
            "winner_name": "Player G", "loser_name": "Player H",
        },
    ])

    out = loader.normalize(raw, fetch_tennis_data=False)
    print(f"  date_confidence values: {list(out['date_confidence'])}")
    assert out.iloc[0]["date_confidence"] == "irregular_format", "Davis Cup should be irregular_format"
    assert out.iloc[1]["date_confidence"] == "irregular_format", "ATP Cup should be irregular_format"
    assert out.iloc[2]["date_confidence"] == "irregular_format", "Tour Finals should be irregular_format"
    assert out.iloc[3]["date_confidence"] == "heuristic",        "Normal match should be heuristic"
    print("  PASSED ✓")


def _test_check_inference_safe():
    print("\n" + "=" * 60)
    print("check_inference_safe — hard betting guard")
    print("=" * 60)

    # These should raise
    for tourney, level in [
        ("Davis Cup", "D"),
        ("Tour Finals", "F"),
        ("Tokyo Olympics", "O"),
        ("ATP Cup", "A"),
        ("Laver Cup", "A"),
    ]:
        try:
            TMLMatchLoader.check_inference_safe(tourney, level)
            print(f"  FAILED — should have raised for {tourney}")
        except IrregularFormatError as e:
            print(f"  ✓ Blocked {tourney!r} correctly")

    # These should pass
    for tourney, level in [
        ("Madrid", "M"),
        ("Australian Open", "G"),
        ("Rotterdam", "500"),
        ("Delray Beach", "250"),
    ]:
        result = TMLMatchLoader.check_inference_safe(tourney, level)
        assert result is True
        print(f"  ✓ Allowed {tourney!r} correctly")

    # raise_on_unsafe=False returns False instead of raising
    result = TMLMatchLoader.check_inference_safe("Davis Cup", "D", raise_on_unsafe=False)
    assert result is False
    print(f"  ✓ raise_on_unsafe=False returns False correctly")
    print("  PASSED ✓")


if __name__ == "__main__":
    _test_normalize_name()
    _test_resolve_match_date_exact()
    _test_resolve_match_date_empirical()
    _test_resolve_match_date_heuristic()
    _test_normalize_dummy_tml()
    _test_cutoff_applied_to_match_date()
    _test_irregular_format_flagged()
    _test_check_inference_safe()
    print("\n" + "=" * 60)
    print("All TMLMatchLoader tests passed.")
    print("=" * 60)
