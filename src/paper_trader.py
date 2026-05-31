"""
src/paper_trader.py

PaperTrader — runs the trained Theo model live against currently-open
Kalshi challenger markets, records best-edge bets to a paper-trade log,
and settles them once Kalshi resolves the underlying matches.

PUBLIC API
----------
    PaperTrader(model_path=..., log_dir=..., tml_df=None)

    .scan()             — pull open markets, evaluate, append eligible bets
                          to pending.csv. Markets that fail any filter are
                          appended to dropped.csv with a reason code.
    .settle_pending()   — for each row in pending.csv, refetch the underlying
                          Kalshi market; if it has resolved, compute realized
                          PnL net of fees and move the row to settled.csv.

LOG FILES (data/paper_trades/)
------------------------------
    pending.csv   — currently-open paper bets, one row each.
    settled.csv   — resolved bets with realized PnL.
    dropped.csv   — every market we inspected and rejected, with reason.

BET RULE (what scan() does)
---------------------------
For each match (mirror markets grouped by sorted (player_a_id, player_b_id)
and event_date):

  1. Build TWO synthetic TML rows — one per player perspective.
  2. Run the model on both → theo_a (P player_a wins), theo_b (P player_b wins).
  3. Enumerate the 4 candidate trades:
       - Buy YES on Market A:  cost = yes_ask_a,   wins iff player_a wins
       - Buy NO  on Market A:  cost = 1 - yes_bid_a, wins iff player_a loses
       - Buy YES on Market B:  cost = yes_ask_b,   wins iff player_b wins
       - Buy NO  on Market B:  cost = 1 - yes_bid_b, wins iff player_b loses
     For each, edge = (model's P bet wins) - cost.
  4. Pick the candidate with the largest edge.
  5. Eligibility: edge ≥ MIN_EDGE. If it passes, append to pending.csv.
     Otherwise drop with reason. (No tails-only filter — the model is
     best-calibrated in [0.3, 0.7] per the test-set reliability diagram,
     so restricting to extremes would only ever drop our most trustworthy
     bets. MIN_EDGE alone handles model-noise filtering.)

LOOKAHEAD / ORIENTATION
-----------------------
Synthetic rows are constructed so that winner_name = the side we're
asking about. matches_to_feature_matrix then produces theo for that side
directly — no orientation flip downstream. PlayerResolver upgrades
Kalshi's last-name-only player_b to TML's canonical full name AND emits
a stable player_id, so name-string drift can't silently NaN out features.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.loaders.market_match_joiner import _normalize_tournament
from src.loaders.player_resolver import PlayerResolver
from src.loaders.prediction_market_loader import KalshiLoader
from src.loaders.tml_match_loader import TMLMatchLoader
from src.ml.train import AUGMENTED_FEATURES, compute_feature_attribution, matches_to_feature_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants — bet rule and filters
# ============================================================================

MIN_EDGE              = 0.05   # 5¢ minimum edge after spread (model_p - cost)
MAX_SPREAD            = 0.50   # markets with (yes_ask - yes_bid) above this are
                               # treated as empty-book and skipped
KALSHI_FEE_PCT        = 0.07   # taker fee = 7% × p × (1-p) per contract
MAX_IMPUTED_FEATURES  = 3      # drop bet if >MAX features were NaN at scan time.
                               # Diagnostic from n=175 showed the 0.30+ edge
                               # bucket loses 63% — driven by bets where 10+ of
                               # 15 features are mean-imputed and rank_ratio_a
                               # dominates the (noisy) prediction.
MIN_TOURNEY_YEARS     = 3      # for Challenger events, require ≥N years of TML
                               # history at this tournament. Wuxi/Tunis/Oeiras 4
                               # / Francavilla had thin coverage and accounted
                               # for the bulk of losses.
MIN_PLAYER_COVERAGE   = 15     # require ≥N matches in TML in the 52w preceding
                               # event_date for BOTH players. Counterfactual on
                               # n=176 settled bets: 60 of 94 would-be-dropped
                               # bets failed this (typically cov_b=0 from a
                               # resolver miss). Catches the "phantom matchup"
                               # bets where the opponent has no recent record.
MAX_MIRROR_SUM_DEV    = 0.03   # require |yes_ask_a + yes_ask_b - 1.0| ≤ MAX.
                               # Tight mirrors = market-makers know what they
                               # quote. Loose mirrors (sum 1.04-1.12) are thin
                               # books with phantom edges; 55 of 94 would-be
                               # drops failed this gate alone.
MIN_OPEN_INTEREST     = 0.0    # placeholder — OI not yet plumbed from Kalshi
                               # normalize. Set to 500 once we surface it.
DROP_YES_ON_CHALLENGER = False # v2.2 direction-asymmetry guard — RE-DISABLED in
                               # v2.5 (2026-05-31), a same-day human revert of
                               # v2.4. The guard (drop every YES/back-the-favorite
                               # bet on a Challenger) does work — YES-on-Challenger
                               # ran −25.2% ROI on n=200 vs NO +52.6% on n=40 — but
                               # it patches a MODEL-CALIBRATION deficiency at the
                               # execution layer: the model overrates favorites on
                               # thin Challenger fields, and dropping those bets
                               # masks that symptom rather than fixing it. With all
                               # live flow on Challengers the guard also turned the
                               # bot NO-only. Deliberate decision (see BET_RULES
                               # v2.5): re-enable YES betting and address the
                               # overconfidence in the model itself (better
                               # calibration / Elo features / retrain), not by
                               # vetoing a whole bet direction. The gate code below
                               # is left dormant behind this flag so it stays a
                               # one-line flip if we ever want it back.

# Canonical tier code per Kalshi series. This is the SINGLE SOURCE OF TRUTH
# for `tourney_level` on every scan row: previously we inferred it from a
# TML mode-lookup on tournament name (see `_infer_surface_and_level`), which
# silently mislabeled name-collision tournaments. Now the scan path looks
# `tier` up directly from the market's `kalshi_series` and stamps it on the
# synthetic TML row + on the persisted bet. Surface still comes from TML
# (we can't read surface off the Kalshi ticker), but tier does not.
#
# Add a new entry here whenever Kalshi launches a new series we want to
# trade. TML's own code book is {C, 250, 500, M, G, A, O, F, D} — pick the
# matching code so that the model (which was trained on TML's `tourney_level`)
# sees a value it has actually seen during training.
TIER_FROM_SERIES = {
    "KXATPCHALLENGERMATCH": "C",
    # KXATPMATCH is the main-tour ATP series. It spans 250 / 500 / Masters /
    # Slams — the right tier depends on the specific event, not the series.
    # Until we wire per-event lookup, leave it `None` and fall back to the
    # TML mode-lookup; safe because main-tour name collisions are rare.
    "KXATPMATCH": None,
}

# Bet-rule version — stamped on every recorded bet so weekly_report can
# slice PnL by rule version without timestamp math. See BET_RULES.md
# at the repo root for the full version history & what each cut changed.
GATE_VERSION          = "v2.5"

DEFAULT_MODEL_PATH = REPO / "data" / "processed" / "model_augmented_beta.pkl"
DEFAULT_LOG_DIR    = REPO / "data" / "paper_trades"

# ── Champion / challenger SHADOW A/B plumbing ──────────────────────────────
# A "challenger" is a candidate Theo the model-research agent built (see
# docs/MODEL_RESEARCH_AGENT.md). When one is registered, EVERY scan also runs
# the challenger over the EXACT SAME markets/timestamps/features the champion
# saw and logs its would-be bets to a separate shadow CSV — without ever
# touching the live paper-trade logs or placing a real bet. This is what lets
# the weekly review score challenger calibration vs champion on forward,
# out-of-sample data. When no challenger is registered the whole path is a
# complete no-op.
#
# The registry is a TRACKED json file (so the active slot is auditable);
# the pickle it points to and the shadow CSVs are GITIGNORED (they are data,
# per CLAUDE.md #4).
ACTIVE_CHALLENGER_PATH = REPO / "data" / "research" / "active_challenger.json"
SHADOW_DIR             = REPO / "data" / "research" / "shadow"


# Reason codes for dropped markets — kept short so they're greppable in the CSV.
REASON_MISSING_ID   = "missing_player_id"
REASON_WIDE_SPREAD  = "wide_spread"
REASON_NO_TOURNEY   = "tournament_not_in_tml"
REASON_THIN_HISTORY  = "thin_player_history"
REASON_BELOW_EDGE    = "below_min_edge"
REASON_DUPLICATE     = "duplicate_match"
REASON_HIGH_IMPUTED  = "high_imputation"
REASON_THIN_TOURNEY  = "thin_tournament_history"
REASON_LOW_COVERAGE  = "low_player_coverage"
REASON_LOOSE_MIRROR  = "loose_mirror_sum"
REASON_YES_ON_CHALL  = "yes_on_challenger"


# ============================================================================
# Helpers
# ============================================================================

def _kalshi_url(ticker: str) -> str:
    """
    Build a deep link to the Kalshi UI for a given market ticker.

    Kalshi tickers look like "KXATPCHALLENGERMATCH-26MAY03MARPAO-MAR":
      - series       : everything before the first dash
      - event ticker : everything except the trailing player segment
    The UI uses lowercase paths.
    """
    if not isinstance(ticker, str) or "-" not in ticker:
        return ""
    parts = ticker.split("-")
    series = parts[0].lower()
    event = "-".join(parts[:-1]).lower()
    return f"https://kalshi.com/markets/{series}/{event}"


def _match_key(market_id) -> Optional[str]:
    """
    Stable key identifying a match across mirror markets and re-scans.

    Uses the Kalshi event-ticker prefix — i.e. the ticker with its trailing
    per-player segment removed. Mirror markets for the same match share this
    prefix (e.g. KXATPCHALLENGERMATCH-26MAY07GEEVAS-GEE and -VAS both have
    prefix KXATPCHALLENGERMATCH-26MAY07GEEVAS).

    Robust to PlayerResolver inconsistencies that resolve the same Kalshi
    player to different TML ids across mirrors (e.g. the Vasa brothers, when
    Kalshi sends "Eero Vasa" in one mirror's title and "Iiro Vasa" in the
    other). Player-id-based keys would emit two bets on one real match;
    ticker-prefix dedup catches it.

    Returns None if market_id is malformed or missing.
    """
    if not isinstance(market_id, str) or not market_id:
        return None
    ticker = market_id.replace("kalshi::", "")
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    return "-".join(parts[:-1])


def _pick_primary_mirror(group: list) -> dict:
    """
    Choose the cleanest mirror from a group sharing a match_key.

    Kalshi mirrors sometimes disagree on metadata (e.g. one has both full names,
    another has a last-name-only player_b that the resolver maps to the wrong
    sibling). Score each mirror by: (1) both player_ids non-null, (2) player_a
    has a first name, (3) player_b has a first name. Highest score wins; ties
    broken by original group order so re-runs are deterministic.
    """
    def score(m: dict) -> int:
        s = 0
        if pd.notna(m.get("player_a_id")) and pd.notna(m.get("player_b_id")):
            s += 1
        pa = m.get("player_a")
        pb = m.get("player_b")
        if isinstance(pa, str) and len(pa.split()) > 1:
            s += 1
        if isinstance(pb, str) and len(pb.split()) > 1:
            s += 1
        return s

    best_idx = 0
    best_score = score(group[0])
    for i in range(1, len(group)):
        s = score(group[i])
        if s > best_score:
            best_score = s
            best_idx = i
    return group[best_idx]


def _series_from_market_id(market_id) -> str:
    """
    Extract the Kalshi series ticker prefix from a fully-qualified market_id.

    Examples:
        "kalshi::KXATPCHALLENGERMATCH-26MAY12ESTCOL-EST" → "KXATPCHALLENGERMATCH"
        "kalshi::KXATPMATCH-26APR25CERDAR-DAR"           → "KXATPMATCH"
        "KXATPCHALLENGERMATCH-..."                       → "KXATPCHALLENGERMATCH"

    Returns "" if the input is empty/NaN or has no recognisable prefix.
    The series ticker is the only reliable scan-time tier signal — tier
    inferred from tournament-name mode-lookup in TML is unreliable for
    name-collision tournaments (see v2.3 in BET_RULES.md).
    """
    if market_id is None:
        return ""
    s = str(market_id)
    if not s or s.lower() == "nan":
        return ""
    # Drop the "kalshi::" prefix if present.
    if "::" in s:
        s = s.split("::", 1)[1]
    # The series ticker is everything before the first '-'.
    return s.split("-", 1)[0]


def _kalshi_fee(entry_price: float) -> float:
    """Kalshi taker fee per contract: 7% × p × (1-p)."""
    if pd.isna(entry_price):
        return np.nan
    p = float(entry_price)
    return KALSHI_FEE_PCT * p * (1.0 - p)


def _build_synthetic_row(
    *, market_id, player_a, player_b, player_a_id, player_b_id,
    tournament, surface, tourney_level, event_date,
    rank_a, rank_b, round_,
) -> dict:
    """
    Construct a TML-shape row so compute_all can produce features for the
    upcoming match. winner_name = the player we want a probability for;
    set winner_rank / loser_rank from latest TML lookups (NaN if no history).
    Stat columns are NaN — compute_all uses prior-match stats only, never
    this row's own.
    """
    return {
        "tml_match_id":      f"synthetic::{market_id}::{player_a_id}",
        "winner_name":       player_a,
        "loser_name":        player_b,
        "winner_id":         player_a_id,
        "loser_id":          player_b_id,
        "match_date":        pd.Timestamp(event_date),
        "tournament":        tournament,
        "tourney_level":     tourney_level if tourney_level is not None else "C",
        "surface":           surface,
        "round_":            round_,
        "indoor":            False,
        "minutes":           np.nan,
        "winner_rank":       rank_a,
        "loser_rank":        rank_b,
        "winner_rank_points": np.nan, "loser_rank_points": np.nan,
        "winner_age":        np.nan, "loser_age": np.nan,
        "winner_hand":       "U",   "loser_hand": "U",
        "winner_ht":         np.nan, "loser_ht": np.nan,
        "winner_ioc":        "",     "loser_ioc": "",
        "score":             "",     "best_of": 3,
        "w_ace": np.nan, "w_df": np.nan, "w_svpt": np.nan,
        "w_1stIn": np.nan, "w_1stWon": np.nan, "w_2ndWon": np.nan,
        "w_SvGms": np.nan, "w_bpSaved": np.nan, "w_bpFaced": np.nan,
        "l_ace": np.nan, "l_df": np.nan, "l_svpt": np.nan,
        "l_1stIn": np.nan, "l_1stWon": np.nan, "l_2ndWon": np.nan,
        "l_SvGms": np.nan, "l_bpSaved": np.nan, "l_bpFaced": np.nan,
    }


# ============================================================================
# PaperTrader
# ============================================================================

class PaperTrader:

    PENDING_COLS = [
        "timestamp_recorded", "match_key", "market_id", "kalshi_url",
        # Kalshi series ticker for the market (e.g. "KXATPCHALLENGERMATCH",
        # "KXATPMATCH"). This is the ONLY scan-time source of truth for
        # market tier — `tourney_level` further down is an inferred-from-TML
        # mode lookup on tournament name and can mislabel Challenger markets
        # whose name collides with a tour-level event (added in v2.3 after
        # auto-review 2026-05-20 found Cordoba Challengers labeled as 250).
        "kalshi_series",
        "player_a", "player_a_id", "player_b", "player_b_id",
        "tournament", "surface", "tourney_level", "event_date",
        # Full timestamps for cadence analysis — derived from Kalshi metadata.
        # `market_open_time` = when Kalshi opened the contract for trading;
        # `match_start_time` = scheduled tip-off (occurrence_datetime). The
        # gap (timestamp_recorded - market_open_time) is "scan latency"; the
        # gap (match_start_time - timestamp_recorded) is "lead time before
        # match." Used to study whether scan timing affects PnL.
        "market_open_time", "match_start_time",
        "theo_a", "theo_b",
        "yes_ask_a", "yes_bid_a", "yes_ask_b", "yes_bid_b",
        "chosen_market_id", "chosen_direction",  # YES or NO
        "chosen_player_name", "chosen_player_id",  # the player whose win we're betting on
        "entry_price", "theo_chosen", "edge", "fee",
        # JSON dict of {feature_name: signed_log_odds_shift} from the
        # perspective of the player we're betting on. Frozen at scan time
        # so weekly_report attribution can't drift if the model is retrained.
        "feature_shifts_json",
        # Bet-rule version this bet was placed under. See BET_RULES.md.
        "gate_version",
    ]
    SETTLED_COLS = PENDING_COLS + [
        "timestamp_settled", "resolution", "bet_won", "gross_pnl", "net_pnl",
    ]
    DROPPED_COLS = [
        "timestamp", "market_id", "kalshi_series", "kalshi_url",
        "player_a", "player_a_id", "player_b", "player_b_id",
        "tournament", "event_date",
        "yes_ask", "yes_bid",
        "reason", "reason_detail",
    ]

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        log_dir: Path = DEFAULT_LOG_DIR,
        tml_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ):
        """
        Args:
            model_path: pickled sklearn pipeline that takes AUGMENTED_FEATURES
                        and returns calibrated P(player_a wins).
            log_dir:    where pending/settled/dropped CSVs live.
            tml_df:     pre-loaded TMLMatchLoader.normalize() output. If None,
                        loaded internally on construction (slow ~70s cold cache).
            verbose:    log progress lines during scan/settle.
        """
        self.verbose = verbose
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.pending_path = self.log_dir / "pending.csv"
        self.settled_path = self.log_dir / "settled.csv"
        self.dropped_path = self.log_dir / "dropped.csv"

        if tml_df is None:
            tml_df = self._load_tml()
        self.tml_df = (
            tml_df[tml_df["date_confidence"] != "irregular_format"]
            .reset_index(drop=True)
            .copy()
        )

        self.resolver = PlayerResolver(self.tml_df)
        self.kalshi = KalshiLoader(
            series_tickers=["KXATPCHALLENGERMATCH"],
            player_resolver=self.resolver,
        )
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # SHADOW A/B: load the active challenger model, if one is registered.
        # On any failure (missing registry, null id, missing/unloadable pickle)
        # this leaves challenger_id/model as None and the shadow path no-ops.
        # It must NEVER crash or degrade the live scan.
        self.challenger_id: Optional[str] = None
        self.challenger_model = None
        self.shadow_pending_path: Optional[Path] = None
        self.shadow_settled_path: Optional[Path] = None
        self._load_active_challenger()

        if verbose:
            logger.info(f"PaperTrader ready (model={model_path.name}, "
                        f"tml_rows={len(self.tml_df):,})")
            if self.challenger_id:
                logger.info(f"  shadow challenger active: {self.challenger_id}")

    @staticmethod
    def _load_tml() -> pd.DataFrame:
        loader = TMLMatchLoader()
        raw = loader.load(start_year=2018, end_year=2026, include_challenger=True)
        return loader.normalize(raw)

    # Shadow log column schemas mirror the champion's pending/settled schemas
    # exactly (so calibration is scored with identical fields), plus a
    # `challenger_id` provenance column.
    @property
    def SHADOW_PENDING_COLS(self) -> list:
        return ["challenger_id"] + self.PENDING_COLS

    @property
    def SHADOW_SETTLED_COLS(self) -> list:
        return ["challenger_id"] + self.SETTLED_COLS

    def _load_active_challenger(self) -> None:
        """
        Read data/research/active_challenger.json and, if it names a usable
        challenger, load its pickle via the SAME code path the champion uses
        (pickle.load) and wire up the shadow log path.

        No-op (and never raises) in every degraded case:
          - registry file absent
          - `challenger_id` null / empty / missing
          - `pickle` path missing / unreadable / not a valid model

        On a missing/unloadable pickle for a named challenger we log a WARNING
        and skip shadow entirely, so the live scan is never affected.
        """
        path = ACTIVE_CHALLENGER_PATH
        if not path.exists():
            return
        try:
            reg = json.loads(path.read_text())
        except Exception as e:
            logger.warning(f"shadow: could not parse {path.name}: {e} — "
                           "skipping challenger.")
            return

        cid = reg.get("challenger_id")
        if not cid or not str(cid).strip():
            # Explicit "no challenger" state — silent no-op.
            return
        cid = str(cid).strip()

        pkl_rel = reg.get("pickle")
        if not pkl_rel:
            logger.warning(f"shadow: challenger {cid!r} has no 'pickle' path — "
                           "skipping.")
            return
        pkl_path = Path(pkl_rel)
        if not pkl_path.is_absolute():
            pkl_path = REPO / pkl_path
        if not pkl_path.exists():
            logger.warning(f"shadow: challenger {cid!r} pickle not found at "
                           f"{pkl_path} — skipping shadow.")
            return
        try:
            with open(pkl_path, "rb") as f:
                challenger_model = pickle.load(f)
        except Exception as e:
            logger.warning(f"shadow: failed to load challenger {cid!r} pickle "
                           f"({pkl_path}): {e} — skipping shadow.")
            return

        self.challenger_id = cid
        self.challenger_model = challenger_model
        SHADOW_DIR.mkdir(parents=True, exist_ok=True)
        # `<cid>.csv` holds open would-be bets; `<cid>_settled.csv` holds the
        # resolved ones (mirrors the champion's pending/settled split so the
        # exact same settle logic can be reused).
        self.shadow_pending_path = SHADOW_DIR / f"{cid}.csv"
        self.shadow_settled_path = SHADOW_DIR / f"{cid}_settled.csv"

    # ------------------------------------------------------------------ #
    # scan()
    # ------------------------------------------------------------------ #

    def scan(self) -> dict:
        """
        Pull all currently-open challenger markets, run the model, and
        record best-edge bets that pass the tails + edge filters.

        Returns a small summary dict with counts. Side-effect is appending
        to pending.csv and dropped.csv.
        """
        t0 = time.time()
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # ── 1. Pull + normalize open markets ──────────────────────────────
        raw = self.kalshi.load(status="open", limit=500)
        if raw.empty:
            logger.info("scan(): no open challenger markets right now.")
            return {"scanned": 0, "bets": 0, "dropped": 0, "reasons": {}}
        norm = self.kalshi.normalize(raw)
        n_total = len(norm)

        # ── 2. Per-market filters (id resolution, spread) ─────────────────
        already_open_keys = self._load_existing_match_keys()
        kept_markets = []
        dropped_rows = []
        reason_counts: dict = {}

        for _, m in norm.iterrows():
            ticker = m.get("_ticker") or m.get("market_id", "").replace("kalshi::", "")
            url = _kalshi_url(ticker)
            base_drop = {
                "timestamp":   ts,
                "market_id":   m["market_id"],
                "kalshi_series": _series_from_market_id(m.get("market_id")),
                "kalshi_url":  url,
                "player_a":    m.get("player_a"),
                "player_a_id": m.get("player_a_id"),
                "player_b":    m.get("player_b"),
                "player_b_id": m.get("player_b_id"),
                "tournament":  m.get("tournament"),
                "event_date":  m.get("event_date"),
                "yes_ask":     m.get("yes_ask"),
                "yes_bid":     m.get("yes_bid"),
            }

            # Filter: missing player ids
            if pd.isna(m.get("player_a_id")) or pd.isna(m.get("player_b_id")):
                detail = (
                    f"player_a_id={m.get('player_a_id')!r}, "
                    f"player_b_id={m.get('player_b_id')!r}"
                )
                dropped_rows.append({**base_drop,
                                     "reason": REASON_MISSING_ID,
                                     "reason_detail": detail})
                reason_counts[REASON_MISSING_ID] = reason_counts.get(REASON_MISSING_ID, 0) + 1
                continue

            # Filter: wide spread (empty book)
            ask = m.get("yes_ask")
            bid = m.get("yes_bid")
            if pd.notna(ask) and pd.notna(bid) and (ask - bid) > MAX_SPREAD:
                dropped_rows.append({**base_drop,
                                     "reason": REASON_WIDE_SPREAD,
                                     "reason_detail": f"spread={ask-bid:.2f}"})
                reason_counts[REASON_WIDE_SPREAD] = reason_counts.get(REASON_WIDE_SPREAD, 0) + 1
                continue

            # Filter: duplicate (already have a bet on this match)
            mk = _match_key(m.get("market_id"))
            if mk and mk in already_open_keys:
                dropped_rows.append({**base_drop,
                                     "reason": REASON_DUPLICATE,
                                     "reason_detail": f"match_key={mk}"})
                reason_counts[REASON_DUPLICATE] = reason_counts.get(REASON_DUPLICATE, 0) + 1
                continue

            kept_markets.append(m)

        if self.verbose:
            logger.info(f"scan(): {n_total} markets → "
                        f"{len(kept_markets)} after per-market filters")

        # ── 3. Group remaining markets by match_key ───────────────────────
        match_groups: dict = {}
        for m in kept_markets:
            mk = _match_key(m["market_id"])
            match_groups.setdefault(mk, []).append(m)

        # ── 4a. Build synthetic rows for every match (BOTH perspectives) ──
        # We score in a single compute_all pass at the end, not per-match,
        # because compute_all rebuilds player history from the whole TML
        # corpus (~10s per call) and that work is shared across rows.
        synth_rows = []           # list[dict] — TML-shape synthetic rows
        synth_meta = {}           # synthetic_id -> match-evaluation context
        for mk, group in match_groups.items():
            primary = _pick_primary_mirror(group)
            tournament = primary.get("tournament")
            series = _series_from_market_id(primary.get("market_id"))
            tier_hint = TIER_FROM_SERIES.get(series)
            surface, level = self._infer_surface_and_level(tournament, tier=tier_hint)
            if surface is None:
                detail = f"tournament={tournament!r}"
                for m in group:
                    ticker = (m.get("_ticker")
                              or m.get("market_id", "").replace("kalshi::", ""))
                    dropped_rows.append({
                        "timestamp": ts, "market_id": m["market_id"],
                        "kalshi_series": _series_from_market_id(m.get("market_id")),
                        "kalshi_url": _kalshi_url(ticker),
                        "player_a": m.get("player_a"), "player_a_id": m.get("player_a_id"),
                        "player_b": m.get("player_b"), "player_b_id": m.get("player_b_id"),
                        "tournament": tournament, "event_date": m.get("event_date"),
                        "yes_ask": m.get("yes_ask"), "yes_bid": m.get("yes_bid"),
                        "reason": REASON_NO_TOURNEY, "reason_detail": detail,
                    })
                    reason_counts[REASON_NO_TOURNEY] = reason_counts.get(REASON_NO_TOURNEY, 0) + 1
                continue

            pa, pb = primary["player_a"], primary["player_b"]
            pa_id, pb_id = primary["player_a_id"], primary["player_b_id"]
            event_date = primary["event_date"]
            round_ = primary.get("round_")
            rank_a = self._latest_rank(pa_id, pa, event_date)
            rank_b = self._latest_rank(pb_id, pb, event_date)

            row_a = _build_synthetic_row(
                market_id=primary["market_id"], player_a=pa, player_b=pb,
                player_a_id=pa_id, player_b_id=pb_id,
                tournament=primary["tournament"], surface=surface, tourney_level=level,
                event_date=event_date, rank_a=rank_a, rank_b=rank_b, round_=round_,
            )
            row_b = _build_synthetic_row(
                market_id=primary["market_id"], player_a=pb, player_b=pa,
                player_a_id=pb_id, player_b_id=pa_id,
                tournament=primary["tournament"], surface=surface, tourney_level=level,
                event_date=event_date, rank_a=rank_b, rank_b=rank_a, round_=round_,
            )
            id_a = f"{row_a['tml_match_id']}::A"
            id_b = f"{row_b['tml_match_id']}::B"
            row_a["tml_match_id"] = id_a
            row_b["tml_match_id"] = id_b
            synth_rows.append(row_a)
            synth_rows.append(row_b)
            synth_meta[mk] = {
                "group":      group,
                "primary":    primary,
                "surface":    surface,
                "level":      level,
                "id_a":       id_a,
                "id_b":       id_b,
                "rank_a":     rank_a,
                "rank_b":     rank_b,
            }

        # ── 4b. One compute_all pass over (TML + all synthetic rows) ──────
        bets = []
        if synth_rows:
            t_feat = time.time()
            tml_renamed = self.tml_df.rename(columns={"player_a": "winner_name",
                                                       "player_b": "loser_name"})
            combined = pd.concat([tml_renamed, pd.DataFrame(synth_rows)],
                                 ignore_index=True, sort=False)
            t_concat = time.time()
            feats_all = matches_to_feature_matrix(combined)
            t_features = time.time()
            feat_idx = feats_all.set_index("tml_match_id")
            if self.verbose:
                logger.info(
                    f"  features: concat={t_concat-t_feat:.1f}s, "
                    f"compute_all={t_features-t_concat:.1f}s, "
                    f"matches={len(synth_meta)}, synth_rows={len(synth_rows)}"
                )

            for mk, meta in synth_meta.items():
                eval_result = self._score_match_from_features(
                    meta=meta, feat_idx=feat_idx, ts=ts,
                )
                if eval_result["status"] == "bet":
                    bets.append(eval_result["row"])
                elif eval_result["status"] == "drop":
                    dropped_rows.extend(eval_result["dropped"])
                    for d in eval_result["dropped"]:
                        reason_counts[d["reason"]] = reason_counts.get(d["reason"], 0) + 1

            # ── 4c. SHADOW A/B pass (no-op if no active challenger) ────────
            # Re-score the SAME matches/features with the challenger model and
            # log its would-be bets to the shadow CSV. This never touches
            # `bets`/`dropped_rows`/`reason_counts`, so the champion logs are
            # byte-for-byte identical to a no-challenger run.
            self._run_shadow_scan(synth_meta, feat_idx, ts)

        # ── 5. Persist ────────────────────────────────────────────────────
        if bets:
            self._append_csv(self.pending_path, bets, self.PENDING_COLS)
        if dropped_rows:
            self._append_csv(self.dropped_path, dropped_rows, self.DROPPED_COLS)
        self._refresh_markdown()

        if self.verbose:
            logger.info(
                f"scan() done in {time.time()-t0:.1f}s: "
                f"{n_total} scanned, {len(bets)} bets recorded, "
                f"{len(dropped_rows)} dropped"
            )
            for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
                logger.info(f"  drop[{reason}]: {cnt}")
        return {
            "scanned": n_total,
            "bets":    len(bets),
            "dropped": len(dropped_rows),
            "reasons": reason_counts,
        }

    # ------------------------------------------------------------------ #
    # _evaluate_match — synthetic rows + model + best-edge for one match
    # ------------------------------------------------------------------ #

    def _score_match_from_features(
        self,
        meta: dict,
        feat_idx: pd.DataFrame,
        ts: str,
        model=None,
    ) -> dict:
        """
        Score one match given the precomputed feature matrix indexed by
        tml_match_id. Apply the bet rule. Returns {"status": "bet", "row": ...}
        or {"status": "drop", "dropped": [...]} for the dropped.csv writer.

        `model` selects which Theo to score with. It defaults to the champion
        (`self.model`); the SHADOW A/B path passes the active challenger here so
        the challenger runs through the IDENTICAL feature assembly and bet gates
        as the champion (we are A/B-testing the model, not the bet rules). Every
        gate/threshold below is shared — only `predict_proba` and the feature
        attribution use the supplied model.
        """
        if model is None:
            model = self.model
        group = meta["group"]
        primary = meta["primary"]
        surface = meta["surface"]
        level = meta["level"]
        id_a, id_b = meta["id_a"], meta["id_b"]
        rank_a, rank_b = meta["rank_a"], meta["rank_b"]

        pa = primary["player_a"]
        pb = primary["player_b"]
        pa_id = primary["player_a_id"]
        pb_id = primary["player_b_id"]
        event_date = primary["event_date"]

        if id_a not in feat_idx.index or id_b not in feat_idx.index:
            return self._drop_group(group, ts, REASON_THIN_HISTORY,
                                    "synthetic rows missing from feature matrix")

        feat_a = feat_idx.loc[id_a, AUGMENTED_FEATURES]
        feat_b = feat_idx.loc[id_b, AUGMENTED_FEATURES]

        # Thin-history guard: if rank_ratio_a (the structural feature) is NaN
        # for either side, the mean-imputer fills it but the theo is junk.
        if pd.isna(feat_a["rank_ratio_a"]) or pd.isna(feat_b["rank_ratio_a"]):
            return self._drop_group(
                group, ts, REASON_THIN_HISTORY,
                f"rank_ratio_a NaN (rank_a={rank_a}, rank_b={rank_b})",
            )

        # Player-coverage gate: require both players to have ≥N matches in
        # the 52w preceding event_date. Catches the cov_b=0 failure mode
        # where the resolver mapped Kalshi's player_b to a TML id with no
        # recent record (counterfactual on n=176 settled bets: 60 / 94
        # would-be-dropped bets failed this leg). Strictly tighter than
        # high-imputation in many cases because imputation looks only at
        # AUGMENTED_FEATURES, while coverage looks at raw match presence.
        cov_a = self._player_coverage(pa_id, event_date)
        cov_b = self._player_coverage(pb_id, event_date)
        if cov_a < MIN_PLAYER_COVERAGE or cov_b < MIN_PLAYER_COVERAGE:
            return self._drop_group(
                group, ts, REASON_LOW_COVERAGE,
                f"cov_a={cov_a} cov_b={cov_b} "
                f"(min {MIN_PLAYER_COVERAGE})",
            )

        # High-imputation guard: if either player has too many features
        # mean-imputed, the model's theo is dominated by whatever is non-NaN
        # (usually just rank_ratio_a). This was the n=175 diagnostic finding —
        # bets in the 0.30+ edge bucket lost 63% ROI because they were
        # disproportionately matches where 10+ features were imputed.
        n_imp_a = int(feat_a.isna().sum())
        n_imp_b = int(feat_b.isna().sum())
        if n_imp_a > MAX_IMPUTED_FEATURES or n_imp_b > MAX_IMPUTED_FEATURES:
            return self._drop_group(
                group, ts, REASON_HIGH_IMPUTED,
                f"n_imputed_a={n_imp_a} n_imputed_b={n_imp_b} "
                f"(threshold {MAX_IMPUTED_FEATURES})",
            )

        # Tournament-history guard: for Challenger events specifically,
        # require ≥MIN_TOURNEY_YEARS of TML history at this tournament.
        # The "cursed 4" (Wuxi/Tunis/Oeiras 4/Francavilla) at n=175 had
        # thin or one-time appearances in TML and concentrated the losses.
        if level == "C":
            n_years = self._tournament_history_years(primary["tournament"])
            if n_years < MIN_TOURNEY_YEARS:
                return self._drop_group(
                    group, ts, REASON_THIN_TOURNEY,
                    f"{primary['tournament']!r} has {n_years} years in TML "
                    f"(need {MIN_TOURNEY_YEARS})",
                )

        # Predict
        X = np.vstack([feat_a.values.astype(float), feat_b.values.astype(float)])
        theos = model.predict_proba(X)[:, 1]
        theo_a, theo_b = float(theos[0]), float(theos[1])

        # Build the candidate-bet table — up to 4 (per market × YES/NO),
        # but only as many actual markets as the mirror group contains.
        candidates = []
        for m in group:
            ask = m.get("yes_ask")
            bid = m.get("yes_bid")
            # Determine which player this market is the YES side for
            if m["player_a_id"] == pa_id:
                theo_yes = theo_a
            elif m["player_a_id"] == pb_id:
                theo_yes = theo_b
            else:
                # Should never happen given the match_key grouping
                continue
            theo_no = 1.0 - theo_yes
            # Player we're betting on for this candidate:
            #   YES on market m  → bet that m["player_a"] wins
            #   NO  on market m  → bet that m["player_a"] loses (= player_b wins)
            yes_player_name = m["player_a"]
            yes_player_id   = m["player_a_id"]
            no_player_name  = m["player_b"]
            no_player_id    = m["player_b_id"]
            if pd.notna(ask):
                candidates.append({
                    "market_id":   m["market_id"],
                    "direction":   "YES",
                    "cost":        float(ask),
                    "theo":        theo_yes,
                    "edge":        theo_yes - float(ask),
                    "ask":         float(ask),
                    "bid":         float(bid) if pd.notna(bid) else np.nan,
                    "ticker":      m.get("_ticker"),
                    "player_name": yes_player_name,
                    "player_id":   yes_player_id,
                })
            if pd.notna(bid):
                candidates.append({
                    "market_id":   m["market_id"],
                    "direction":   "NO",
                    "cost":        1.0 - float(bid),
                    "theo":        theo_no,
                    "edge":        theo_no - (1.0 - float(bid)),
                    "ask":         float(ask) if pd.notna(ask) else np.nan,
                    "bid":         float(bid),
                    "ticker":      m.get("_ticker"),
                    "player_name": no_player_name,
                    "player_id":   no_player_id,
                })
        if not candidates:
            return self._drop_group(group, ts, REASON_WIDE_SPREAD,
                                    "no quoted ask/bid on any mirror")

        # Pick best edge
        best = max(candidates, key=lambda c: c["edge"])

        # v2.2 direction-asymmetry guard, re-enabled in v2.4 (see BET_RULES.md
        # v2.4 / auto-review 2026-05-31). Gates on the canonical `kalshi_series`
        # field — NOT `level`, which is a TML mode-lookup that mislabeled
        # name-collision Challengers and got v2.2 reverted. YES picks on
        # Challenger markets ran −25.2% ROI on n=200 (Wilson-95 CI excludes the
        # 0.528 avg theo); NO-on-Challenger (+52.6%, n=40) and any future
        # main-tour KXATPMATCH series are preserved. Both `direction` and
        # `series` are scan-time known (no lookahead).
        series = _series_from_market_id(primary["market_id"])
        if DROP_YES_ON_CHALLENGER and best["direction"] == "YES" \
                and series == "KXATPCHALLENGERMATCH":
            return self._drop_group(
                group, ts, REASON_YES_ON_CHALL,
                f"best={best['direction']} on {series} "
                f"(theo={best['theo']:.3f}, edge={best['edge']:.3f})",
            )

        # Eligibility filters
        if best["edge"] < MIN_EDGE:
            return self._drop_group(
                group, ts, REASON_BELOW_EDGE,
                f"best_edge={best['edge']:.3f} on {best['direction']} @ "
                f"theo={best['theo']:.3f}",
            )

        # Collect quotes from both mirrors before the mirror-sum gate.
        ask_a = bid_a = ask_b = bid_b = np.nan
        for m in group:
            if m["player_a_id"] == pa_id:
                ask_a, bid_a = m.get("yes_ask"), m.get("yes_bid")
            elif m["player_a_id"] == pb_id:
                ask_b, bid_b = m.get("yes_ask"), m.get("yes_bid")

        # Mirror-sum gate: yes_ask_A + yes_ask_B should sum to ~1.0 (plus
        # a small spread) for tight, well-quoted books. Loose mirrors
        # (sum 1.04-1.12) signal phantom edges from thin liquidity; 55 of
        # 94 would-be-dropped bets in the counterfactual failed this leg.
        # Skip the check if either ask is missing (single-sided market).
        if pd.notna(ask_a) and pd.notna(ask_b):
            mirror_dev = abs((float(ask_a) + float(ask_b)) - 1.0)
            if mirror_dev > MAX_MIRROR_SUM_DEV:
                return self._drop_group(
                    group, ts, REASON_LOOSE_MIRROR,
                    f"yes_ask_a={ask_a:.2f} + yes_ask_b={ask_b:.2f} "
                    f"= {ask_a+ask_b:.3f} (dev {mirror_dev:.3f} > "
                    f"{MAX_MIRROR_SUM_DEV})",
                )

        # Per-feature signed log-odds shifts, in the perspective of the
        # player we're betting ON winning (so positive = pushed toward bet).
        # We use the synthetic row whose winner_name == that player.
        bet_feat = feat_a if best["player_id"] == pa_id else feat_b
        try:
            shifts = compute_feature_attribution(
                model, bet_feat.values.astype(float), AUGMENTED_FEATURES,
            )
            shifts_json = json.dumps({k: round(v, 6) for k, v in shifts.items()})
        except Exception as e:
            logger.warning(f"feature attribution failed: {e}")
            shifts_json = ""

        # Pull the timestamps from whichever mirror has them (both should
        # agree across mirrors of the same match). Coerce to ISO strings
        # for clean CSV round-tripping.
        def _iso(v):
            if v is None or pd.isna(v):
                return ""
            return pd.Timestamp(v).isoformat()
        market_open = next((m.get("_open_time") for m in group
                            if pd.notna(m.get("_open_time"))), None)
        match_start = next((m.get("_event_datetime") for m in group
                            if pd.notna(m.get("_event_datetime"))), None)

        row = {
            "timestamp_recorded":   ts,
            "match_key":            _match_key(primary["market_id"]),
            "market_id":            best["market_id"],          # the market we BET on
            "kalshi_series":        _series_from_market_id(best["market_id"]),
            "kalshi_url":           _kalshi_url(best.get("ticker")),
            "player_a":             pa,  "player_a_id": pa_id,
            "player_b":             pb,  "player_b_id": pb_id,
            "tournament":           primary["tournament"],
            "surface":              surface,
            "tourney_level":        level,
            "event_date":           event_date,
            "market_open_time":     _iso(market_open),
            "match_start_time":     _iso(match_start),
            "theo_a":               theo_a,
            "theo_b":               theo_b,
            "yes_ask_a":            ask_a, "yes_bid_a": bid_a,
            "yes_ask_b":            ask_b, "yes_bid_b": bid_b,
            "chosen_market_id":     best["market_id"],
            "chosen_direction":     best["direction"],
            "chosen_player_name":   best["player_name"],
            "chosen_player_id":     best["player_id"],
            "entry_price":          best["cost"],
            "theo_chosen":          best["theo"],
            "edge":                 best["edge"],
            "fee":                  _kalshi_fee(best["cost"]),
            "feature_shifts_json":  shifts_json,
            "gate_version":         GATE_VERSION,
        }
        return {"status": "bet", "row": row}

    # ------------------------------------------------------------------ #
    # SHADOW A/B — challenger scoring on the same scan
    # ------------------------------------------------------------------ #

    def _run_shadow_scan(
        self,
        synth_meta: dict,
        feat_idx: pd.DataFrame,
        ts: str,
    ) -> None:
        """
        Score the active challenger over the EXACT same matches/timestamps/
        features the champion just saw, and append its would-be bets to the
        shadow CSV. Complete no-op when no challenger is active.

        Reuses `_score_match_from_features` with the challenger model swapped
        in, so the challenger passes through the identical feature assembly and
        identical bet gates as the champion — we A/B-test the MODEL, not the
        bet rules. The challenger NEVER writes the champion's logs and NEVER
        places a real bet. Wrapped in a broad try/except so a challenger fault
        can never break the live scan.
        """
        if not self.challenger_id or self.challenger_model is None:
            return
        try:
            already_open = self._load_match_keys(self.shadow_pending_path)
            shadow_bets = []
            for mk, meta in synth_meta.items():
                # Same dedup discipline the champion uses: one open shadow bet
                # per match_key.
                match_key = _match_key(meta["primary"].get("market_id"))
                if match_key and match_key in already_open:
                    continue
                result = self._score_match_from_features(
                    meta=meta, feat_idx=feat_idx, ts=ts,
                    model=self.challenger_model,
                )
                if result["status"] == "bet":
                    row = {"challenger_id": self.challenger_id, **result["row"]}
                    shadow_bets.append(row)
            if shadow_bets:
                self._append_csv(self.shadow_pending_path, shadow_bets,
                                 self.SHADOW_PENDING_COLS)
            if self.verbose:
                logger.info(f"  shadow[{self.challenger_id}]: "
                            f"{len(shadow_bets)} would-be bets logged")
        except Exception as e:
            # Never let a challenger fault degrade the live scan.
            logger.warning(f"shadow scan for {self.challenger_id!r} failed: {e}")

    # ------------------------------------------------------------------ #
    # settle_pending()
    # ------------------------------------------------------------------ #

    def settle_pending(self) -> dict:
        """
        Refetch settled Kalshi markets and resolve any pending paper bets.
        Settled rows are appended to settled.csv and removed from pending.
        """
        champ_has_pending = self.pending_path.exists()
        if champ_has_pending:
            pending = pd.read_csv(self.pending_path)
            champ_has_pending = not pending.empty
        # The shadow log may have open rows even when the champion does not
        # (e.g. a challenger that bets matches the champion gated out). Only
        # bail entirely if NEITHER has anything to settle.
        shadow_has_pending = (
            self.challenger_id
            and self.shadow_pending_path is not None
            and self.shadow_pending_path.exists()
        )
        if not champ_has_pending and not shadow_has_pending:
            return {"settled": 0, "still_open": 0}

        # Pull settled markets. force_refresh bypasses KalshiLoader's
        # disk cache — without it, the cache would return yesterday's
        # snapshot and today's resolutions would be invisible.
        raw_settled = self.kalshi.load(status="settled", limit=2000, force_refresh=True)
        if raw_settled.empty:
            n_open = int(len(pending)) if champ_has_pending else 0
            return {"settled": 0, "still_open": n_open}
        norm_settled = self.kalshi.normalize(raw_settled)
        res_by_market = norm_settled.set_index("market_id")["resolution"]

        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if champ_has_pending:
            newly_settled, still_open = self._resolve_rows(pending, res_by_market, ts)
        else:
            newly_settled, still_open = [], []

        # Persist
        if newly_settled:
            self._append_csv(self.settled_path, newly_settled, self.SETTLED_COLS)
        # Rewrite pending with only the still-open rows. Backfill any
        # PENDING_COLS columns missing from older rows (e.g. when new
        # columns are added in code after rows were already written).
        if still_open:
            still_open_df = pd.DataFrame(still_open)
            for c in self.PENDING_COLS:
                if c not in still_open_df.columns:
                    still_open_df[c] = ""
            still_open_df = still_open_df[self.PENDING_COLS]
            still_open_df.to_csv(self.pending_path, index=False)
        else:
            self.pending_path.unlink(missing_ok=True)

        # SHADOW A/B: settle the active challenger's shadow pending rows using
        # the SAME resolution index, so the weekly review can score challenger
        # calibration vs champion. No-op if no challenger / no shadow pending.
        self._settle_shadow(res_by_market, ts)

        self._refresh_markdown()

        if self.verbose:
            wins = sum(1 for r in newly_settled if r["bet_won"])
            net = sum(r["net_pnl"] for r in newly_settled)
            logger.info(
                f"settle_pending(): {len(newly_settled)} newly settled "
                f"({wins} wins, net_pnl={net:+.3f})  still_open={len(still_open)}"
            )
        return {
            "settled":    len(newly_settled),
            "still_open": len(still_open),
        }

    def _resolve_rows(self, pending: pd.DataFrame, res_by_market: pd.Series,
                      ts: str) -> Tuple[list, list]:
        """
        Split a pending DataFrame into (newly_settled, still_open) using the
        Kalshi resolution index. Shared by the champion settle and the shadow
        settle so they use IDENTICAL resolution / PnL logic. Pure on its inputs
        — does not read or write any file.
        """
        newly_settled = []
        still_open = []
        for _, row in pending.iterrows():
            market_id = row["chosen_market_id"]
            if market_id not in res_by_market.index:
                still_open.append(row)
                continue
            resolution = res_by_market.loc[market_id]
            if pd.isna(resolution):
                still_open.append(row)
                continue

            direction = row["chosen_direction"]
            cost = float(row["entry_price"])
            # YES bet wins iff resolution == 1; NO bet wins iff resolution == 0
            yes_won = float(resolution) >= 0.5
            bet_won = (direction == "YES" and yes_won) or (direction == "NO" and not yes_won)
            gross_pnl = (1.0 - cost) if bet_won else (-cost)
            fee = float(row["fee"]) if pd.notna(row["fee"]) else _kalshi_fee(cost)
            net_pnl = gross_pnl - fee

            settled_row = row.to_dict()
            settled_row.update({
                "timestamp_settled": ts,
                "resolution":        float(resolution),
                "bet_won":           bool(bet_won),
                "gross_pnl":         gross_pnl,
                "net_pnl":           net_pnl,
            })
            newly_settled.append(settled_row)
        return newly_settled, still_open

    def _settle_shadow(self, res_by_market: pd.Series, ts: str) -> None:
        """
        Settle the active challenger's shadow pending rows against the same
        Kalshi resolution index the champion used, moving resolved rows from
        `<cid>.csv` to `<cid>_settled.csv`. Complete no-op when no challenger
        is active or no shadow pending file exists. Wrapped so a shadow fault
        can never break the live settle.
        """
        if (not self.challenger_id
                or self.shadow_pending_path is None
                or not self.shadow_pending_path.exists()):
            return
        try:
            pending = pd.read_csv(self.shadow_pending_path)
            if pending.empty:
                return
            newly_settled, still_open = self._resolve_rows(pending, res_by_market, ts)
            if newly_settled:
                self._append_csv(self.shadow_settled_path, newly_settled,
                                 self.SHADOW_SETTLED_COLS)
            if still_open:
                still_open_df = pd.DataFrame(still_open)
                for c in self.SHADOW_PENDING_COLS:
                    if c not in still_open_df.columns:
                        still_open_df[c] = ""
                still_open_df = still_open_df[self.SHADOW_PENDING_COLS]
                still_open_df.to_csv(self.shadow_pending_path, index=False)
            else:
                self.shadow_pending_path.unlink(missing_ok=True)
            if self.verbose and newly_settled:
                wins = sum(1 for r in newly_settled if r["bet_won"])
                net = sum(r["net_pnl"] for r in newly_settled)
                logger.info(
                    f"  shadow[{self.challenger_id}] settle: "
                    f"{len(newly_settled)} settled ({wins} wins, "
                    f"net_pnl={net:+.3f}) still_open={len(still_open)}"
                )
        except Exception as e:
            logger.warning(f"shadow settle for {self.challenger_id!r} failed: {e}")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _drop_group(self, group: list, ts: str, reason: str, detail: str) -> dict:
        """Build a dropped-rows payload for every market in a mirror group."""
        rows = []
        for m in group:
            ticker = m.get("_ticker") or m.get("market_id", "").replace("kalshi::", "")
            rows.append({
                "timestamp":   ts,
                "market_id":   m["market_id"],
                "kalshi_series": _series_from_market_id(m.get("market_id")),
                "kalshi_url":  _kalshi_url(ticker),
                "player_a":    m.get("player_a"),
                "player_a_id": m.get("player_a_id"),
                "player_b":    m.get("player_b"),
                "player_b_id": m.get("player_b_id"),
                "tournament":  m.get("tournament"),
                "event_date":  m.get("event_date"),
                "yes_ask":     m.get("yes_ask"),
                "yes_bid":     m.get("yes_bid"),
                "reason":      reason,
                "reason_detail": detail,
            })
        return {"status": "drop", "dropped": rows}

    def _load_existing_match_keys(self) -> set:
        """
        Read pending.csv (if present) and return the set of match_keys for
        already-recorded bets — so scan() doesn't re-bet the same match
        when the mirror appears, or re-bet on subsequent re-scans.
        """
        return self._load_match_keys(self.pending_path)

    @staticmethod
    def _load_match_keys(path: Optional[Path]) -> set:
        """Return the set of `match_key` values in a pending CSV, or empty set
        if the file is absent/unreadable. Used for both the champion and the
        shadow pending logs to enforce one-open-bet-per-match dedup."""
        if path is None or not Path(path).exists():
            return set()
        try:
            df = pd.read_csv(path)
            return set(df["match_key"].dropna().astype(str).tolist())
        except Exception as e:
            logger.warning(f"failed to read {Path(path).name}: {e}")
            return set()

    def _latest_rank(self, player_id, player_name: str, cutoff) -> float:
        """Most recent ATP rank for the player strictly before cutoff."""
        cutoff = pd.Timestamp(cutoff)
        md = pd.to_datetime(self.tml_df["match_date"])
        if pd.notna(player_id) and "winner_id" in self.tml_df.columns:
            mask = (
                ((self.tml_df["winner_id"] == player_id)
                 | (self.tml_df["loser_id"] == player_id))
                & (md < cutoff)
            )
            side_winner = self.tml_df["winner_id"] == player_id
        elif isinstance(player_name, str) and player_name:
            mask = (
                ((self.tml_df["player_a"] == player_name)
                 | (self.tml_df["player_b"] == player_name))
                & (md < cutoff)
            )
            side_winner = self.tml_df["player_a"] == player_name
        else:
            return np.nan
        hits = self.tml_df.loc[mask].assign(_md=md[mask]).sort_values(
            "_md", ascending=False)
        if hits.empty:
            return np.nan
        last = hits.iloc[0]
        return float(last["winner_rank"]) if side_winner.loc[last.name] else float(last["loser_rank"])

    def _player_coverage(self, player_id, cutoff) -> int:
        """
        Count matches for `player_id` in the 52w window immediately before
        cutoff (the match date). Strict < cutoff — never counts the match
        we're scoring.

        Returns 0 if player_id is missing OR if the resolver mapped to a
        TML id with no recent matches (the cov_b=0 failure mode that
        drives the worst counterfactual drops).
        """
        if pd.isna(player_id):
            return 0
        cutoff = pd.Timestamp(cutoff)
        win_start = cutoff - pd.Timedelta(days=365)
        md = pd.to_datetime(self.tml_df["match_date"], errors="coerce")
        mask = (
            ((self.tml_df["winner_id"] == player_id)
             | (self.tml_df["loser_id"] == player_id))
            & (md < cutoff)
            & (md >= win_start)
        )
        return int(mask.sum())

    def _tournament_history_years(self, tournament: str) -> int:
        """
        Count how many distinct calendar years this tournament appears in TML.

        Used by the Challenger-tier guard in _score_match_from_features:
        a Challenger event with only 0–2 years of TML history is likely a
        one-off / regional event where our features (built from match
        history) are thin and the market may price local knowledge we
        can't see. The "cursed 4" tournaments from the n=175 diagnostic
        had thin coverage and accounted for ~all the loss.
        """
        if not isinstance(tournament, str) or not tournament:
            return 0
        target = _normalize_tournament(tournament)
        if not target:
            return 0
        norm_tml = self.tml_df["tournament"].astype(str).map(_normalize_tournament)
        matched = self.tml_df.loc[norm_tml == target, "match_date"]
        if matched.empty:
            return 0
        years = pd.to_datetime(matched, errors="coerce").dt.year.dropna().unique()
        return int(len(years))

    def _infer_surface_and_level(
        self,
        tournament: str,
        tier: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Look up surface + tourney_level from prior TML rows for this tournament.

        When `tier` is provided (a TML-style level code such as "C" for
        Challenger, sourced from `kalshi_series` via `TIER_FROM_SERIES`), it
        is treated as the canonical answer for `tourney_level` — the returned
        level is forced to `tier`, regardless of what a TML mode-lookup would
        say. The TML pool is still filtered to `tier`-matching rows before the
        surface lookup so that name-collision tournaments (e.g. Cordoba 250 vs
        Cordoba Challenger) return the Challenger surface and not the main-
        tour one. When `tier` is None, level is inferred from the TML mode
        as before.
        """
        if not isinstance(tournament, str) or not tournament:
            return None, None
        target = _normalize_tournament(tournament)
        if not target:
            return None, None
        pool = self.tml_df
        if tier is not None:
            pool = pool[pool["tourney_level"] == tier]
            if pool.empty:
                return None, None
        norm_tml = pool["tournament"].astype(str).map(_normalize_tournament)
        rows = pool[norm_tml == target]
        if rows.empty:
            return None, None
        surface = rows["surface"].mode()
        surface_out = surface.iloc[0] if len(surface) else None
        if tier is not None:
            return surface_out, tier
        level = rows["tourney_level"].mode()
        return surface_out, (level.iloc[0] if len(level) else None)

    @staticmethod
    def _append_csv(path: Path, rows: Iterable[dict], columns: list) -> None:
        new_df = pd.DataFrame(list(rows))
        for c in columns:
            if c not in new_df.columns:
                new_df[c] = np.nan
        new_df = new_df[columns]

        if not path.exists():
            new_df.to_csv(path, index=False)
            return

        # File exists. If its header is a subset of our `columns`, we can
        # append rows directly. Otherwise the on-disk schema is older —
        # migrate the file in place by re-reading, adding missing columns
        # (filled with empty), and writing back with the unified schema
        # before appending.
        existing_header = pd.read_csv(path, nrows=0).columns.tolist()
        if existing_header == columns:
            new_df.to_csv(path, mode="a", header=False, index=False)
            return

        existing_df = pd.read_csv(path)
        for c in columns:
            if c not in existing_df.columns:
                existing_df[c] = ""
        existing_df = existing_df[columns]
        merged = pd.concat([existing_df, new_df], ignore_index=True)
        merged.to_csv(path, index=False)

    # ------------------------------------------------------------------ #
    # Markdown rendering — friendlier eyeballing of the CSVs.
    # Re-generated from the source CSV on every scan/settle so it always
    # reflects current state; never appended to.
    # ------------------------------------------------------------------ #

    def _refresh_markdown(self) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._write_pending_md(ts)
        self._write_dropped_md(ts)
        self._write_settled_md(ts)

    def _write_pending_md(self, ts: str) -> None:
        path = self.log_dir / "pending.md"
        if not self.pending_path.exists():
            path.write_text(f"# Pending paper bets\n\n_No open bets._  \n_(generated {ts})_\n")
            return
        df = pd.read_csv(self.pending_path)
        if df.empty:
            path.write_text(f"# Pending paper bets\n\n_No open bets._  \n_(generated {ts})_\n")
            return
        df = df.sort_values("edge", ascending=False)
        lines = [
            f"# Pending paper bets ({len(df)})",
            "",
            f"_Generated {ts}_",
            "",
            "| Match | Tournament | Date | Bet | Cost | Theo | Edge | Fee | Market |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in df.iterrows():
            chosen = r.get("chosen_player_name") or "?"
            match = f"**{chosen}** ({r['chosen_direction']}) vs {r['player_b'] if chosen == r['player_a'] else r['player_a']}"
            lines.append(
                f"| {match} | {r['tournament']} | {r['event_date']} "
                f"| {r['chosen_direction']} {chosen} "
                f"| {float(r['entry_price']):.2f} "
                f"| {float(r['theo_chosen']):.3f} "
                f"| {float(r['edge']):+.3f} "
                f"| {float(r['fee']):.4f} "
                f"| [link]({r['kalshi_url']}) |"
            )
        path.write_text("\n".join(lines) + "\n")

    def _write_dropped_md(self, ts: str) -> None:
        path = self.log_dir / "dropped.md"
        if not self.dropped_path.exists():
            path.write_text(f"# Dropped markets\n\n_No drops yet._  \n_(generated {ts})_\n")
            return
        df = pd.read_csv(self.dropped_path)
        if df.empty:
            path.write_text(f"# Dropped markets\n\n_No drops yet._  \n_(generated {ts})_\n")
            return
        lines = [
            f"# Dropped markets ({len(df)})",
            "",
            f"_Generated {ts}_",
            "",
            "## Summary by reason",
            "",
            "| Reason | Count |",
            "|---|---|",
        ]
        for reason, count in df["reason"].value_counts().items():
            lines.append(f"| `{reason}` | {count} |")
        lines += [
            "",
            "## Detail",
            "",
            "| Match | Tournament | Date | Ask / Bid | Reason | Detail | Market |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in df.sort_values(["reason", "tournament"]).iterrows():
            ask = r.get("yes_ask"); bid = r.get("yes_bid")
            ab = f"{float(ask):.2f} / {float(bid):.2f}" if pd.notna(ask) and pd.notna(bid) else "—"
            match = f"{r['player_a']} vs {r['player_b']}"
            lines.append(
                f"| {match} | {r['tournament']} | {r['event_date']} "
                f"| {ab} | `{r['reason']}` | {r['reason_detail']} "
                f"| [link]({r['kalshi_url']}) |"
            )
        path.write_text("\n".join(lines) + "\n")

    def _write_settled_md(self, ts: str) -> None:
        path = self.log_dir / "settled.md"
        if not self.settled_path.exists():
            path.write_text(f"# Settled paper bets\n\n_None yet._  \n_(generated {ts})_\n")
            return
        df = pd.read_csv(self.settled_path)
        if df.empty:
            path.write_text(f"# Settled paper bets\n\n_None yet._  \n_(generated {ts})_\n")
            return
        df = df.sort_values("timestamp_settled", ascending=False)
        n_bets = len(df)
        wins   = int(df["bet_won"].sum()) if "bet_won" in df.columns else 0
        net    = float(df["net_pnl"].sum()) if "net_pnl" in df.columns else 0.0
        lines = [
            f"# Settled paper bets ({n_bets})",
            "",
            f"_Generated {ts}_",
            "",
            f"**Wins:** {wins} / {n_bets}  ({(wins/n_bets if n_bets else 0):.1%})  ",
            f"**Net PnL (per contract):** {net:+.3f}",
            "",
            "| Match | Tournament | Date | Bet | Cost | Theo | Won? | Net PnL | Market |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in df.iterrows():
            chosen = r.get("chosen_player_name") or "?"
            won = "✓" if bool(r.get("bet_won")) else "✗"
            lines.append(
                f"| {chosen} ({r['chosen_direction']}) vs "
                f"{r['player_b'] if chosen == r['player_a'] else r['player_a']} "
                f"| {r['tournament']} | {r['event_date']} "
                f"| {r['chosen_direction']} {chosen} "
                f"| {float(r['entry_price']):.2f} "
                f"| {float(r['theo_chosen']):.3f} "
                f"| {won} "
                f"| {float(r['net_pnl']):+.3f} "
                f"| [link]({r['kalshi_url']}) |"
            )
        path.write_text("\n".join(lines) + "\n")


# ============================================================================
# CLI: run scan() then settle_pending()
# ============================================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run a paper-trading scan + settle pass")
    p.add_argument("--scan-only",   action="store_true")
    p.add_argument("--settle-only", action="store_true")
    args = p.parse_args()

    pt = PaperTrader()
    if not args.settle_only:
        pt.scan()
    if not args.scan_only:
        pt.settle_pending()
