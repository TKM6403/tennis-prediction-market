"""
live_match_state_loader.py

In-play match-state loader for the Live Tennis API (livetennisapi.com).

WHAT THIS IS
------------
The other loaders in this directory (tml_loader, prediction_market_loader)
supply the two inputs the Theo pipeline trains and backtests on: historical
ATP results and Kalshi market prices. This loader supplies a THIRD, different
kind of data — the live, in-play STATE of a match that is happening right now:
score, whose serve it is, whether the receiver is at break point, and whether
the match has finished / retired / walked over.

WHY WE WANT IT (the honest fit)
-------------------------------
This project entries PRE-MATCH and holds to resolution (see CLAUDE.md). It does
not do live trading, and this loader does NOT change that. Its job is
settlement-adjacent monitoring of positions we already hold:

  * paper_trader.settle_pending() waits for Kalshi to officially resolve a
    market. Kalshi tennis markets resolve on the match RESULT, and a
    retirement or walkover settles differently from a played-out match
    ("...after a ball has been played..."). A live, independent view of
    "match 90209 just went `completed` / `event_status=Retired`, winner=p1"
    lets an operator anticipate a settlement before Kalshi finalizes it, and
    lets us sanity-check Kalshi's resolution against the actual on-court result.

  * It is a read-only DATA feed. It never places, sizes, or routes an order —
    that stays entirely inside paper_trader under the existing pre-match entry
    rule. This module only ingests and normalizes state, consistent with the
    "loaders do data ingestion only, no business logic" rule for this package.

NO LOOKAHEAD EXPOSURE
---------------------
CLAUDE.md rule #1 requires every function that accesses HISTORICAL data for
features to take and enforce a `cutoff_date`. This loader is exempt by
construction: it reads the live present, and its output never enters the
feature-engineering / training path (that path is fed only by tml_loader,
which owns the cutoff guard). To keep the loader interface symmetric with the
rest of the package, `load()` still accepts an optional `cutoff_date`; when
given it simply drops matches scheduled on/after that instant, which is only
useful for replaying a captured snapshot deterministically. It is never a
substitute for tml_loader's guard and must not be relied on as one.

VENDOR DISCLOSURE
-----------------
The Live Tennis API is a commercial live-tennis DATA provider (it is not a
venue or an execution/settlement service — it only reports match state). This
loader targets its documented free keyed tier: 30 requests/minute,
100 requests/day, no card required. Get a key at
https://livetennisapi.com/subscribe/free and set LIVETENNIS_API_KEY.
The request shape here mirrors the vendor's own open-source reference client
(github.com/livetennisapi/polymarket-tennis, MIT): base
https://api.livetennisapi.com/api/public/v1, `Authorization: Bearer <key>`,
responses shaped `{"data": [...], "meta": {...}}`.

----------------------------------------------------------------------------
SCHEMA (returned by normalize())
----------------------------------------------------------------------------

match_id        int      Vendor match id (globally unique per match).
status          str      "live" | "completed" | "upcoming" | ...
event_status    str/None Terminal detail when present: "Retired", "Walkover",
                         "Completed", ... None while nothing special applies.
tournament      str      Event name, e.g. "Cincinnati Open".
tour            str      "atp" | "wta" | "challenger" | ...
surface         str/None "hard" | "clay" | "grass" | ...
round_          str/None Round string e.g. "Round of 16". Named round_ for
                         consistency with the market schema in this package.
is_doubles      bool     True for doubles rubbers.
draw            str/None "singles" | "doubles".
scheduled_time  datetime Scheduled start (UTC, tz-aware). NaT if unknown.
player1_id      int/None Vendor player id for p1.
player1_name    str      p1 display name (a "A/B" pair string for doubles).
player1_ranking int/None p1 singles ranking, None if unranked/doubles.
player2_id      int/None Vendor player id for p2.
player2_name    str      p2 display name.
player2_ranking int/None p2 singles ranking.
sets_p1         int/None Sets won by p1 so far.
sets_p2         int/None Sets won by p2 so far.
games_p1        int/None Games p1 has in the CURRENT set (last set entry).
games_p2        int/None Games p2 has in the current set.
points_p1       str/None p1 game points: "0"/"15"/"30"/"40"/"AD", or a
                         tiebreak point count as a string, or None between
                         points / at completion.
points_p2       str/None p2 game points, same encoding.
server          int/None 1 if p1 is serving, 2 if p2, None if unknown/not live.
is_tiebreak     bool     True if the current game is a tiebreak.
break_point     str      THREE-VALUED break-point flag, one of:
                           "true"      receiver is one point from breaking
                           "false"     no break point on the current point
                           "undefined" cannot be determined (tiebreak, no
                                       server, or missing point data)
                         See break_point_state() for the exact rule.
winner          int/None 1 or 2 once decided, else None.
withdrew        int/None The player who retired/withdrew (1 or 2), else None.
is_completed    bool     status == "completed".
is_retirement   bool     event_status indicates a retirement.
is_walkover     bool     event_status indicates a walkover.
score_timestamp datetime When the score snapshot was taken (UTC), NaT if none.
source          str      Always "livetennisapi".
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

BASE_URL = "https://api.livetennisapi.com/api/public/v1"
API_KEY_ENV = "LIVETENNIS_API_KEY"
FREE_KEY_URL = "https://livetennisapi.com/subscribe/free"

# Where to cache the most recent raw payload. Unlike the historical loaders,
# live state is NOT reused across runs — this snapshot exists only for audit /
# offline debugging of the last fetch. data/raw/ is gitignored (CLAUDE.md #4).
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "live_tennis"

# Three-valued break-point flag values.
BP_TRUE = "true"
BP_FALSE = "false"
BP_UNDEFINED = "undefined"

# Point tokens the receiver's opponent (the server) can hold while the
# receiver is still exactly one point from winning the game at "40".
_SERVER_BEHIND_AT_40 = {"0", "15", "30"}

STANDARD_SCHEMA = [
    "match_id", "status", "event_status", "tournament", "tour", "surface",
    "round_", "is_doubles", "draw", "scheduled_time",
    "player1_id", "player1_name", "player1_ranking",
    "player2_id", "player2_name", "player2_ranking",
    "sets_p1", "sets_p2", "games_p1", "games_p2",
    "points_p1", "points_p2", "server", "is_tiebreak", "break_point",
    "winner", "withdrew", "is_completed", "is_retirement", "is_walkover",
    "score_timestamp", "source",
]


class MissingAPIKeyError(RuntimeError):
    """Raised when load() is called with no Live Tennis API key available."""

    def __init__(self) -> None:
        super().__init__(
            f"No Live Tennis API key. Set the {API_KEY_ENV} environment "
            f"variable or pass api_key=. Free keys: {FREE_KEY_URL}"
        )


# ============================================================================
# Break-point rule — the load-bearing enrichment
# ============================================================================

def break_point_state(
    points: Optional[list],
    server: Optional[int],
    is_tiebreak: bool,
) -> str:
    """
    Compute the THREE-VALUED break-point flag for the current point.

    WHY: "Is the returner at break point?" is a state the raw feed does not
    hand you directly — you have to derive it from the point score and who is
    serving. A break point is a swing moment (the returner can win the server's
    game on the next point), so it is exactly the kind of live signal a
    position-monitor wants surfaced explicitly rather than re-derived ad hoc by
    every caller.

    RULE (standard tennis game scoring, tiebreaks excluded):
        A break point exists for the RETURNER when they are one point from
        winning the current service game, i.e.
            returner has "AD"                      (advantage returner), OR
            returner has "40" AND server has 0/15/30
        There are NO break points during a tiebreak (different scoring), and
        the state is UNDEFINED whenever we cannot know it: no server set, or
        the point tokens are missing (e.g. between points, or at completion).

    Args:
        points:      Two-element [p1_point, p2_point] list as strings
                     ("0"/"15"/"30"/"40"/"AD"), or None entries. May be None.
        server:      1 if p1 is serving, 2 if p2 is serving, else None.
        is_tiebreak: True if the current game is a tiebreak.

    Returns:
        One of "true", "false", "undefined".

    Dummy example:
        points=["40", "AD"], server=1, is_tiebreak=False
            -> server is p1; returner is p2 holding "AD" -> "true"
        points=["30", "40"], server=2, is_tiebreak=False
            -> server is p2 (has "40"); returner p1 has "30" -> "false"
        points=["6", "6"], server=1, is_tiebreak=True
            -> tiebreak -> "undefined"
    """
    if is_tiebreak:
        return BP_UNDEFINED
    if server not in (1, 2):
        return BP_UNDEFINED
    if not isinstance(points, (list, tuple)) or len(points) < 2:
        return BP_UNDEFINED

    p1_pts, p2_pts = points[0], points[1]
    if p1_pts is None or p2_pts is None:
        return BP_UNDEFINED

    if server == 1:
        server_pts, returner_pts = str(p1_pts), str(p2_pts)
    else:
        server_pts, returner_pts = str(p2_pts), str(p1_pts)

    if returner_pts == "AD":
        return BP_TRUE
    if returner_pts == "40" and server_pts in _SERVER_BEHIND_AT_40:
        return BP_TRUE
    return BP_FALSE


# ============================================================================
# Small extraction helpers
# ============================================================================

def _current_set_games(games_side: Any) -> Optional[int]:
    """
    Games in the CURRENT set for one player.

    The feed sends games per player as a per-set list, e.g. p1=[6, 3, 2] means
    6 games in set 1, 3 in set 2, 2 in the (current) set 3. The current set is
    the last entry. Returns None if the list is empty/missing (e.g. a match
    that has finished, where games is []).
    """
    if isinstance(games_side, (list, tuple)) and games_side:
        try:
            return int(games_side[-1])
        except (TypeError, ValueError):
            return None
    return None


def _as_int(value: Any) -> Optional[int]:
    """Best-effort int, else None (never raises)."""
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _event_status_is(event_status: Any, needle: str) -> bool:
    """Case-insensitive substring test on the event_status string."""
    return isinstance(event_status, str) and needle in event_status.lower()


# ============================================================================
# Loader
# ============================================================================

class LiveMatchStateLoader:
    """
    Ingests live in-play match state from the Live Tennis API free tier.

    Two-step, mirroring the other loaders in this package:
        loader = LiveMatchStateLoader()          # reads LIVETENNIS_API_KEY
        raw   = loader.load(status="live")        # list of venue-native matches
        state = loader.normalize(raw)             # tidy DataFrame (STANDARD_SCHEMA)

    normalize() calls validate() before returning, so schema bugs surface at
    the source. All network access goes through load(); normalize() is pure and
    offline, which is what the __main__ smoke test exercises with dummy data.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        cache_dir: Optional[str] = None,
        timeout: float = 15.0,
    ):
        """
        Args:
            api_key:   Live Tennis API key. Falls back to $LIVETENNIS_API_KEY.
                       Only needed for load(); normalize() never uses it.
            base_url:  API base, override for testing.
            cache_dir: Where to write the last raw payload (audit snapshot).
                       Defaults to data/raw/live_tennis/.
            timeout:   Per-request timeout in seconds.
        """
        import os
        self.api_key = api_key or os.environ.get(API_KEY_ENV) or ""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else RAW_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def load(
        self,
        status: str = "live",
        tour: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        cutoff_date=None,
    ) -> list:
        """
        Fetch raw match objects for a lifecycle status.

        Args:
            status:      "live" (default) or "upcoming"/"completed" — the free
                         tier serves live and scheduled matches.
            tour:        Optional tour filter, e.g. "atp" / "challenger".
            limit:       Page size (vendor caps apply).
            offset:      Page offset.
            cutoff_date: Optional. If given, drop matches whose scheduled_time
                         is on/after this instant. This exists only for
                         deterministic replay of a captured snapshot; it is NOT
                         the pipeline's lookahead guard (see module docstring).

        Returns:
            List of venue-native match dicts (the "data" array), one per match.
            Pass to normalize() for a tidy DataFrame.
        """
        if not self.api_key:
            raise MissingAPIKeyError()

        params: dict[str, Any] = {"status": status, "limit": limit, "offset": offset}
        if tour:
            params["tour"] = tour

        resp = requests.get(
            f"{self.base_url}/matches",
            params=params,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "accept": "application/json",
            },
            timeout=self.timeout,
        )
        if resp.status_code == 429:
            raise RuntimeError(
                "Live Tennis API rate limit hit (free tier: 30 req/min, "
                "100 req/day). Slow the polling cadence."
            )
        resp.raise_for_status()
        payload = resp.json()
        matches = payload.get("data", []) if isinstance(payload, dict) else []

        # Audit snapshot of the last fetch (not a reuse cache — live state is
        # never replayed for features).
        try:
            snap = self.cache_dir / f"matches_{status}.json"
            snap.write_text(json.dumps(payload))
        except OSError as exc:  # pragma: no cover - disk best-effort only
            logger.warning("could not write live snapshot: %s", exc)

        if cutoff_date is not None and matches:
            cutoff = pd.Timestamp(cutoff_date, tz="UTC")
            kept = []
            for m in matches:
                sched = pd.to_datetime(m.get("scheduled_time"), errors="coerce", utc=True)
                if pd.isna(sched) or sched < cutoff:
                    kept.append(m)
            matches = kept

        return matches

    def normalize(self, raw: Any) -> pd.DataFrame:
        """
        Transform raw match objects into a tidy DataFrame (STANDARD_SCHEMA).

        Accepts the list[dict] returned by load(), or a DataFrame of the same
        records. Calls validate() before returning. Never performs I/O.
        """
        records = self._to_records(raw)
        if not records:
            return pd.DataFrame(columns=STANDARD_SCHEMA)

        rows = []
        for m in records:
            players = m.get("players") or {}
            p1 = players.get("p1") or {}
            p2 = players.get("p2") or {}
            score = m.get("score") or {}

            sets = score.get("sets") or [None, None]
            games = score.get("games") or [[], []]
            points = score.get("points") or [None, None]
            server = _as_int(score.get("server"))
            is_tiebreak = bool(score.get("is_tiebreak"))
            event_status = m.get("event_status")

            rows.append({
                "match_id":        _as_int(m.get("id")),
                "status":          m.get("status"),
                "event_status":    event_status,
                "tournament":      m.get("tournament"),
                "tour":            m.get("tour"),
                "surface":         m.get("surface"),
                "round_":          m.get("round"),
                "is_doubles":      bool(m.get("is_doubles")),
                "draw":            m.get("draw"),
                "scheduled_time":  pd.to_datetime(
                                       m.get("scheduled_time"), errors="coerce", utc=True),
                "player1_id":      _as_int(p1.get("id")),
                "player1_name":    p1.get("name"),
                "player1_ranking": _as_int(p1.get("ranking")),
                "player2_id":      _as_int(p2.get("id")),
                "player2_name":    p2.get("name"),
                "player2_ranking": _as_int(p2.get("ranking")),
                "sets_p1":         _as_int(sets[0] if len(sets) > 0 else None),
                "sets_p2":         _as_int(sets[1] if len(sets) > 1 else None),
                "games_p1":        _current_set_games(games[0] if len(games) > 0 else None),
                "games_p2":        _current_set_games(games[1] if len(games) > 1 else None),
                "points_p1":       points[0] if len(points) > 0 else None,
                "points_p2":       points[1] if len(points) > 1 else None,
                "server":          server,
                "is_tiebreak":     is_tiebreak,
                "break_point":     break_point_state(points, server, is_tiebreak),
                "winner":          _as_int(m.get("winner")),
                "withdrew":        _as_int(m.get("withdrew")),
                "is_completed":    m.get("status") == "completed",
                "is_retirement":   _event_status_is(event_status, "retir"),
                "is_walkover":     _event_status_is(event_status, "walkover"),
                "score_timestamp": pd.to_datetime(
                                       score.get("timestamp"), errors="coerce", utc=True),
                "source":          "livetennisapi",
            })

        df = pd.DataFrame(rows)[STANDARD_SCHEMA]
        self.validate(df)
        return df

    def live_state(self, tour: Optional[str] = None) -> pd.DataFrame:
        """Convenience: load() live matches and normalize() in one call."""
        return self.normalize(self.load(status="live", tour=tour))

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def validate(self, df: pd.DataFrame) -> None:
        """
        Validate a normalized DataFrame against the schema's invariants.

        Checks:
            1. All STANDARD_SCHEMA columns are present.
            2. break_point is exactly one of the three allowed tokens.
            3. server / winner / withdrew are 1, 2, or NULL — nothing else.

        Raises ValueError on the first failure, naming the offending match_id.
        """
        missing = [c for c in STANDARD_SCHEMA if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Got columns: {list(df.columns)}"
            )

        if len(df) == 0:
            return

        valid_bp = {BP_TRUE, BP_FALSE, BP_UNDEFINED}
        bad_bp = df[~df["break_point"].isin(valid_bp)]
        if len(bad_bp) > 0:
            first = bad_bp.iloc[0]
            raise ValueError(
                f"break_point must be one of {sorted(valid_bp)}; got "
                f"{first['break_point']!r} on match_id={first['match_id']}"
            )

        for col in ("server", "winner", "withdrew"):
            bad = df[df[col].notna() & ~df[col].isin([1, 2])]
            if len(bad) > 0:
                first = bad.iloc[0]
                raise ValueError(
                    f"{col} must be 1, 2, or NULL; got {first[col]!r} "
                    f"on match_id={first['match_id']}"
                )

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_records(raw: Any) -> list:
        """Coerce load() output (or an equivalent DataFrame) to list[dict]."""
        if raw is None:
            return []
        if isinstance(raw, pd.DataFrame):
            return raw.to_dict(orient="records")
        if isinstance(raw, dict):
            # tolerate being handed the whole {"data": [...]} envelope
            data = raw.get("data")
            return list(data) if isinstance(data, list) else [raw]
        if isinstance(raw, (list, tuple)):
            return list(raw)
        raise TypeError(f"Unsupported raw type for normalize(): {type(raw)!r}")


# ============================================================================
# Smoke test — runnable with dummy data, no network or API key required
# ============================================================================

def _dummy_matches() -> list:
    """
    Minimal dummy payloads mirroring the vendor's documented match shape:
    a live singles match with a break point against the server, a live
    tiebreak (break point undefined), and a completed-by-retirement match.
    """
    return [
        {   # Break point: p1 serving, p2 (returner) at AD -> break_point true
            "id": 90211, "tournament": "Cincinnati Open", "tour": "atp",
            "surface": "hard", "round": "Round of 16", "status": "live",
            "event_status": None, "is_doubles": False, "draw": "singles",
            "scheduled_time": "2026-08-18T01:15:00Z",
            "players": {
                "p1": {"id": 50101, "name": "Jiri Lehecka", "ranking": 21},
                "p2": {"id": 50102, "name": "Arthur Fils", "ranking": 15},
            },
            "score": {
                "sets": [1, 1], "games": [[6, 3, 2], [4, 6, 2]],
                "points": ["40", "AD"], "server": 1, "is_tiebreak": False,
                "timestamp": "2026-08-18T02:31:20Z",
            },
            "winner": None, "withdrew": None,
        },
        {   # Tiebreak in progress -> break_point undefined regardless of points
            "id": 90213, "tournament": "Sion", "tour": "challenger",
            "surface": "clay", "round": "Round of 32", "status": "live",
            "event_status": None, "is_doubles": False, "draw": "singles",
            "scheduled_time": "2026-08-17T12:30:00Z",
            "players": {
                "p1": {"id": 50301, "name": "Dimitar Kuzmanov", "ranking": 290},
                "p2": {"id": 50302, "name": "Marvin Moeller", "ranking": 315},
            },
            "score": {
                "sets": [0, 0], "games": [[5], [6]], "points": ["6", "6"],
                "server": 1, "is_tiebreak": True,
                "timestamp": "2026-08-18T02:31:12Z",
            },
            "winner": None, "withdrew": None,
        },
        {   # Completed by retirement: p1 won, p2 withdrew, no live score
            "id": 90209, "tournament": "Cincinnati Open", "tour": "atp",
            "surface": "hard", "round": "Round of 32", "status": "completed",
            "event_status": "Retired", "is_doubles": False, "draw": "singles",
            "scheduled_time": "2026-08-17T16:00:00Z",
            "players": {
                "p1": {"id": 50601, "name": "Tommy Paul", "ranking": 12},
                "p2": {"id": 50602, "name": "Adolfo Vallejo", "ranking": 480},
            },
            "score": {
                "sets": [1, 0], "games": [], "points": [None, None],
                "server": None, "is_tiebreak": False, "timestamp": None,
            },
            "winner": 1, "withdrew": 2,
        },
    ]


def _test_break_point_state():
    print("=" * 60)
    print("break_point_state() — DUMMY DATA TESTS")
    print("=" * 60)

    cases = [
        # (points, server, is_tiebreak, expected, why)
        (["40", "AD"], 1, False, BP_TRUE,  "returner p2 has AD vs server p1"),
        (["AD", "40"], 2, False, BP_TRUE,  "returner p1 has AD vs server p2"),
        (["40", "40"], 1, False, BP_FALSE, "deuce — server not behind"),
        (["30", "40"], 1, False, BP_TRUE,  "returner p2 at 40, server p1 at 30"),
        (["40", "30"], 1, False, BP_FALSE, "server p1 leads 40-30 (game point, not break)"),
        (["AD", "40"], 1, False, BP_FALSE, "server p1 has AD — no break point"),
        (["6", "6"],   1, True,  BP_UNDEFINED, "tiebreak — never a break point"),
        (["40", "AD"], None, False, BP_UNDEFINED, "no server known"),
        ([None, None], 1, False, BP_UNDEFINED, "no point data (e.g. completed)"),
    ]
    for points, server, tb, expected, why in cases:
        got = break_point_state(points, server, tb)
        ok = "PASS" if got == expected else "FAIL"
        print(f"  [{ok}] points={points} server={server} tiebreak={tb} "
              f"-> {got!r} (expected {expected!r}; {why})")
        assert got == expected, f"{points}/{server}/{tb}: got {got}, want {expected}"
    print("  All break_point_state cases PASSED\n")


def _test_normalize():
    print("=" * 60)
    print("LiveMatchStateLoader.normalize() — DUMMY DATA TESTS")
    print("=" * 60)

    loader = LiveMatchStateLoader(api_key="dummy")  # no network is touched
    df = loader.normalize(_dummy_matches())

    print(f"  rows: {len(df)} (expected 3)")
    assert len(df) == 3

    m = df.set_index("match_id")

    print("\n  Match 90211 — live, break point against server:")
    r = m.loc[90211]
    print(f"    server={r['server']} points=({r['points_p1']},{r['points_p2']}) "
          f"break_point={r['break_point']!r} sets=({r['sets_p1']},{r['sets_p2']}) "
          f"games=({r['games_p1']},{r['games_p2']})")
    assert r["break_point"] == BP_TRUE
    assert r["server"] == 1
    assert r["sets_p1"] == 1 and r["sets_p2"] == 1
    assert r["games_p1"] == 2 and r["games_p2"] == 2   # current (3rd) set
    assert bool(r["is_completed"]) is False

    print("\n  Match 90213 — tiebreak, break point undefined:")
    r = m.loc[90213]
    print(f"    is_tiebreak={r['is_tiebreak']} break_point={r['break_point']!r}")
    assert r["is_tiebreak"] is True or r["is_tiebreak"] == True
    assert r["break_point"] == BP_UNDEFINED

    print("\n  Match 90209 — completed by retirement:")
    r = m.loc[90209]
    print(f"    status={r['status']!r} event_status={r['event_status']!r} "
          f"winner={r['winner']} withdrew={r['withdrew']} "
          f"is_completed={r['is_completed']} is_retirement={r['is_retirement']} "
          f"is_walkover={r['is_walkover']} break_point={r['break_point']!r}")
    assert bool(r["is_completed"]) is True
    assert bool(r["is_retirement"]) is True
    assert bool(r["is_walkover"]) is False
    assert r["winner"] == 1 and r["withdrew"] == 2
    assert r["break_point"] == BP_UNDEFINED           # no server / no points
    assert pd.isna(r["games_p1"])                     # games [] -> None

    print("\n  Empty input -> empty, well-formed frame:")
    empty = loader.normalize([])
    assert list(empty.columns) == STANDARD_SCHEMA and len(empty) == 0
    print("    PASS")

    print("\n  DataFrame round-trip (normalize accepts its own records):")
    again = loader.normalize(df)
    assert len(again) == 3
    print("    PASS")

    print("\n  validate() rejects a bad server value:")
    bad = df.copy()
    bad.loc[bad.index[0], "server"] = 9
    try:
        loader.validate(bad)
        print("    FAIL — no error raised")
        raise AssertionError("validate did not reject server=9")
    except ValueError as e:
        print(f"    PASS — {str(e)[:70]}")

    print("\n  All normalize() tests PASSED")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_break_point_state()
    _test_normalize()
    print("\n" + "=" * 60)
    print("All live_match_state_loader tests passed.")
    print("=" * 60)
