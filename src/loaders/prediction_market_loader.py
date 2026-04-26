"""
prediction_market_loader.py

Abstract base class for all prediction market data loaders.

Each prediction market source (Kalshi, Polymarket, ...) has its own
subclass that implements load() and normalize(). The base class defines
the shared schema contract and validate() which every subclass calls at
the end of normalize().

----------------------------------------------------------------------------
SCHEMA (designed for our specific use case: pre-match probability betting,
hold to resolution)
----------------------------------------------------------------------------

market_id           str       Globally unique, prefixed with source.
                              e.g. "kalshi::KXATPMATCH-26APR25CERDAR-DAR"
                                   "polymarket::501614"
question            str       Raw question text from the venue.
                              e.g. "Will Sinner win vs Alcaraz?"
player_a            str       The player whose YES contract this market
                              represents. Best-effort extracted from question.
                              May be NaN if extraction fails — that's fine,
                              we'll join on TML data downstream.
player_b            str       The opponent. Same caveat.
tournament          str       Best-effort tournament name. May be NaN.
round_              str       Round string (e.g. "R64"). May be NaN.
                              Named round_ to avoid Python builtin.
event_date          date      Match date (UTC).
entry_price         float     Pre-match YES ask price in [0, 1]. This is the
                              price you'd realistically pay to enter the
                              position before the match. For settled markets
                              this is the last quoted ask before close.
resolution          float     1.0 if YES resolved, 0.0 if NO, NaN if unsettled.
source              str       Venue identifier: "kalshi", "polymarket", etc.

Why this schema and not more:
- We're not doing live trading, so no current_bid / current_ask / volume needed
  for real-time decisions. The entry_price IS the trade price for our purposes.
- surface, fatigue, etc. come from TML in the feature pipeline. Loaders only
  produce market data.
- Player extraction is best-effort. If a venue's question format is messy,
  we leave it NaN and rely on the TML join (which has clean player IDs).

----------------------------------------------------------------------------
LOOKAHEAD GUARD
----------------------------------------------------------------------------

Per CLAUDE.md, every historical-data function accepts cutoff_date and enforces
strict event_date < cutoff. Subclass load() methods accept cutoff_date and
filter accordingly. validate() enforces nothing about cutoff — that's the
caller's responsibility — but downstream code MUST never see a row where
event_date >= cutoff_date.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import json
import re
import pandas as pd
import numpy as np
from typing import Optional, List


# ============================================================================
# Schema constants
# ============================================================================

STANDARD_SCHEMA = {
    "market_id":    "string",
    "question":     "string",
    "player_a":     "string",
    "player_b":     "string",
    "tournament":   "string",
    "round_":       "string",
    "event_date":   "date",
    # Opening prices from the first hourly candle after market open.
    # Use yes_ask when buying YES, (1 - yes_bid) when buying NO.
    # Both NaN until enrich_entry_prices() is called.
    # Do NOT use a mid price for trading decisions — spreads are 20-40¢
    # on challenger markets and mid significantly understates entry cost.
    "yes_ask":      "float",   # price to buy YES (what you pay)
    "yes_bid":      "float",   # price buyers will pay for YES (your sell price)
    "resolution":   "float",
    "source":       "string",
}

REQUIRED_COLUMNS = list(STANDARD_SCHEMA.keys())


# ============================================================================
# Base class
# ============================================================================

class PredictionMarketLoader(ABC):
    """
    Abstract base for all prediction market data loaders.

    Subclasses implement:
        load(cutoff_date, ...) -> pd.DataFrame   (raw, venue-specific)
        normalize(raw)         -> pd.DataFrame   (matches STANDARD_SCHEMA)

    Both methods are required. normalize() must call self.validate() before
    returning so we catch schema bugs at the source.
    """

    @abstractmethod
    def load(self, cutoff_date=None, **kwargs) -> pd.DataFrame:
        """
        Pull raw venue-specific market data.

        Args:
            cutoff_date: Optional date. If provided, only markets with
                         event_date strictly before this are returned.
                         REQUIRED for any backtest use.
            **kwargs:    Subclass-specific parameters (limits, status filters).

        Returns:
            DataFrame with venue-native columns. NOT yet normalized.
        """
        ...

    @abstractmethod
    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw venue data to STANDARD_SCHEMA.

        Must call self.validate() before returning.
        """
        ...

    def validate(self, df: pd.DataFrame) -> None:
        """
        Validate normalized DataFrame against STANDARD_SCHEMA.

        Checks:
            1. All REQUIRED_COLUMNS are present
            2. entry_price is in [0, 1] (or NaN for unpriced markets)
            3. resolution is exactly 1.0, 0.0, or NaN — nothing else

        Raises ValueError on the first failure with the offending row(s).
        """
        # 1. Required columns present
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Got columns: {list(df.columns)}"
            )

        if len(df) == 0:
            return

        # 2. yes_ask and yes_bid in [0, 1] or NaN
        for col in ("yes_ask", "yes_bid"):
            if col not in df.columns:
                continue
            bad = df[df[col].notna() & ((df[col] < 0) | (df[col] > 1))]
            if len(bad):
                raise ValueError(
                    f"{col} out of [0,1] on {len(bad)} row(s). "
                    f"First: market_id={bad.iloc[0]['market_id']}, {col}={bad.iloc[0][col]}"
                )

        # 3. resolution must be 1.0, 0.0, or NaN
        valid = {0.0, 1.0}
        bad_res = df[
            df["resolution"].notna()
            & ~df["resolution"].isin(valid)
        ]
        if len(bad_res) > 0:
            first = bad_res.iloc[0]
            raise ValueError(
                f"resolution must be 1.0, 0.0, or NaN. "
                f"Found invalid value on {len(bad_res)} row(s).\n"
                f"First bad row: market_id={first['market_id']}, "
                f"resolution={first['resolution']}"
            )


# ============================================================================
# Helpers shared across loaders
# ============================================================================

# Pattern: "Will [PLAYER A] win the [PLAYER B] vs [PLAYER A]: [ROUND] match?"
# (Kalshi format)
_KALSHI_TITLE_RE = re.compile(
    r"Will\s+(?P<player>.+?)\s+win\s+the\s+"
    r"(?P<left>.+?)\s+vs\s+(?P<right>.+?)"
    r"(?::\s+(?P<round>[^?]+?))?\s+match\?",
    re.IGNORECASE,
)


def _parse_kalshi_title(title: str) -> dict:
    """
    Best-effort extraction of player_a, player_b from a Kalshi match title.

    Returns dict with keys: player_a, player_b, round_.
    Any field that can't be extracted is None.

    Example:
        "Will Luciano Darderi win the Cerundolo vs Darderi: Round Of 64 match?"
        -> {
            "player_a": "Luciano Darderi",
            "player_b": "Cerundolo",        # last name only — TML join enriches
            "round_":   "Round Of 64",
        }
    """
    if not isinstance(title, str):
        return {"player_a": None, "player_b": None, "round_": None}

    m = _KALSHI_TITLE_RE.search(title)
    if not m:
        return {"player_a": None, "player_b": None, "round_": None}

    player = m.group("player").strip()
    left = m.group("left").strip()
    right = m.group("right").strip()
    round_ = m.group("round").strip() if m.group("round") else None

    # The "vs" clause uses last names; player is the full name.
    # Pick whichever side of "vs" doesn't match the player's last name.
    last = player.split()[-1].lower() if player else ""
    if last and last in left.lower():
        opponent = right
    elif last and last in right.lower():
        opponent = left
    else:
        opponent = right if player.lower() not in right.lower() else left

    return {"player_a": player, "player_b": opponent, "round_": round_}


def _extract_tournament_from_rules(rules_primary) -> Optional[str]:
    """
    Extract tournament name from Kalshi rules_primary field, normalized
    to TML format (no "ATP" / "WTA" / "Challenger" prefixes).

    Example inputs and outputs:
        "...in the 2026 ATP Madrid Round Of 64..."
            → "Madrid"
        "...in the 2026 ATP Challenger Kigali 2 Round Of 16..."
            → "Kigali 2"
        "...in the 2026 WTA Challenger Buenos Aires Quarterfinal..."
            → "Buenos Aires"

    The TML database stores tournament names without these tour/level
    prefixes, so we strip them here for consistent joining.

    Returns: tournament name (string) or None.
    """
    if not isinstance(rules_primary, str):
        return None
    m = re.search(
        r"in the \d{4}\s+(.+?)\s+(?:Round|Quarter|Semi|Final|R\d+)",
        rules_primary,
        re.IGNORECASE,
    )
    if not m:
        return None
    name = m.group(1).strip()

    # Strip leading tour/level qualifier prefixes. Keep the rest verbatim
    # because TML preserves city + suffix like "Kigali 2", "Oeiras 3".
    # Order matters: strip combined prefixes before single ones.
    PREFIXES = [
        "ATP Challenger",
        "WTA Challenger",
        "ATP",
        "WTA",
        "Challenger",
    ]
    for prefix in PREFIXES:
        if name.lower().startswith(prefix.lower() + " "):
            name = name[len(prefix):].strip()
            break

    return name or None


# ============================================================================
# KalshiLoader
# ============================================================================

# ============================================================================
# KalshiLoader
# ============================================================================

class KalshiLoader(PredictionMarketLoader):
    """
    Loads tennis prediction market data from the Kalshi REST API.

    Kalshi's tennis markets come in pairs — one YES contract per player per
    match. We return both rows; deduplication to the side our model predicts
    happens downstream.

    TYPICAL USAGE
    -------------
    For backtesting against real entry prices:

        loader = KalshiLoader(series_tickers=["KXATPCHALLENGERMATCH"])
        raw  = loader.load(status="settled")
        norm = loader.normalize(raw)          # entry_price is NaN here
        norm = loader.enrich_entry_prices(norm)  # fills entry_price via candlesticks
        # norm is now ready for MarketMatchJoiner

    For just checking market coverage (no price fetch needed):

        raw  = loader.load(status="settled")
        norm = loader.normalize(raw)          # entry_price stays NaN

    WHY TWO-STEP PRICE ENRICHMENT
    ------------------------------
    The settled-markets endpoint only carries settlement-time prices (0.01/0.99),
    not pre-match prices. Real pre-match prices come from the candlestick API.
    Separating normalize() from enrich_entry_prices() means:
      - normalize() is always fast (no extra API calls)
      - callers only pay for candlestick fetches when they need real prices
      - the cost is visible: ~0.3s per market × n_markets

    SERIES TICKERS
    --------------
    KXATPMATCH            — ATP tour match markets
    KXATPCHALLENGERMATCH  — ATP Challenger match markets (confirmed March 2026)
    Others in DEFAULT_TENNIS_SERIES — Grand Slams, WTA, etc.

    KALSHI FEE FORMULA
    ------------------
    Taker fee = 7% × p × (1 − p) per contract, charged at entry.
    Maximum fee at p=0.50: 1.75¢. At p=0.90: 0.63¢.
    PnL calculations must deduct this from every trade.
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    # Default curated list of tennis series tickers
    DEFAULT_TENNIS_SERIES = [
        "KXATPMATCH",             # ATP tour match markets
        "KXATPCHALLENGERMATCH",   # ATP Challenger match markets
        "KXWTAMATCH",             # WTA match markets
        "KXATPGRANDSLAM",
        "KXWTAGRANDSLAM",
        "KXATPGAME",
        "KXWMENSINGLES",
        "KXWWOMENSINGLES",
        "KXFOMENSINGLE",
        "KXFOWOMENSINGLE",
        "KXUSOMENSINGLE",
        "KXUSOWOMENSINGLE",
        "KXIWMEN",
        "KXIWO",
        "KXATPMIA",
        "KXATPIT",
        "KXATPMC",
        "KXATPMAD",
        "KXDDFMENSINGLES",
        "KXDDFWOMENSINGLES",
    ]

    # Series that are challenger-tier — used by callers for routing/filtering
    CHALLENGER_SERIES = {"KXATPCHALLENGERMATCH"}

    def __init__(
        self,
        series_tickers: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ):
        self.series_tickers = series_tickers or self.DEFAULT_TENNIS_SERIES

        if cache_dir is None:
            repo_root = Path(__file__).resolve().parents[2]
            self.cache_dir = repo_root / "data" / "raw" / "kalshi"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def load(
        self,
        cutoff_date=None,
        limit: Optional[int] = None,
        status: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch raw markets for all configured series tickers.

        Args:
            cutoff_date: Drop rows where event_date >= this date.
                         REQUIRED for backtest correctness.
            limit:       Max markets per series. None = all.
            status:      "settled", "finalized", "open", or None for all.

        Returns:
            Raw DataFrame with Kalshi-native columns.
            Pass to normalize() to get STANDARD_SCHEMA.
        """
        all_rows = []
        for series in self.series_tickers:
            rows = self._fetch_series(series, limit=limit, status=status)
            all_rows.extend(rows)

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        if cutoff_date is not None and "occurrence_datetime" in df.columns:
            event_dates = pd.to_datetime(
                df["occurrence_datetime"], errors="coerce", utc=True
            ).dt.date
            df = df[event_dates < cutoff_date].reset_index(drop=True)

        return df

    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw Kalshi data to STANDARD_SCHEMA.

        entry_price is NaN after this call — call enrich_entry_prices()
        separately when you need real pre-match prices.

        The returned DataFrame has extra private columns _ticker and
        _open_time (prefixed with _ to signal they are non-schema extras)
        so that enrich_entry_prices() can find them without requiring the
        caller to pass them separately.
        """
        if raw.empty:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        df  = raw.copy()
        out = pd.DataFrame(index=df.index)

        out["market_id"] = "kalshi::" + df["ticker"].astype(str)
        out["question"]  = df["title"].astype(str)

        # Player names ────────────────────────────────────────────────────
        # Title parsing gives player_a (full name) and player_b (last name
        # only from the "X vs Y" clause). We upgrade player_b to full name
        # when possible using expiration_value, which Kalshi always sets to
        # the actual winner's full name.
        parsed       = df["title"].apply(_parse_kalshi_title)
        out["player_a"] = [p["player_a"] for p in parsed]
        out["player_b"] = [p["player_b"] for p in parsed]
        out["round_"]   = [p["round_"]   for p in parsed]

        if "expiration_value" in df.columns and "result" in df.columns:
            exp_val = df["expiration_value"].astype(str)
            yes_st  = df.get("yes_sub_title",
                             pd.Series("", index=df.index)).astype(str)
            # result=='no' means YES side lost → expiration_value is the
            # opponent's full name (the actual winner)
            opp_full = exp_val.where(
                (df["result"].astype(str) == "no") & (exp_val != yes_st),
                None,
            )
            for idx in opp_full.dropna().index:
                full      = opp_full.loc[idx]
                current_b = out.loc[idx, "player_b"]
                if current_b is None or (
                    isinstance(current_b, str)
                    and isinstance(full, str)
                    and current_b.strip().lower() in full.lower()
                ):
                    out.loc[idx, "player_b"] = full

        # Tournament ──────────────────────────────────────────────────────
        out["tournament"] = (
            df["rules_primary"].apply(_extract_tournament_from_rules)
            if "rules_primary" in df.columns
            else None
        )

        # Event date — prefer occurrence_datetime, fall back to close_time
        # (occurrence_datetime exists as a column on most markets but is
        # populated only on recently-listed events; close_time is always set)
        ev_dt = pd.to_datetime(
            df.get("occurrence_datetime"), errors="coerce", utc=True
        ) if "occurrence_datetime" in df.columns else pd.Series(pd.NaT, index=df.index)
        ct = pd.to_datetime(
            df.get("close_time"), errors="coerce", utc=True
        ) if "close_time" in df.columns else pd.Series(pd.NaT, index=df.index)
        # Use close_time wherever occurrence_datetime is missing
        out["event_date"] = ev_dt.fillna(ct).dt.date

        # Entry prices — NaN until enrich_entry_prices() is called.
        # yes_ask  = what you pay to buy YES
        # yes_bid  = what buyers will pay (your sell price / cost of NO)
        out["yes_ask"] = np.nan
        out["yes_bid"] = np.nan

        # Resolution ──────────────────────────────────────────────────────
        out["resolution"] = (
            df["result"].map({"yes": 1.0, "no": 0.0}).astype(float)
        )

        out["source"] = "kalshi"

        # Private columns for enrich_entry_prices() ───────────────────────
        # Prefixed _ to signal these are non-schema implementation details.
        # enrich_entry_prices() reads these; callers should not depend on them.
        out["_ticker"]    = df["ticker"].astype(str)
        out["_open_time"] = pd.to_datetime(
            df.get("open_time"), errors="coerce", utc=True
        )
        out["_series"] = df.get("series_ticker", pd.Series(dtype=str))
        # Fall back: derive series from ticker prefix. Kalshi tickers are
        # formatted as SERIESNAME-YYMMMDDXXX, e.g.:
        #   KXATPCHALLENGERMATCH-26APR26SVRGUE-SVR
        # Split on the first segment that looks like a date (2 digits + 3 letters)
        missing_series = out["_series"].isna() | (out["_series"] == "")
        if missing_series.any():
            derived = out.loc[missing_series, "_ticker"].str.extract(
                r"^([A-Z0-9]+)-\d{2}[A-Z]{3}", expand=False
            )
            out.loc[missing_series, "_series"] = derived

        self.validate(out[REQUIRED_COLUMNS])
        return out

    def enrich_entry_prices(
        self,
        norm_df: pd.DataFrame,
        sleep_between: float = 0.3,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Populate yes_ask and yes_bid from the Kalshi candlestick API.

        Takes the close values of yes_ask and yes_bid from the first hourly
        candle after each market opens. These are the prices you would
        realistically pay or receive ~1h after the market was created.

        WHY BID AND ASK SEPARATELY (not mid)
        -------------------------------------
        On Kalshi challenger markets, spreads are typically 20-40¢. Using
        a mid price would materially misstate entry cost. The correct edge
        calculation is:
            Buy YES:  edge = theo - yes_ask     (you pay the ask)
            Buy NO:   edge = yes_bid - theo     (market pays you the bid)
        Both sides are NaN until this method is called.

        Args:
            norm_df:        DataFrame from normalize().
            sleep_between:  Seconds between API calls (~300 req/min limit).
            verbose:        Log progress every 50 markets.

        Returns:
            norm_df with yes_ask and yes_bid filled where data is available.
            Rows with no candle data stay NaN.
        """
        import time as _time

        df     = norm_df.copy()
        n      = len(df)
        filled = 0

        for i, (idx, row) in enumerate(df.iterrows()):
            series  = row.get("_series")
            ticker  = row.get("_ticker")
            open_ts = row.get("_open_time")

            if not series or not ticker or pd.isna(open_ts):
                continue

            bid, ask = self._fetch_opening_prices(series, ticker, open_ts)

            if ask is not None:
                df.at[idx, "yes_ask"] = ask
                filled += 1
            if bid is not None:
                df.at[idx, "yes_bid"] = bid

            if verbose and (i + 1) % 50 == 0:
                pct = (i + 1) / n * 100
                print(
                    f"  enrich_entry_prices: {i+1}/{n} ({pct:.0f}%)"
                    f"  ask_filled={filled}",
                    flush=True,
                )
            _time.sleep(sleep_between)

        if verbose:
            nan_ask = df["yes_ask"].isna().sum()
            nan_bid = df["yes_bid"].isna().sum()
            print(
                f"  enrich_entry_prices done: "
                f"yes_ask filled={n-nan_ask}/{n}, "
                f"yes_bid filled={n-nan_bid}/{n}",
                flush=True,
            )

        self.validate(df[REQUIRED_COLUMNS])
        return df

    # ------------------------------------------------------------------ #
    #  Private API helpers                                                 #
    # ------------------------------------------------------------------ #

    def _api_get(
        self,
        url: str,
        params: dict,
        max_retries: int = 4,
        base_wait: float = 10.0,
    ) -> Optional[dict]:
        """
        Single HTTP GET with exponential-backoff retry on 429.

        All network calls in this class go through here so rate-limit
        handling and timeouts are consistent.

        Returns parsed JSON dict, or None on non-retryable failure.
        Raises requests.HTTPError on 4xx/5xx other than 429.
        """
        import requests, time as _time

        for attempt in range(max_retries):
            r = requests.get(
                url,
                params=params,
                headers={"accept": "application/json"},
                timeout=30,
            )
            if r.status_code == 429:
                wait = base_wait * (2 ** attempt)
                if attempt < max_retries - 1:
                    _time.sleep(wait)
                    continue
                return None
            if r.status_code in (404, 503):
                return None
            r.raise_for_status()
            return r.json()

        return None

    def _fetch_series(
        self,
        series_ticker: str,
        limit: Optional[int] = None,
        status: Optional[str] = None,
    ) -> list:
        """
        Paginate through all markets for a series ticker.

        Results are cached to disk for settled/finalized queries so
        subsequent runs don't re-hit the API.
        """
        cache_key  = f"{series_ticker}_{status or 'all'}"
        cache_path = self.cache_dir / f"{cache_key}.json"

        if cache_path.exists() and status in ("settled", "finalized"):
            with open(cache_path) as f:
                return json.load(f)

        rows      = []
        cursor    = None
        page_size = min(100, limit) if limit else 100
        fetched   = 0

        while True:
            params = {"limit": page_size, "series_ticker": series_ticker}
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor

            data = self._api_get(f"{self.BASE_URL}/markets", params)
            if data is None:
                break

            batch = data.get("markets", [])
            rows.extend(batch)
            fetched += len(batch)

            cursor = data.get("cursor")
            if not cursor or not batch:
                break
            if limit and fetched >= limit:
                rows = rows[:limit]
                break

        if status in ("settled", "finalized") and rows:
            with open(cache_path, "w") as f:
                json.dump(rows, f)

        return rows

    def _fetch_candlesticks(
        self,
        series_ticker: str,
        market_ticker: str,
        start_ts: int,
        end_ts: int,
        period_minutes: int = 60,
    ) -> list:
        """
        Fetch OHLC candlestick data for a single market.

        URL: /series/{series}/markets/{ticker}/candlesticks
        Period values: 1, 60, or 1440 minutes (Kalshi constraint).

        Returns list of candlestick dicts, or [] on any failure.
        """
        url = (
            f"{self.BASE_URL}/series/{series_ticker}"
            f"/markets/{market_ticker}/candlesticks"
        )
        params = {
            "period_interval": period_minutes,
            "start_ts": start_ts,
            "end_ts":   end_ts,
        }
        data = self._api_get(url, params)
        if data is None:
            return []
        return data.get("candlesticks", [])

    @staticmethod
    def _prices_from_candle(candle: dict) -> tuple:
        """
        Extract (yes_bid, yes_ask) close prices from a candlestick dict.

        Returns a (bid, ask) tuple of floats. Either value is None if
        that side has no data (zero or missing). Callers should handle
        None values — they indicate a one-sided or empty market at that time.

        Args:
            candle: Single dict from the Kalshi candlesticks response:
                {
                    "yes_ask": {"close_dollars": "0.63", ...},
                    "yes_bid": {"close_dollars": "0.30", ...},
                    ...
                }

        Returns:
            (bid, ask) where each is a float in (0, 1) or None.

        Example:
            bid, ask = KalshiLoader._prices_from_candle(candle)
            # bid=0.30, ask=0.63
            # To buy YES: pay ask (0.63)
            # To buy NO:  pay (1 - bid) = 0.70
            # Edge for buying YES: theo - ask
            # Edge for buying NO:  bid - theo
        """
        ask_raw = (candle.get("yes_ask") or {}).get("close_dollars")
        bid_raw = (candle.get("yes_bid") or {}).get("close_dollars")

        ask = float(ask_raw) if ask_raw not in (None, "", "0", 0) else None
        bid = float(bid_raw) if bid_raw not in (None, "", "0", 0) else None

        # Sanity bounds — Kalshi prices are always in (0, 1)
        if ask is not None and not (0 < ask < 1):
            ask = None
        if bid is not None and not (0 < bid < 1):
            bid = None

        return bid, ask

    def _fetch_opening_prices(
        self,
        series_ticker: str,
        market_ticker: str,
        open_time: "pd.Timestamp",
        window_hours: int = 6,
    ) -> tuple:
        """
        Return (yes_bid, yes_ask) from the first hourly candle after open.

        Fetches the [open_time, open_time + window_hours) window and takes
        the first available candle. This is ~1h after market creation —
        a realistic pre-match quote since markets open 3-4 days early.

        Returns (None, None) if no candle data is available.
        """
        start_ts = int(open_time.timestamp())
        end_ts   = int((open_time + pd.Timedelta(hours=window_hours)).timestamp())

        candles = self._fetch_candlesticks(
            series_ticker, market_ticker, start_ts, end_ts
        )
        if not candles:
            return None, None

        return self._prices_from_candle(candles[0])


# ============================================================================
# PolymarketLoader
# ============================================================================

class PolymarketLoader(PredictionMarketLoader):
    """
    Loads tennis prediction market data from the Polymarket Gamma API.

    Polymarket structure:
        Event       — a tournament or context (e.g. "French Open Winner")
        Market      — a single YES/NO question within the event
                      (e.g. "Will Djokovic win the 2024 French Open?")

    Tennis events are tagged with tag_slug="tennis". Each event contains
    multiple markets — one per candidate player. We flatten this:
    one row per (event, market) pair.

    The Gamma API (gamma-api.polymarket.com) is fully public, no auth.

    NOTE on coverage:
        Polymarket's tennis coverage is dominated by Grand Slam outright
        markets ("who wins the tournament") rather than match-level markets.
        Match-level coverage is newer and thinner. This loader returns
        whatever is tagged tennis — caller can filter by event title or
        question text downstream.

    Dummy example (normalize):
        Raw market row (after flattening from event):
            id:                501614
            question:          "Will Novak Djokovic win the 2024 French Open Men's Singles?"
            outcomes:          '["Yes", "No"]'
            outcomePrices:     '["0", "1"]'
            endDate:           "2024-06-09T12:00:00Z"
            closed:            True
            groupItemTitle:    "Novak Djokovic"
            _event_title:      "French Open Winner"   (synthetic, copied from parent event)

        Normalized output row:
            market_id:   "polymarket::501614"
            question:    "Will Novak Djokovic win the 2024 French Open Men's Singles?"
            player_a:    "Novak Djokovic"        (from groupItemTitle)
            player_b:    None                    (outright — no specific opponent)
            tournament:  "French Open Winner"    (from event_title)
            round_:      None                    (outright — no round)
            event_date:  date(2024, 6, 9)
            entry_price: 0.0                     (YES priced at 0 — Djokovic didn't win)
            resolution:  0.0
            source:      "polymarket"
    """

    BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(
        self,
        tag_slug: str = "tennis",
        cache_dir: Optional[str] = None,
    ):
        self.tag_slug = tag_slug

        if cache_dir is None:
            repo_root = Path(__file__).resolve().parents[2]
            self.cache_dir = repo_root / "data" / "raw" / "polymarket"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        cutoff_date=None,
        limit: Optional[int] = None,
        closed: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Fetch tennis events and flatten to one row per market.

        Args:
            cutoff_date: If given, drop rows with event_date >= cutoff.
                         REQUIRED for backtest correctness.
            limit:       Max events to fetch (each event has ~5-30 markets).
                         None = fetch all (paginates).
            closed:      True for settled events only, False for active,
                         None for both.

        Returns:
            Raw DataFrame, one row per market, with Polymarket-native fields
            plus synthetic _event_* columns copied from the parent event.
        """
        events = self._fetch_events(limit=limit, closed=closed)
        if not events:
            return pd.DataFrame()

        rows = []
        for ev in events:
            ev_title = ev.get("title")
            ev_end = ev.get("endDate")
            ev_ticker = ev.get("ticker")
            for mkt in ev.get("markets", []):
                row = dict(mkt)
                row["_event_title"] = ev_title
                row["_event_endDate"] = ev_end
                row["_event_ticker"] = ev_ticker
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Lookahead guard
        if cutoff_date is not None and "endDate" in df.columns:
            event_dates = pd.to_datetime(
                df["endDate"], errors="coerce", utc=True
            ).dt.date
            df = df[event_dates < cutoff_date].reset_index(drop=True)

        return df

    def _fetch_events(
        self,
        limit: Optional[int] = None,
        closed: Optional[bool] = None,
    ) -> list:
        import requests

        cache_key = f"{self.tag_slug}_{closed if closed is not None else 'all'}"
        cache_path = self.cache_dir / f"{cache_key}.json"

        if cache_path.exists() and closed is True:
            with open(cache_path) as f:
                return json.load(f)

        rows = []
        offset = 0
        page_size = 100
        fetched = 0

        while True:
            params = {
                "tag_slug": self.tag_slug,
                "limit": page_size,
                "offset": offset,
            }
            if closed is not None:
                params["closed"] = "true" if closed else "false"

            resp = requests.get(
                f"{self.BASE_URL}/events",
                params=params,
                headers={"accept": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            if not isinstance(data, list) or not data:
                break

            rows.extend(data)
            fetched += len(data)
            offset += page_size

            if len(data) < page_size:
                break
            if limit and fetched >= limit:
                rows = rows[:limit]
                break

        if closed is True and rows:
            with open(cache_path, "w") as f:
                json.dump(rows, f)

        return rows

    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        df = raw.copy()
        out = pd.DataFrame(index=df.index)

        out["market_id"] = "polymarket::" + df["id"].astype(str)
        out["question"] = df["question"].astype(str)

        # player_a from groupItemTitle (Polymarket's per-candidate label)
        if "groupItemTitle" in df.columns:
            out["player_a"] = df["groupItemTitle"].where(
                df["groupItemTitle"].notna() & (df["groupItemTitle"] != ""),
                None,
            )
        else:
            out["player_a"] = None

        # player_b: not reliably extractable for outrights. Leave None for
        # the TML join to populate when match context is known.
        out["player_b"] = None

        if "_event_title" in df.columns:
            out["tournament"] = df["_event_title"]
        else:
            out["tournament"] = None

        out["round_"] = None

        out["event_date"] = pd.to_datetime(
            df["endDate"], errors="coerce", utc=True
        ).dt.date

        # entry_price: parse outcomePrices[0] which is the YES price.
        # Format is a JSON-string array like '["0.42", "0.58"]'.
        def yes_price(v):
            if v is None:
                return np.nan
            try:
                arr = json.loads(v) if isinstance(v, str) else v
                if isinstance(arr, list) and len(arr) >= 1:
                    return float(arr[0])
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
            return np.nan

        out["entry_price"] = df["outcomePrices"].apply(yes_price)

        # resolution: closed market with outcomePrices[0]==1 -> YES won (1.0)
        # closed with outcomePrices[0]==0 -> NO won (0.0)
        # not closed -> NaN
        def resolution(row):
            if not row.get("closed", False):
                return np.nan
            v = row.get("outcomePrices")
            try:
                arr = json.loads(v) if isinstance(v, str) else v
                if isinstance(arr, list) and len(arr) >= 1:
                    p = float(arr[0])
                    if p >= 0.99:
                        return 1.0
                    if p <= 0.01:
                        return 0.0
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
            return np.nan

        out["resolution"] = df.apply(resolution, axis=1)

        out["source"] = "polymarket"

        out = out[REQUIRED_COLUMNS].copy()
        self.validate(out)
        return out


# ============================================================================
# Smoke tests
# ============================================================================

def _test_validate():
    print("=" * 60)
    print("validate() — DUMMY DATA TESTS")
    print("=" * 60)

    class _Dummy(PredictionMarketLoader):
        def load(self, cutoff_date=None, **k):
            return pd.DataFrame()
        def normalize(self, raw):
            return raw

    v = _Dummy()

    print("\nTest 1: clean dataframe")
    good = pd.DataFrame([{
        "market_id": "x::1", "question": "Q?",
        "player_a": "A", "player_b": "B",
        "tournament": "T", "round_": "R64",
        "event_date": pd.Timestamp("2025-01-01").date(),
        "entry_price": 0.42, "resolution": 1.0, "source": "x",
    }])
    v.validate(good)
    print("  PASSED ✓")

    print("\nTest 2: entry_price > 1 raises")
    bad = good.copy()
    bad["entry_price"] = 1.5
    try:
        v.validate(bad)
        print("  FAILED")
    except ValueError as e:
        print(f"  PASSED ✓ — {str(e)[:80]}")

    print("\nTest 3: missing column raises")
    miss = good.drop(columns=["source"])
    try:
        v.validate(miss)
        print("  FAILED")
    except ValueError as e:
        print(f"  PASSED ✓ — {str(e)[:80]}")

    print("\nTest 4: resolution=0.5 raises")
    badres = good.copy()
    badres["resolution"] = 0.5
    try:
        v.validate(badres)
        print("  FAILED")
    except ValueError as e:
        print(f"  PASSED ✓ — {str(e)[:80]}")

    print("\nTest 5: NaN resolution allowed (open market)")
    open_mkt = good.copy()
    open_mkt["resolution"] = np.nan
    v.validate(open_mkt)
    print("  PASSED ✓")

    print("\nTest 6: NaN entry_price allowed (no quotes yet)")
    no_price = good.copy()
    no_price["entry_price"] = np.nan
    v.validate(no_price)
    print("  PASSED ✓")


def _test_kalshi_loader():
    print("\n" + "=" * 60)
    print("KalshiLoader — DUMMY DATA TESTS")
    print("=" * 60)

    loader = KalshiLoader()

    print("\nTest 1: settled match pair (Cerundolo vs Darderi)")
    print("  Input:    2 markets, Darderi YES wins, Cerundolo YES loses")
    print("  Expected: 2 rows, entry_price from previous_yes_ask_dollars")
    raw = pd.DataFrame([
        {
            "ticker": "KXATPMATCH-26APR25CERDAR-DAR",
            "title": "Will Luciano Darderi win the Cerundolo vs Darderi: Round Of 64 match?",
            "occurrence_datetime": "2026-04-25T12:00:00Z",
            "previous_yes_ask_dollars": 0.63,
            "yes_ask_dollars": 1.00,
            "result": "yes",
            "rules_primary": "If Luciano Darderi wins the Cerundolo vs Darderi professional tennis match in the 2026 ATP Madrid Round Of 64 after a ball has been played, then the market resolves to Yes.",
        },
        {
            "ticker": "KXATPMATCH-26APR25CERDAR-CER",
            "title": "Will Juan Manuel Cerundolo win the Cerundolo vs Darderi: Round Of 64 match?",
            "occurrence_datetime": "2026-04-25T12:00:00Z",
            "previous_yes_ask_dollars": 0.39,
            "yes_ask_dollars": 0.01,
            "result": "no",
            "rules_primary": "If Juan Manuel Cerundolo wins the Cerundolo vs Darderi professional tennis match in the 2026 ATP Madrid Round Of 64 after a ball has been played, then the market resolves to Yes.",
        },
    ])
    out = loader.normalize(raw)
    print(f"  Output rows: {len(out)} (expected 2)")
    print(f"  Row 0: player_a={out.iloc[0]['player_a']!r}, "
          f"entry_price={out.iloc[0]['entry_price']}, "
          f"resolution={out.iloc[0]['resolution']}, "
          f"tournament={out.iloc[0]['tournament']!r}, "
          f"round_={out.iloc[0]['round_']!r}")
    print(f"  Row 1: player_a={out.iloc[1]['player_a']!r}, "
          f"entry_price={out.iloc[1]['entry_price']}, "
          f"resolution={out.iloc[1]['resolution']}")
    assert len(out) == 2
    assert out.iloc[0]["entry_price"] == 0.63
    assert out.iloc[0]["resolution"] == 1.0
    assert out.iloc[1]["entry_price"] == 0.39
    assert out.iloc[1]["resolution"] == 0.0
    assert out.iloc[0]["player_a"] == "Luciano Darderi"
    assert out.iloc[0]["tournament"] == "ATP Madrid"
    assert out.iloc[0]["round_"] == "Round Of 64"
    print("  PASSED ✓")

    print("\nTest 2: open (unresolved) market")
    raw2 = pd.DataFrame([{
        "ticker": "KXATPMATCH-26MAY01SINALI-SIN",
        "title": "Will Sinner win vs Alcaraz?",
        "occurrence_datetime": "2026-05-01T14:00:00Z",
        "previous_yes_ask_dollars": 0.0,  # never traded yet
        "yes_ask_dollars": 0.55,
        "result": "",
        "rules_primary": "",
    }])
    out2 = loader.normalize(raw2)
    print(f"  resolution: {out2.iloc[0]['resolution']} (expected NaN)")
    print(f"  entry_price: {out2.iloc[0]['entry_price']} (expected 0.55, fallback to yes_ask_dollars)")
    assert pd.isna(out2.iloc[0]["resolution"])
    assert out2.iloc[0]["entry_price"] == 0.55
    print("  PASSED ✓")

    print("\nTest 3: validate() catches out-of-range price")
    raw3 = raw2.copy()
    raw3["yes_ask_dollars"] = 1.42
    try:
        loader.normalize(raw3)
        print("  FAILED")
    except ValueError as e:
        print(f"  PASSED ✓ — {str(e)[:80]}")


def _test_polymarket_loader():
    print("\n" + "=" * 60)
    print("PolymarketLoader — DUMMY DATA TESTS")
    print("=" * 60)

    loader = PolymarketLoader()

    print("\nTest 1: closed Grand Slam outright (Djokovic YES lost)")
    print("  Input:    1 market, outcomePrices=['0','1'], closed=True")
    print("  Expected: resolution=0.0, entry_price=0.0")
    raw = pd.DataFrame([{
        "id": 501614,
        "question": "Will Novak Djokovic win the 2024 French Open Men's Singles?",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0", "1"]',
        "endDate": "2024-06-09T12:00:00Z",
        "closed": True,
        "groupItemTitle": "Novak Djokovic",
        "_event_title": "French Open Winner",
        "_event_endDate": "2024-06-09T12:00:00Z",
        "_event_ticker": "french-open-mens-singles",
    }])
    out = loader.normalize(raw)
    print(f"  market_id:   {out.iloc[0]['market_id']}")
    print(f"  player_a:    {out.iloc[0]['player_a']!r}")
    print(f"  tournament:  {out.iloc[0]['tournament']!r}")
    print(f"  entry_price: {out.iloc[0]['entry_price']}")
    print(f"  resolution:  {out.iloc[0]['resolution']}")
    assert out.iloc[0]["market_id"] == "polymarket::501614"
    assert out.iloc[0]["player_a"] == "Novak Djokovic"
    assert out.iloc[0]["tournament"] == "French Open Winner"
    assert out.iloc[0]["entry_price"] == 0.0
    assert out.iloc[0]["resolution"] == 0.0
    assert out.iloc[0]["source"] == "polymarket"
    print("  PASSED ✓")

    print("\nTest 2: closed YES winner (price=['1','0'])")
    raw2 = raw.copy()
    raw2["outcomePrices"] = '["1", "0"]'
    raw2["question"] = "Will Alcaraz win the 2024 French Open?"
    raw2["groupItemTitle"] = "Carlos Alcaraz"
    out2 = loader.normalize(raw2)
    print(f"  entry_price: {out2.iloc[0]['entry_price']} (expected 1.0)")
    print(f"  resolution:  {out2.iloc[0]['resolution']} (expected 1.0)")
    assert out2.iloc[0]["entry_price"] == 1.0
    assert out2.iloc[0]["resolution"] == 1.0
    print("  PASSED ✓")

    print("\nTest 3: open market (closed=False, mid-market price)")
    raw3 = raw.copy()
    raw3["closed"] = False
    raw3["outcomePrices"] = '["0.42", "0.58"]'
    out3 = loader.normalize(raw3)
    print(f"  entry_price: {out3.iloc[0]['entry_price']} (expected 0.42)")
    print(f"  resolution:  {out3.iloc[0]['resolution']} (expected NaN)")
    assert out3.iloc[0]["entry_price"] == 0.42
    assert pd.isna(out3.iloc[0]["resolution"])
    print("  PASSED ✓")

    print("\nTest 4: malformed outcomePrices -> NaN price, NaN resolution")
    raw4 = raw.copy()
    raw4["outcomePrices"] = "not json"
    raw4["closed"] = False
    out4 = loader.normalize(raw4)
    print(f"  entry_price: {out4.iloc[0]['entry_price']} (expected NaN)")
    assert pd.isna(out4.iloc[0]["entry_price"])
    print("  PASSED ✓")

    print("\nTest 5: out-of-range price after parsing -> validation error")
    raw5 = raw.copy()
    raw5["outcomePrices"] = '["1.5", "-0.5"]'
    try:
        loader.normalize(raw5)
        print("  FAILED")
    except ValueError as e:
        print(f"  PASSED ✓ — {str(e)[:80]}")


if __name__ == "__main__":
    _test_validate()
    _test_kalshi_loader()
    _test_polymarket_loader()
    print("\n" + "=" * 60)
    print("All loader tests passed.")
    print("=" * 60)
