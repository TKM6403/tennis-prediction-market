"""
prediction_market_loader.py

Abstract base class for all prediction market data loaders.

Each prediction market source (Kalshi, Polymarket, PredictionHunt) has its
own subclass that implements load() and normalize(). The base class defines
the shared schema contract and validate() which every subclass calls at the
end of normalize().

Class hierarchy:
    PredictionMarketLoader          <- base class (this file)
        KalshiLoader                <- TODO
        PolymarketLoader            <- TODO
        PredictionHuntLoader        <- TODO
"""

from abc import ABC, abstractmethod
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

STANDARD_SCHEMA = {
    "market_id":       str,      # source-prefixed unique ID e.g. "kalshi::TENNIS-123"
    "question":        str,      # raw question text verbatim from source
    "event_date":      "date",   # date the market resolves
    "category":        str,      # "tennis" — loaders filter to this at load time
    "yes_price":       float,    # implied probability in [0, 1] — enforced by validate()
    "snapshot_time":   "datetime",  # when this price was observed
    "volume":          float,    # total contracts traded (NaN if unavailable)
    "resolved":        bool,     # has this market settled
    "resolution":      float,    # 1.0 = YES, 0.0 = NO, NaN = not yet resolved
    "source":          str,      # "kalshi" / "polymarket" / "predictionhunt"
}

REQUIRED_COLUMNS = list(STANDARD_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class PredictionMarketLoader(ABC):
    """
    Abstract base class for prediction market data loaders.

    All subclasses must implement:
        load()       — hit the source API or cache, return raw data
        normalize()  — clean source-specific fields, return StandardMarketDF,
                       and call self.validate() before returning

    The validate() method is concrete and shared. It enforces:
        1. All required columns are present
        2. yes_price is strictly in [0, 1] for every row
        3. resolution is 1.0, 0.0, or NaN for every row

    Any single row violating these conditions raises a ValueError immediately.
    This threshold can be relaxed to a percentage in the future.

    Dummy example (validate):
        Input:
            market_id   yes_price   resolved   resolution
            "k::001"    0.42        True        1.0         <- valid
            "k::002"    1.30        False       NaN         <- bad yes_price

        Output:
            ValueError: "yes_price out of [0,1] range on 1 row(s).
                         First bad row: market_id=k::002, yes_price=1.3"
    """

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Pull raw data from the source and return it uncleaned.
        Subclass is responsible for caching to data/raw/.
        """
        pass

    @abstractmethod
    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Clean source-specific fields and return a DataFrame matching
        STANDARD_SCHEMA. Must call self.validate() before returning.
        """
        pass

    def validate(self, df: pd.DataFrame) -> None:
        """
        Validate a DataFrame against STANDARD_SCHEMA.

        Checks:
            1. All required columns present
            2. yes_price in [0, 1] on every row
            3. resolution is 1.0, 0.0, or NaN on every row

        Raises:
            ValueError immediately on the first condition that fails,
            with a plain-English message identifying the bad data.

        Args:
            df: DataFrame to validate, should match STANDARD_SCHEMA

        Returns:
            None — raises if invalid, silent if clean
        """

        # --- Check 1: required columns ---
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Got columns: {list(df.columns)}"
            )

        # --- Check 2: yes_price in [0, 1] ---
        bad_price = df[
            df["yes_price"].notna() &
            ((df["yes_price"] < 0) | (df["yes_price"] > 1))
        ]
        if len(bad_price) > 0:
            first = bad_price.iloc[0]
            raise ValueError(
                f"yes_price out of [0, 1] range on {len(bad_price)} row(s).\n"
                f"First bad row: market_id={first['market_id']}, "
                f"yes_price={first['yes_price']}"
            )

        # --- Check 3: resolution is 1.0, 0.0, or NaN ---
        valid_resolutions = {0.0, 1.0, float("nan")}
        bad_resolution = df[
            df["resolution"].notna() &
            ~df["resolution"].isin([0.0, 1.0])
        ]
        if len(bad_resolution) > 0:
            first = bad_resolution.iloc[0]
            raise ValueError(
                f"resolution must be 1.0, 0.0, or NaN. "
                f"Found invalid value on {len(bad_resolution)} row(s).\n"
                f"First bad row: market_id={first['market_id']}, "
                f"resolution={first['resolution']}"
            )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback
    import json

    print("=" * 60)
    print("BASE CLASS VALIDATE() — DUMMY DATA TESTS")
    print("=" * 60)

    # Helper to build a minimal valid row
    def make_row(market_id="test::001", yes_price=0.42, resolved=True,
                 resolution=1.0):
        return {
            "market_id":     market_id,
            "question":      "Will Player A win?",
            "event_date":    pd.Timestamp("2024-07-07").date(),
            "category":      "tennis",
            "yes_price":     yes_price,
            "snapshot_time": pd.Timestamp("2024-07-06 12:00:00"),
            "volume":        500.0,
            "resolved":      resolved,
            "resolution":    resolution,
            "source":        "test",
        }

    # Concrete subclass stub so we can instantiate the abstract base
    class _TestLoader(PredictionMarketLoader):
        def load(self): pass
        def normalize(self, raw): pass

    loader = _TestLoader()

    # ------------------------------------------------------------------
    print("\nTest 1: Clean DataFrame — should pass silently")
    print("  Input:  2 valid rows, yes_price=[0.42, 0.61], resolution=[1.0, 0.0]")
    df_clean = pd.DataFrame([
        make_row("test::001", yes_price=0.42, resolved=True,  resolution=1.0),
        make_row("test::002", yes_price=0.61, resolved=False, resolution=float("nan")),
    ])
    try:
        loader.validate(df_clean)
        print("  Output: PASSED — no error raised ✓")
    except ValueError as e:
        print(f"  Output: FAILED unexpectedly — {e}")

    # ------------------------------------------------------------------
    print("\nTest 2: yes_price > 1 — should raise ValueError")
    print("  Input:  row with yes_price=1.30")
    df_bad_price = pd.DataFrame([
        make_row("test::001", yes_price=0.42),
        make_row("test::002", yes_price=1.30),   # bad
    ])
    try:
        loader.validate(df_bad_price)
        print("  Output: FAILED — should have raised but didn't")
    except ValueError as e:
        print(f"  Output: PASSED — raised correctly ✓\n  Message: {e}")

    # ------------------------------------------------------------------
    print("\nTest 3: yes_price < 0 — should raise ValueError")
    print("  Input:  row with yes_price=-0.05")
    df_negative = pd.DataFrame([
        make_row("test::001", yes_price=-0.05),  # bad
    ])
    try:
        loader.validate(df_negative)
        print("  Output: FAILED — should have raised but didn't")
    except ValueError as e:
        print(f"  Output: PASSED — raised correctly ✓\n  Message: {e}")

    # ------------------------------------------------------------------
    print("\nTest 4: Missing required column — should raise ValueError")
    print("  Input:  DataFrame missing 'volume' and 'source' columns")
    df_missing_cols = df_clean.drop(columns=["volume", "source"])
    try:
        loader.validate(df_missing_cols)
        print("  Output: FAILED — should have raised but didn't")
    except ValueError as e:
        print(f"  Output: PASSED — raised correctly ✓\n  Message: {e}")

    # ------------------------------------------------------------------
    print("\nTest 5: Bad resolution value — should raise ValueError")
    print("  Input:  row with resolution=0.5 (only 0.0, 1.0, NaN are valid)")
    df_bad_res = pd.DataFrame([
        make_row("test::001", resolution=0.5),   # bad
    ])
    try:
        loader.validate(df_bad_res)
        print("  Output: FAILED — should have raised but didn't")
    except ValueError as e:
        print(f"  Output: PASSED — raised correctly ✓\n  Message: {e}")

    # ------------------------------------------------------------------
    print("\nTest 6: Unresolved market (resolution=NaN) — should pass")
    print("  Input:  row with resolved=False, resolution=NaN")
    df_unresolved = pd.DataFrame([
        make_row("test::001", resolved=False, resolution=float("nan")),
    ])
    try:
        loader.validate(df_unresolved)
        print("  Output: PASSED — NaN resolution correctly allowed ✓")
    except ValueError as e:
        print(f"  Output: FAILED unexpectedly — {e}")

    print("\n" + "=" * 60)
    print("All base class tests complete.")
    print("=" * 60)


# ===========================================================================
# KalshiLoader
# ===========================================================================

class KalshiLoader(PredictionMarketLoader):
    """
    Loads tennis prediction market data from the Kalshi REST API.

    Each Kalshi tennis match creates TWO markets — one per player.
    e.g. Cerundolo vs Darderi produces:
        KXATPMATCH-26APR25CERDAR-CER  (YES = Cerundolo wins)
        KXATPMATCH-26APR25CERDAR-DAR  (YES = Darderi wins)

    Both sides of each match are returned (one row per player).
    e.g. Cerundolo vs Darderi returns 2 rows — use event_ticker to group
    by match when needed downstream.

    Prices come from the API in dollars already (0.62 = 62 cents = 62%).
    No division needed — just validate they're in [0, 1].

    Relevant series tickers for tennis:
        KXATPMATCH          ATP match level (most granular, most volume)
        KXWTAMATCH          WTA match level
        KXATPGRANDSLAM      ATP Grand Slam outrights
        KXWTAGRANDSLAM      WTA Grand Slam outrights
        KXATPGAME           ATP match (older series, similar to KXATPMATCH)
        KXWMENSINGLES       Wimbledon men's singles
        KXWWOMENSINGLES     Wimbledon women's singles
        KXFOMENSINGLE       French Open men's singles
        KXFOWOMENSINGLE     French Open women's singles
        KXUSOMENSINGLE      US Open men's singles
        KXUSOWOMENSINGLE    US Open women's singles
        KXIWMEN / KXIWO     Indian Wells
        KXATPMIA / KXATPIT  Miami / Italian Open

    Dummy example (normalize):
        Raw Kalshi market:
            ticker:               "KXATPMATCH-26APR25CERDAR-DAR"
            title:                "Will Luciano Darderi win the Cerundolo vs Darderi: Round Of 64 match?"
            last_price_dollars:   0.99
            result:               "yes"
            status:               "finalized"
            open_time:            "2026-04-24T09:07:00Z"
            settlement_ts:        "2026-04-25T12:33:49Z"
            volume_fp:            353648.73

        Normalized output row:
            market_id:      "kalshi::KXATPMATCH-26APR25CERDAR-DAR"
            question:       "Will Luciano Darderi win the Cerundolo vs Darderi: Round Of 64 match?"
            event_date:     date(2026, 4, 25)
            category:       "tennis"
            yes_price:      0.99
            snapshot_time:  Timestamp("2026-04-25 12:33:49")
            volume:         353648.73
            resolved:       True
            resolution:     1.0
            source:         "kalshi"
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    # Series tickers we consider tennis
    DEFAULT_TENNIS_SERIES = [
        "KXATPMATCH",
        "KXWTAMATCH",
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

    def __init__(
        self,
        series_tickers: list = None,
        cache_dir: str = None,
    ):
        """
        Args:
            series_tickers:  List of Kalshi series tickers to pull.
                             Defaults to DEFAULT_TENNIS_SERIES.
            cache_dir:       Where to cache raw JSON. Defaults to
                             data/raw/kalshi/ relative to repo root.
        """
        self.series_tickers = series_tickers or self.DEFAULT_TENNIS_SERIES

        if cache_dir is None:
            repo_root = Path(__file__).resolve().parents[2]
            self.cache_dir = repo_root / "data" / "raw" / "kalshi"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, limit: int = None, status: str = None) -> pd.DataFrame:
        """
        Pull markets from Kalshi for all configured series tickers.

        Args:
            limit:   Max markets to fetch per series ticker. None = fetch all
                     (paginates automatically). Use a small number like 20
                     during development to avoid hammering the API.
            status:  Filter by market status. Options: "open", "closed",
                     "settled", "finalized". None = all statuses.

        Returns:
            Raw DataFrame with one row per Kalshi market (not yet normalized).
            Columns are the raw Kalshi API field names.

        What "limit" means in practice:
            Kalshi paginates at 100 rows per request. If limit=None we keep
            fetching until the cursor is empty. If limit=50 we stop after
            50 rows for that series even if more exist. This is useful during
            development — set limit=20 to quickly check the data shape.
        """
        import requests

        all_rows = []

        for series in self.series_tickers:
            rows = self._fetch_series(series, limit=limit, status=status)
            all_rows.extend(rows)

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        return df

    def _fetch_series(
        self,
        series_ticker: str,
        limit: int = None,
        status: str = None,
    ) -> list:
        """
        Paginate through all markets for a single series ticker.
        Caches the raw response to disk.
        """
        import requests

        cache_key = f"{series_ticker}_{status or 'all'}"
        cache_path = self.cache_dir / f"{cache_key}.json"

        # Use cache for anything that isn't live (settled/finalized won't change)
        if cache_path.exists() and status in ("settled", "finalized"):
            with open(cache_path) as f:
                return json.load(f)

        rows = []
        cursor = None
        page_size = min(100, limit) if limit else 100
        fetched = 0

        while True:
            params = {"limit": page_size, "series_ticker": series_ticker}
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor

            resp = requests.get(
                f"{self.BASE_URL}/markets",
                params=params,
                headers={"accept": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

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

    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw Kalshi API fields to STANDARD_SCHEMA.

        Kalshi-specific cleaning:
        - Prices are already in [0,1] as dollars (0.62 = 62 cents)
        - result field: "yes" → 1.0, "no" → 0.0, "" → NaN
        - status "finalized" or "settled" → resolved=True
        - event_date extracted from occurrence_datetime or close_time
        - market_id prefixed with "kalshi::"
        - Deduplication: per match, keep YES-resolving side if resolved,
          else keep lower ticker alphabetically

        Args:
            raw:  DataFrame from load()

        Returns:
            DataFrame matching STANDARD_SCHEMA, validated.
        """
        if raw.empty:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        df = raw.copy()

        # --- market_id ---
        df["market_id"] = "kalshi::" + df["ticker"].astype(str)

        # --- question ---
        df["question"] = df["title"].astype(str)

        # --- event_date ---
        # Use occurrence_datetime if available, else close_time
        date_col = df.get("occurrence_datetime", df.get("close_time"))
        if "occurrence_datetime" in df.columns:
            df["event_date"] = pd.to_datetime(
                df["occurrence_datetime"], errors="coerce", utc=True
            ).dt.date
        else:
            df["event_date"] = pd.to_datetime(
                df["close_time"], errors="coerce", utc=True
            ).dt.date

        # --- category ---
        df["category"] = "tennis"

        # --- yes_price ---
        # last_price_dollars is already in [0,1]
        df["yes_price"] = pd.to_numeric(
            df["last_price_dollars"], errors="coerce"
        )

        # --- snapshot_time ---
        # Use settlement_ts if resolved, else updated_time
        ts_col = df["settlement_ts"] if "settlement_ts" in df.columns else df["updated_time"]
        df["snapshot_time"] = pd.to_datetime(ts_col, errors="coerce", utc=True)

        # --- volume ---
        df["volume"] = pd.to_numeric(df.get("volume_fp", float("nan")), errors="coerce")

        # --- resolved ---
        df["resolved"] = df["status"].isin(["finalized", "settled"])

        # --- resolution ---
        def map_result(r):
            if r == "yes":
                return 1.0
            elif r == "no":
                return 0.0
            else:
                return float("nan")

        df["resolution"] = df["result"].apply(map_result)

        # --- source ---
        df["source"] = "kalshi"

        # Select only standard columns
        # Both sides of each match are kept (one row per player per match).
        # The join key for grouping by match is event_ticker in the raw data.
        out = df[REQUIRED_COLUMNS].copy()

        self.validate(out)
        return out




# ===========================================================================
# KalshiLoader smoke test
# ===========================================================================

def _test_kalshi_loader():
    print("=" * 60)
    print("KalshiLoader DUMMY DATA TESTS")
    print("=" * 60)

    import pandas as pd
    import numpy as np

    loader = KalshiLoader()

    # ------------------------------------------------------------------
    print("\nTest 1: normalize() with dummy raw rows matching real API shape")
    print("  Input:  2 markets from same match (Cerundolo vs Darderi)")
    print("          Darderi side (YES, resolution=1.0) + Cerundolo side (NO, resolution=0.0)")
    print("  Expected output: 2 rows — both sides kept, no deduplication")

    dummy_raw = pd.DataFrame([
        {
            "ticker":                  "KXATPMATCH-26APR25CERDAR-DAR",
            "title":                   "Will Luciano Darderi win the Cerundolo vs Darderi: Round Of 64 match?",
            "event_ticker":            "KXATPMATCH-26APR25CERDAR",
            "series_ticker":           "KXATPMATCH",
            "last_price_dollars":      0.99,
            "previous_yes_bid_dollars": 0.61,
            "previous_yes_ask_dollars": 0.63,
            "occurrence_datetime":     "2026-04-25T12:00:00Z",
            "close_time":              "2026-04-25T12:31:43Z",
            "settlement_ts":           "2026-04-25T12:33:49Z",
            "updated_time":            "2026-04-25T12:33:49Z",
            "status":                  "finalized",
            "result":                  "yes",
            "volume_fp":               353648.73,
            "market_type":             "binary",
        },
        {
            "ticker":                  "KXATPMATCH-26APR25CERDAR-CER",
            "title":                   "Will Juan Manuel Cerundolo win the Cerundolo vs Darderi: Round Of 64 match?",
            "event_ticker":            "KXATPMATCH-26APR25CERDAR",
            "series_ticker":           "KXATPMATCH",
            "last_price_dollars":      0.01,
            "previous_yes_bid_dollars": 0.37,
            "previous_yes_ask_dollars": 0.39,
            "occurrence_datetime":     "2026-04-25T12:00:00Z",
            "close_time":              "2026-04-25T12:31:43Z",
            "settlement_ts":           "2026-04-25T12:33:49Z",
            "updated_time":            "2026-04-25T12:33:49Z",
            "status":                  "finalized",
            "result":                  "no",
            "volume_fp":               320699.72,
            "market_type":             "binary",
        },
    ])

    out = loader.normalize(dummy_raw)
    print(f"  Output rows: {len(out)}  (expected 2)")
    darderi = out[out["market_id"] == "kalshi::KXATPMATCH-26APR25CERDAR-DAR"].iloc[0]
    cerundolo = out[out["market_id"] == "kalshi::KXATPMATCH-26APR25CERDAR-CER"].iloc[0]
    print(f"  Darderi yes_price:   {darderi['yes_price']}  (expected 0.99)")
    print(f"  Darderi resolution:  {darderi['resolution']}  (expected 1.0)")
    print(f"  Cerundolo yes_price: {cerundolo['yes_price']}  (expected 0.01)")
    print(f"  Cerundolo resolution:{cerundolo['resolution']}  (expected 0.0)")
    assert len(out) == 2
    assert darderi["resolution"] == 1.0
    assert cerundolo["resolution"] == 0.0
    print("  PASSED ✓")

    # ------------------------------------------------------------------
    print("\nTest 2: normalize() with open (unresolved) market")
    print("  Input:  1 open market, result='', status='open'")
    print("  Expected: resolved=False, resolution=NaN")

    dummy_open = pd.DataFrame([{
        "ticker":             "KXATPMATCH-26MAY01SINALI-SIN",
        "title":              "Will Sinner win vs Alcaraz?",
        "event_ticker":       "KXATPMATCH-26MAY01SINALI",
        "series_ticker":      "KXATPMATCH",
        "last_price_dollars": 0.55,
        "occurrence_datetime": "2026-05-01T14:00:00Z",
        "close_time":         "2026-05-01T14:00:00Z",
        "settlement_ts":      None,
        "updated_time":       "2026-04-25T10:00:00Z",
        "status":             "open",
        "result":             "",
        "volume_fp":          12500.0,
        "market_type":        "binary",
    }])

    out2 = loader.normalize(dummy_open)
    assert out2.iloc[0]["resolved"] == False
    assert pd.isna(out2.iloc[0]["resolution"])
    assert out2.iloc[0]["yes_price"] == 0.55
    print(f"  resolved:   {out2.iloc[0]['resolved']}   (expected False)")
    print(f"  resolution: {out2.iloc[0]['resolution']}     (expected NaN)")
    print(f"  yes_price:  {out2.iloc[0]['yes_price']}   (expected 0.55)")
    print("  PASSED ✓")

    # ------------------------------------------------------------------
    print("\nTest 3: validate() catches bad price from normalize")
    print("  Input:  market with last_price_dollars=1.42 (API bug / bad data)")
    print("  Expected: ValueError from validate()")

    dummy_bad = dummy_open.copy()
    dummy_bad["last_price_dollars"] = 1.42

    try:
        loader.normalize(dummy_bad)
        print("  FAILED — should have raised ValueError")
    except ValueError as e:
        print(f"  PASSED ✓ — raised: {e}")

    print("\n" + "=" * 60)
    print("All KalshiLoader dummy tests passed.")
    print("=" * 60)
    print()
    print("--- Schema of normalized output ---")
    print(out.iloc[0].to_string())



if __name__ == "__main__":
    import json
    _test_kalshi_loader()

