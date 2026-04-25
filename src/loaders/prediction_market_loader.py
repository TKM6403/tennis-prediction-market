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
