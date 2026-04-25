# src/loaders/

Data ingestion only. No feature logic, no model logic, no business logic.
A loader's only job is: pull data from a source, cache it, return a clean DataFrame.

---

## Files

### tml_loader.py
Pulls ATP match data from stats.tennismylife.org (TML database).

**Key functions:**
- `load_matches()` — loads one or more years of match data with optional
  filters for surface, tourney level, and a hard cutoff_date lookahead guard
- `get_player_history()` — returns all matches for a player before a cutoff,
  normalized to player perspective (serve stats, win/loss, rank)

**Caching:** raw CSVs are saved to `data/raw/tml/`. Past years cache forever,
current year re-downloads on every run since TML updates daily.

**Cutoff enforcement:** `load_matches()` accepts `cutoff_date` and hard-drops
all rows on or after that date. This is the primary lookahead guard for the
entire pipeline. Always pass a cutoff when building features.

### kalshi_loader.py _(not yet built)_
Will pull tennis market data from the Kalshi REST API.

**Planned functions:**
- `get_tennis_markets()` — fetch all resolved tennis markets
- `get_market_history()` — time series of YES prices for a specific market
- `match_to_market()` — fuzzy join between TML match and Kalshi market
  (the hard part — player name and tournament name won't match exactly)

---

## Adding a New Loader

1. One file per data source
2. Must cache raw data to `data/raw/<source>/`
3. Must return a pandas DataFrame with a documented schema
4. Must accept and enforce `cutoff_date` if returning time-series data
5. Must have a runnable `if __name__ == "__main__"` smoke test
