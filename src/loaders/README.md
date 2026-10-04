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

### live_match_state_loader.py
Pulls live, in-play match STATE (not prices) from the Live Tennis API free
tier (livetennisapi.com — a commercial live-tennis data feed, disclosed).

**Key pieces:**
- `LiveMatchStateLoader.load()` / `.normalize()` — two-step fetch + tidy, same
  shape as the Kalshi/Polymarket loaders. `normalize()` is pure/offline and
  returns a documented `STANDARD_SCHEMA` DataFrame (score, server, winner,
  retirement/walkover flags).
- `break_point_state()` — derives the three-valued break-point flag
  (`true`/`false`/`undefined`) from the point score and server.

**Why it's here:** this stays a pre-match, hold-to-resolution project. The
loader is settlement-adjacent MONITORING of positions we already hold — an
independent view of whether a held match went `completed` / `Retired` /
`Walkover` before Kalshi finalizes it. It is read-only data ingestion; it never
places or sizes an order, and it never enters the feature/training path (so it
carries no lookahead exposure — see the module docstring).

**Auth/limits:** `Authorization: Bearer $LIVETENNIS_API_KEY`, free tier
30 req/min & 100 req/day. The last raw payload is snapshotted to
`data/raw/live_tennis/` for audit only (live state is never replayed).

---

## Adding a New Loader

1. One file per data source
2. Must cache raw data to `data/raw/<source>/`
3. Must return a pandas DataFrame with a documented schema
4. Must accept and enforce `cutoff_date` if returning time-series data
5. Must have a runnable `if __name__ == "__main__"` smoke test
