# data/

This directory is **never committed to git.** Both subdirectories are
gitignored. Do not work around this.

---

## raw/

Cached source files exactly as downloaded. Nothing is modified here.

```
data/raw/tml/           ATP match CSVs from stats.tennismylife.org
                        One file per year e.g. 2024.csv, 2024_challenger.csv
                        Current year re-downloads on every run (data updates daily)
                        Past years are cached indefinitely

data/raw/kalshi/        Kalshi market data pulled from their REST API
                        Stored per market series e.g. tennis_wimbledon_2024.json
```

## processed/

Cleaned, joined, and feature-engineered files ready for the ML pipeline.
These are derived from raw and can always be regenerated.

```
data/processed/matches.parquet       Cleaned TML data, all years, both tours
data/processed/features.parquet      Per-match feature vectors with labels
data/processed/market_prices.parquet Kalshi prices joined to match data
```

## Regenerating processed data

```bash
python src/loaders/tml_loader.py        # pulls raw TML CSVs
python src/loaders/kalshi_loader.py     # pulls raw Kalshi data
python src/ml/features/build.py         # builds features.parquet
```
