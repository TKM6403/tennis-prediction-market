# tennis-prediction-market

Theo model and backtesting framework for tennis prediction markets on Kalshi.

## Structure

```
data/
  raw/          # TML CSVs, Kalshi market data — never committed, gitignored
  processed/    # cleaned, joined, feature-engineered parquet files

src/
  loaders/      # data ingestion: TML + Kalshi
  features/     # feature engineering per match
  models/       # Theo model training + calibration
  backtest/     # PnL simulation against historical market prices

notebooks/      # EDA and experiment writeups
```

## Data Sources
- **TML:** ATP match results 1968–present, daily updated
- **Kalshi API:** Historical prediction market prices and resolutions

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # fill in Kalshi API key
```
