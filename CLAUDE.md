# CLAUDE.md — AI Coding Instructions for tennis-prediction-market

This file governs how AI assistants (Claude or otherwise) should behave
when writing or modifying code in this repo. Read this before touching
anything.

---

## Project Context

We are building a Theo (theoretical fair value) model for tennis prediction
markets on Kalshi. The workflow is:

1. Load historical ATP match data (TML) and Kalshi market prices
2. Engineer per-match features from match history
3. Train a calibrated win probability model (our "Theo")
4. Backtest: compare Theo vs Kalshi market price, simulate PnL

The end goal is a systematic edge in pre-match tennis markets, starting
with Grand Slams and Masters events.

---

## Hard Rules

### 1. No Lookahead Bias — Ever

This is the most important rule in the repo. Every function that computes
features or accesses historical data MUST accept a `cutoff_date` parameter
and MUST enforce it with a strict less-than filter on `tourney_date`.

**Wrong:**
```python
def get_win_rate(player, df):
    return df[df['winner_name'] == player]['won'].mean()
```

**Right:**
```python
def get_win_rate(player, df, cutoff_date):
    return df[
        (df['winner_name'] == player) &
        (df['tourney_date'] < pd.Timestamp(cutoff_date))  # strict <
    ]['won'].mean()
```

If a function touches historical data and does not have `cutoff_date`,
it is wrong. Do not merge it. Do not work around it.

### 2. Every New Feature Must Follow the Spec + Dummy Data Rule

Every new feature function must have:

**A) A docstring spec with:**
- What the feature measures and why it might have predictive signal
- Inputs with types and descriptions
- Output with type and example value
- A concrete dummy data example showing expected input → output

**B) A runnable `if __name__ == "__main__"` block at the bottom of the
file that:**
- Constructs minimal dummy data (no real CSVs required)
- Runs the feature function on it
- Prints the result in a human-readable way
- Communicates to the human developer what the feature is doing

The purpose of this rule is understanding. Before any feature touches
real data, the developer should be able to run the file and immediately
understand what the feature computes and whether it looks correct.

**Example of a compliant feature function:**

```python
def serve_dominance(player: str, df: pd.DataFrame, cutoff_date: str,
                    surface: str) -> float:
    """
    Compute a player's serve dominance score on a given surface,
    using only matches before cutoff_date.

    WHY: On fast surfaces (grass, indoor hard), serve dominance predicts
    match outcomes better than ranking alone. Markets tend to underweight
    this because they anchor on ATP ranking.

    Formula:
        serve_dom = (1stServeIn% * 1stServeWon%) + (2ndServeIn% * 2ndServeWon%)

    Args:
        player:       Player name exactly as in TML data e.g. "Jannik Sinner"
        df:           Full match DataFrame from tml_loader.load_matches()
        cutoff_date:  No data on or after this date is used. Format: 'YYYY-MM-DD'
        surface:      One of 'Hard', 'Clay', 'Grass', 'Carpet'

    Returns:
        float: serve dominance score in [0, 1]. Higher = more dominant server.
               Returns NaN if fewer than 3 matches found (insufficient data).

    Dummy example:
        Input:  player="Test Player", surface="Grass", cutoff="2024-07-01"
                3 grass matches: svpt=60, 1stIn=36, 1stWon=28, 2ndWon=14
        Output: (36/60 * 28/36) + (24/60 * 14/24) = 0.467 + 0.233 = 0.700
    """
```

### 3. No New Dependencies Without Discussion

Do not add packages to requirements.txt without flagging it to the
human developer first. State: what the package does, why the stdlib
or existing deps can't handle it, and how big the dependency is.

### 4. Data Is Never Committed

`data/raw/` and `data/processed/` are gitignored. Never commit CSVs,
parquet files, or any raw/processed data. Never work around the gitignore.

### 5. Train/Val/Test Is a Code Convention, Not a Directory Structure

The ML split is managed inside `src/ml/train.py` via a single
`TimeSeriesSplit` or date-based split. There are no separate `train/`,
`val/`, `test/` folders. The split logic lives in one place and is
explicitly documented there.

---

## Directory Reference

```
src/loaders/    Data ingestion only. TML + Kalshi. No feature logic here.
src/ml/         Feature engineering, model training, calibration, evaluation.
src/backtest/   Market simulation. Takes Theo output + Kalshi prices → PnL.
data/raw/       Cached CSVs from TML and Kalshi. Gitignored.
data/processed/ Cleaned, joined, feature-engineered files. Gitignored.
notebooks/      EDA and one-off experiments. Not production code.
```

---

## Tone for Human Communication

When implementing a new feature, always explain to the human developer:
- What the feature measures in plain English
- What the expected output range is
- What a surprising or wrong-looking output would indicate
- What the next logical step is after this piece

This project is a learning exercise as much as a trading system.
Optimize for the developer's understanding, not just correct output.
