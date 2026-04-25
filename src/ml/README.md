# src/ml/

Everything from raw match data to a trained, calibrated Theo model.
This directory owns the full ML lifecycle: features → train → evaluate.

---

## Mental Model

```
TML match data (via loaders)
        │
        ▼
  features/build.py        Computes per-match feature vectors.
                           For each match: who are the two players,
                           what is each player's history strictly
                           before this match date, compute features.
                           Output: features.parquet
        │
        ▼
    train.py               Loads features.parquet, runs time-based
                           train/val/test split, fits logistic regression,
                           wraps in CalibratedClassifierCV, saves model.
        │
        ▼
   evaluate.py             Loads held-out test set, computes:
                           - log loss
                           - Brier score
                           - AUC-ROC
                           - Calibration plot (predicted prob vs actual win rate)
                           Prints a plain-English summary of where the model
                           is well-calibrated and where it isn't.
```

---

## Train / Val / Test Split Convention

Splits are **time-based, not random.** Randomly shuffling tennis matches
would leak future information (a player's 2025 ranking is influenced by
their 2024 results). We always split on date.

```
2018 – 2022   →  train     (build model)
2023          →  val       (tune hyperparameters, check calibration)
2024 – 2025   →  test      (held out, touch only once at the end)
```

This is enforced inside `train.py`. There are no separate train/val/test
directories. The split is a date filter, not a folder.

**Why this matters:** if you accidentally train on 2024 data and test on
2023 data your model looks great but is completely useless in production.
Time-based splitting is non-negotiable.

---

## Feature Design Principles

Every feature must:
1. Be computable using only data before the match date (cutoff_date rule)
2. Have a clear hypothesis for why it predicts match outcome
3. Be documented with a dummy data example in its docstring
4. Be runnable in isolation via `if __name__ == "__main__"`

Current planned features:
```
rank_ratio                  Ranking of player A / ranking of player B
                            Baseline signal. Market already knows this.

surface_win_rate            Win rate on this surface, last 52 weeks
                            Captures surface specialists the market underweights

serve_dominance_surface     Serve quality on this specific surface
                            Key edge: market doesn't split by indoor/outdoor hard

fatigue_minutes             Total match minutes played in last 21 days
                            Market doesn't price fatigue systematically

h2h_surface_win_rate        Head-to-head win rate on this surface specifically
                            Small sample but high signal when available

indoor_outdoor_split        Win rate indoor vs outdoor, when condition matches
                            Most overlooked split in public tennis analysis
```

---

## Model Choice

**Start:** Logistic regression with L2 regularization + isotonic calibration.

Why not a fancier model:
- ~6,000 labeled examples (slam + masters, 5 years) is small
- Logistic regression forces us to validate that features have real signal
- Calibration is more important than raw accuracy for Theo purposes
- A well-calibrated simple model beats a poorly-calibrated complex one

Upgrade path: once we have evidence that features add signal beyond ranking,
consider gradient boosting (LightGBM). Not before.
