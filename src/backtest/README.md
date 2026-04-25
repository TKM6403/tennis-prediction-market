# src/backtest/

Takes the Theo model output and Kalshi historical market prices,
simulates trading decisions, and computes PnL.

This directory does not touch the ML model internals. It only consumes
the model's probability outputs and asks: if we had traded on these,
what would have happened?

---

## Mental Model

```
features.parquet + trained model
        │
        ▼
   theo.py              For each historical match that has a Kalshi market,
                        run the model to get P(player_A wins) = our Theo.
                        Output: match_id, player_A, theo, market_price, edge
        │
        ▼
  simulate.py           Apply betting rules:
                        - Only bet when edge > threshold (e.g. 5 cents)
                        - Size bets using Kelly criterion (fractional)
                        - Record outcome and PnL per bet
                        Output: results.parquet
        │
        ▼
  evaluate.py           Compute PnL attribution:
                        - Total PnL, PnL per bet, win rate
                        - PnL by surface, round, tourney level
                        - Edge retention: how much of modeled edge did we keep?
                        - Calibration: when we said edge=0.10, did we make ~$0.10/contract?
```

---

## Key Concepts

### Edge
```
edge = our_theo - kalshi_market_price

e.g. our_theo = 0.65, market = 0.48 → edge = 0.17
We think YES is worth 65 cents, market is selling it for 48 cents.
We buy.
```

### Kelly Criterion (Fractional)
Full Kelly bet sizing is theoretically optimal but too aggressive for
a small bankroll and noisy edge estimates. We use fractional Kelly:

```
full_kelly  = (edge * (1/market_price)) / (1/market_price - 1)
              simplified: edge / (1 - market_price)

bet_size    = bankroll * full_kelly * fraction
              where fraction = 0.25 (quarter Kelly to start)
```

Quarter Kelly means we're being conservative — we accept lower expected
returns in exchange for much lower variance. Adjust fraction upward only
after sustained evidence of real edge.

### Edge Retention
The core diagnostic for whether our model is actually good:

```
edge_retention = actual_pnl_per_contract / modeled_edge_per_contract

If edge_retention ≈ 1.0 → model is well-calibrated, we're keeping our edge
If edge_retention ≈ 0   → model edge is noise, we're not making money
If edge_retention < 0   → we're systematically wrong, adverse selection
```

---

## The Join Problem

Kalshi market names and TML player/tournament names will not match exactly.
This is the hardest engineering problem in this directory.

Example:
```
TML:    "Jannik Sinner", "Australian Open", "2025-01-26"
Kalshi: "Will Sinner win the 2025 Australian Open?"
```

The match_to_market() function in kalshi_loader.py handles this with
fuzzy string matching + date proximity. Every join must be human-verified
before being used in backtest. Bad joins silently corrupt PnL results.
