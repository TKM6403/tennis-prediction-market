# Bet-rule version history

Every row in `data/paper_trades/settled.csv` and `data/paper_trades/pending.csv`
carries a `gate_version` string. This file is the source of truth for what each
version did.

Bump rules:
- **Major** (v3.0): model retrain / feature-set change / new dependency on a
  data source we didn't have before (e.g. line-movement, Elo, injury feed).
- **Minor** (v2.1, v2.2): new filter or relaxed threshold inside an existing
  gate family — same model, same features, same data sources.

Each entry below lists: cutoff timestamp (UTC), the commit, what the rule does
in plain English, and the diagnostic that motivated it.

---

## Glossary

Domain terms used throughout this file. For the ML-methodology side
(Beta calibration, calibration-vs-accuracy, log-loss model selection,
etc.) see `LITERATURE_REVIEW.md` rather than duplicating here.

- **Kalshi** — US-regulated prediction-market exchange. We trade their
  per-match ATP tennis contracts that pay $1 if a named player wins.
- **ATP Challenger vs ATP 250+** — tour-level hierarchy. ATP 250 / 500 /
  1000 / Grand Slam are the top tier ("main tour"). Challengers are the
  feeder tier below, with thinner data coverage and shallower fields.
- **TML (TennisMyLife)** — public ATP match-history CSVs (Jeff Sackmann-
  style schema). Our historical-stats data source.
- **bid / ask, yes_ask, yes_bid** — `yes_ask` is the lowest price someone
  will sell a YES contract for; `yes_bid` is the highest someone will pay
  for one. Prices are in dollars in [0, 1] and read as probabilities.
- **theo** — our model's probability estimate ("theoretical fair value")
  that a player wins. We bet YES when `theo > yes_ask + MIN_EDGE`.
- **edge** — `theo − cost`, in price units. A 5¢ edge means the model
  thinks the contract is mispriced by 5 cents on the dollar.
- **ROI / win rate** — betting-PnL terms. ROI = total profit / total
  stake. Win rate = fraction of settled bets that paid out. A break-even
  bettor at fair odds has ~50% win rate but 0% ROI after fees.
- **rank_ratio_a** — the model's primary rank-based feature: roughly
  `ATP_rank(A) / ATP_rank(B)`. When most other features are missing,
  this is what the model falls back on (often spuriously).
- **mean-imputed / imputation rate** — when a feature value is NaN we
  fill it with the training-set mean. "10 of 15 features imputed" means
  the model is essentially guessing for that match. We now gate on this.
- **52w window** — 52-week rolling lookback ending at the match date.
  Used for player coverage and surface form features.
- **mirror markets / mirror sum** — Kalshi lists both sides of a match
  as separate YES contracts (one for A, one for B). The "mirror sum"
  is `yes_ask_a + yes_ask_b`. With a tight spread it should sit just
  above 1.00; sums of 1.04+ signal thin liquidity and phantom edges.
- **open interest** — count of currently-open contracts on a market.
  A proxy for how much real money is on the line; low OI = noisy quotes.
- **candlesticks** — OHLC (open/high/low/close) price bars returned by
  Kalshi's price-series endpoint, one per time bucket. Same format as
  stock-chart candles.
- **closing-line-move** — change between the first and last candlestick
  on a match's price series. In sports betting it's a well-known proxy
  for informed flow: if smart money is on player A, the line drifts
  toward A before close.
- **Elo** — chess-style rating where each player's number updates after
  every match based on win/loss vs opponent's rating. Sackmann maintains
  per-surface Elo for tennis; it's a stronger ranking signal than raw
  ATP rank because it weighs *who* you beat, not just *that* you won.
- **ECE (expected calibration error)** — average gap between predicted
  probability and observed win rate, bucketed across the [0,1] range.
  Lower is better; 0.22 on bets (v1.0) is very poorly calibrated.
- **log-odds / log-odds shift** — `log(p / (1−p))`. Logistic regression
  is linear in log-odds, so feature contributions add up there even
  though they don't add up in probability space.
- **signed feature attribution** — per-feature contribution to a single
  prediction, with a sign indicating which player it favors. Useful for
  asking "why did the model like this bet?" post-hoc.

---

## v1.0 — original bet rule (no quality gates)

**Active:** before `2026-05-15T20:47:27Z`
**Commit:** prior to `914c669`

What it did:
- Pull all open Kalshi ATP Challenger markets.
- For each match, run model → compute `theo` (the model's win probability
  for player A; see glossary).
- Place the candidate bet with the largest `theo − cost` edge, provided
  `edge ≥ MIN_EDGE` (5¢ — i.e. the model thinks the contract is mispriced
  by at least 5 cents on the dollar).
- No filter on feature imputation, no tournament-history check, no player-
  coverage check, no mirror-sum sanity check (all defined in glossary).

Why it failed:
- n=175 settled bets → −17% ROI, 31.4% win rate, ECE on bets = 0.22
  (badly miscalibrated — well-tuned models tend to come in well under
  0.05; see `LITERATURE_REVIEW.md` on calibration).
- Losses concentrated in 4 "cursed" tournaments (Wuxi/Tunis/Oeiras 4/
  Francavilla) where TML (the public match-history feed we use; see
  glossary) coverage was thin.
- The 0.30+ edge bucket lost 63% ROI: those bets were almost all rows
  where 10+ of 15 features were mean-imputed (filled in with training-
  set means because the underlying data was missing) and `rank_ratio_a`
  alone was screaming a fake edge.
- YES bets bled (−29% ROI on n=141) while NO bets won (+37% on n=34) —
  the market knew things our public-stats features couldn't see.

---

## v2.0 — first quality gates (imputation + tournament history)

**Active:** `2026-05-15T20:47:27Z` to `2026-05-15T22:15:19Z`
**Commit:** `914c669 paper_trader: gate bets on imputation rate + tournament-history depth`

New filters (both reject the bet outright):

- **`high_imputation`**: drop if either player has more than 3 of 15 features
  NaN at scan time. Cuts off the "rank_ratio_a is the only thing the model
  knows" failure mode.
  - Constant: `MAX_IMPUTED_FEATURES = 3`
- **`thin_tournament_history`**: for Challenger events specifically, drop if
  the tournament appears in fewer than 3 calendar years of TML. Kills the
  cursed-4 tournament cluster.
  - Constant: `MIN_TOURNEY_YEARS = 3`
- ATP 250+ events are unaffected (positive ROI already).

Why these specifically:
- Diagnostic on n=175 showed both failure modes were concentrated, not
  systematic — these gates target the concentration.

---

## v2.1 — convergent gate (coverage + mirror-sum sanity) — CURRENT

**Active:** `2026-05-15T22:15:19Z` to present
**Commit:** `dc65d0d paper_trader: ship convergent gate (coverage + mirror-sum sanity)`

Adds two more filters on top of v2.0:

- **`low_player_coverage`**: require both players to have ≥15 matches in TML
  in the 52w (52-week rolling window) preceding `event_date`. Catches the
  `cov_b = 0` failure mode where Kalshi's `player_b` resolved to a TML
  player_id with no recent record (often a name-resolution miss to an
  inactive/retired player).
  - Constant: `MIN_PLAYER_COVERAGE = 15`
- **`loose_mirror_sum`**: require `|yes_ask_a + yes_ask_b − 1.0| ≤ 0.03`
  (i.e. the two sides of the same match — see "mirror markets" in
  glossary — should price to ~$1.00 together). Tight mirrors = market-
  makers know what they quote; loose mirrors (sum 1.04–1.12) signal
  phantom edges from thin liquidity.
  - Constant: `MAX_MIRROR_SUM_DEV = 0.03`

Two-agent design conversation produced this gate (each agent independently
arrived at the same three-AND filter shape: player-side data sanity AND
market-side liquidity sanity AND cross-mirror consistency).

Counterfactual on the n=176 settled bets at the time it shipped:
- Actual: n=176, −16.1% ROI.
- Would have passed: n=82, −14.1% ROI.
- Would have dropped: n=94, −17.9% ROI.

Net: ~53% drop rate, ~2pp ROI improvement on the kept set. Necessary but
clearly not sufficient — the kept set is still losing. Next experiments
(see "Planned next" below) target the remaining gap.

Open-interest leg (open contracts on the market — a proxy for real money
at stake; see glossary) is **placeholder** (`MIN_OPEN_INTEREST = 0.0`) —
not yet plumbed from `KalshiLoader.normalize`. Closing that gap is the
smallest remaining piece of the convergent gate.

---

## Planned next (not yet shipped — no version assigned)

In rough priority order. Each would bump a version when shipped.

1. **Plumb `open_interest` + `volume` from `KalshiLoader.normalize`** so the
   `MIN_OPEN_INTEREST = 500` leg of the convergent gate is real. Tiny code
   change, no retrain. Likely v2.2.
2. **Replace `rank_ratio_a` with surface-Elo** (overall + per-surface Elo
   rating — chess-style ratings that update after every match; see
   glossary — updated match-by-match Sackmann-style). Decays to a prior
   for unknown players, so won't fabricate confidence on thin Challenger
   fields. Requires retraining the model on the new feature set. Likely
   v3.0.
3. **Add closing-line-move feature** — last-candle price minus first-candle
   price on the Kalshi candlestick endpoint (OHLC bars; see glossary).
   Captures informed flow we currently can't see. The single highest-
   leverage Kalshi-side feature. Likely v3.1.
4. **Joint features**: `theo_elo_market` triangle agreement (do our
   probability, the Elo-implied probability, and the market price all
   line up?), `injury_confirmation_signal` (TML `RET` flag — opponent
   retired mid-match — AND line moved against player), `edge / spread`
   phantom-edge normalization. After Elo + line-movement are in.

See `LITERATURE_REVIEW.md` for the methodology references (Walsh & Joshi
2024 on calibration-vs-accuracy; Kull et al. 2017 on Beta calibration;
Niculescu-Mizil & Caruana 2005 on why LR is well-calibrated by default)
and for definitions of any ML-side terminology not covered in the
glossary above.
