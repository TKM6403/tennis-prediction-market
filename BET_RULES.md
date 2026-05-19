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

## v1.0 — original bet rule (no quality gates)

**Active:** before `2026-05-15T20:47:27Z`
**Commit:** prior to `914c669`

What it did:
- Pull all open Kalshi ATP Challenger markets.
- For each match, run model → compute `theo`.
- Place the candidate bet with the largest `theo − cost` edge, provided
  `edge ≥ MIN_EDGE` (5¢).
- No filter on feature imputation, no tournament-history check, no player-
  coverage check, no mirror-sum sanity check.

Why it failed:
- n=175 settled bets → −17% ROI, 31.4% win rate, ECE on bets = 0.22.
- Losses concentrated in 4 "cursed" tournaments (Wuxi/Tunis/Oeiras 4/
  Francavilla) where TML coverage was thin.
- The 0.30+ edge bucket lost 63% ROI: those bets were almost all rows
  where 10+ of 15 features were mean-imputed and `rank_ratio_a` alone
  was screaming a fake edge.
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
  in the 52w preceding `event_date`. Catches the `cov_b = 0` failure mode
  where Kalshi's `player_b` resolved to a TML player_id with no recent
  record (often a name-resolution miss to an inactive/retired player).
  - Constant: `MIN_PLAYER_COVERAGE = 15`
- **`loose_mirror_sum`**: require `|yes_ask_a + yes_ask_b − 1.0| ≤ 0.03`.
  Tight mirrors = market-makers know what they quote; loose mirrors
  (sum 1.04–1.12) signal phantom edges from thin liquidity.
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

Open-interest leg is **placeholder** (`MIN_OPEN_INTEREST = 0.0`) — not yet
plumbed from `KalshiLoader.normalize`. Closing that gap is the smallest
remaining piece of the convergent gate.

---

## Planned next (not yet shipped — no version assigned)

In rough priority order. Each would bump a version when shipped.

1. **Plumb `open_interest` + `volume` from `KalshiLoader.normalize`** so the
   `MIN_OPEN_INTEREST = 500` leg of the convergent gate is real. Tiny code
   change, no retrain. Likely v2.2.
2. **Replace `rank_ratio_a` with surface-Elo** (overall + per-surface Elo,
   updated match-by-match Sackmann-style). Decays to a prior for unknown
   players, so won't fabricate confidence on thin Challenger fields. Requires
   retraining the model on the new feature set. Likely v3.0.
3. **Add closing-line-move feature** — last-candle price minus first-candle
   price on the Kalshi candlestick endpoint. Captures informed flow we
   currently can't see. The single highest-leverage Kalshi-side feature.
   Likely v3.1.
4. **Joint features**: `theo_elo_market` triangle agreement, `injury_
   confirmation_signal` (TML `RET` flag AND line moved against player),
   `edge / spread` phantom-edge normalization. After Elo + line-movement
   are in.

See `LITERATURE_REVIEW.md` for the methodology references (Walsh & Joshi
2024 on calibration-vs-accuracy; Kull et al. 2017 on Beta calibration;
Niculescu-Mizil & Caruana 2005 on why LR is well-calibrated by default).
