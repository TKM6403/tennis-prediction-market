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

## v2.1 — convergent gate (coverage + mirror-sum sanity)

**Active:** `2026-05-15T22:15:19Z` to `2026-05-20` (superseded by v2.2)
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

## v2.2 — direction-asymmetry guard (drop YES-on-Challenger) — REVERTED

**Active:** `2026-05-20` only (shipped and reverted same day in v2.3)
**Origin:** auto-review 2026-05-20 (3-agent committee). See
`data/analyses/2026-05-20/` for the diagnostic, proposal, and devil's-advocate
critique.

Adds one filter on top of v2.1:

- **`yes_on_challenger`**: drop the bet if the best-edge candidate is
  `direction == "YES"` AND `tourney_level == "C"`. NO bets on Challenger
  and any bet on ATP-250+ events are preserved.
  - Constant: `DROP_YES_ON_CHALLENGER = True`
  - Both gating fields are scan-time known (no lookahead): `direction`
    comes from the candidate table built off live Kalshi quotes, and
    `level` is resolved via `_infer_surface_and_level` from TML rows
    strictly before `event_date`.

Diagnostic that motivated this (n=197 settled, 179 v1.0 + 18 v2.1):
- **YES bets**: n=160, ROI **−29.2%**, win-rate 27.5% vs avg theo 53.3%
  (z = −7.30 against calibration).
- **NO bets**: n=37, ROI **+47.5%**, well calibrated.
- **Challenger (level=C)**: n=176, ROI **−20.6%** (z = −6.61).
- **ATP-250 (level=A)**: n=21, ROI **+32.3%** (gap −0.05, clean).
- **Intersection YES ∩ Challenger**: n=142, ROI **−36.7%**, net −$19.70.
- **Within v2.1's 18 bets**: YES-on-C is still bleeding (n=14, ROI
  −44.7%); the +1.7% v2.1 overall ROI is an artifact of 4 lucky non-YES-on-C
  bets covering the still-broken cohort. v2.1's coverage + mirror-sum
  gates do not arrest the direction-asymmetry failure mode.

Mechanism (hypothesis): the public-stats features systematically
overrate the nominal favorite on thin Challenger fields, so any
favorite-YES pick inherits the bias. NO-side picks (typically betting
against an over-priced favorite) don't.

Worst-case opportunity cost if the YES-on-C pattern reverses out of
sample: small. Counterfactual kept set (n=55) ran at +47.5% ROI; even
a 20pp reversal of the dropped slice would only zero out, not exceed,
the historical bleed it prevents.

Volume cost: high. Historically 142/197 = **72.1%** of placed bets
would have been dropped; within v2.1 specifically it's 14/18 = 77.8%.
Trades volume for sign given a negative-ROI baseline; if forward
Challenger flow stays near 80% of book, v2.2 will functionally be a
NO-only-or-ATP-250 bot. Acceptable trade — betting less is not a cost
when the bets we'd drop have negative expectation — but worth watching
in the next review.

Rollback: set `DROP_YES_ON_CHALLENGER = False` (single-constant flip).

**Why it was reverted (same-day, in v2.3):**

The filter gates on `tourney_level == "C"`. That field is **not** sourced
from the live Kalshi market — it comes from `_infer_surface_and_level`,
which takes the `mode()` of `tourney_level` from all TML rows matching the
tournament *name string*. For tournaments whose name collides with a
tour-level event of the same name, the mode picks the bigger TML cohort.

Concrete bug: the live ingest is Challenger-only (`paper_trader.py` constructs
`KalshiLoader(series_tickers=["KXATPCHALLENGERMATCH"])`, so 100% of placed
bets are by ticker `KXATPCHALLENGERMATCH-…`). But TML's *Cordoba* rows
break down as 162 ATP-250 (the Argentina clay ATP-250) vs 62 Challenger,
so the mode resolves to `"250"`. All 21 Cordoba settled bets were
*Cordoba Challenger* markets mis-labeled `tourney_level == "250"` in our
own logs.

Consequence for the diagnostic that motivated v2.2: the n=21 "ATP-250
cohort" that the agents trusted as "clean +32.3% ROI, must preserve" was
actually 21 mis-labeled Challenger bets from a single tournament on the
clay swing. There is no ATP-250 cohort in our live data. The
direction-asymmetry finding (YES n=160 ROI −29.2% vs NO n=37 ROI +47.5%)
is still real, but the gate as written would (a) leak YES bets on any
Challenger whose name collides with a tour-level event (Cordoba, and
others not yet observed) and (b) was greenlit on a partly-fictitious
"clean preserved cohort" rather than on the underlying YES-only signal.

Pulled rather than patched in-place because the audit-trail story is
cleaner: v2.2 shipped, a real bug was found within hours, v2.3 reverts
behaviour to v2.1 and lands the plumbing fix that lets a future agent
review re-examine the YES-on-Challenger pattern on correctly-labeled
data. The dropped YES bets the bot would *not* have placed on 2026-05-20
to 2026-05-21 are <1 expected lost dollar — pulling is cheap, leaving a
known-flawed filter live is not.

---

## v2.3 — Kalshi series-ticker plumbing + v2.2 rollback

**Active:** `2026-05-20` to `2026-05-31` (superseded by v2.4)
**Shipped:** same day as v2.2 (within hours), as a hand-applied bugfix
outside the auto-review cadence.

**No behavioural change to bet selection.** v2.3 reverts v2.2's filter
(`DROP_YES_ON_CHALLENGER = False`) and adds data-plumbing only:

- **New column `kalshi_series`** in `pending.csv`, `settled.csv`, and
  `dropped.csv`. Populated at scan time from the market_id prefix
  via `_series_from_market_id()`. Always reflects the actual Kalshi
  series ticker the bet was placed on (`KXATPCHALLENGERMATCH`,
  `KXATPMATCH`, etc.). No name-mode lookup, no TML coupling, no
  lookahead.
- **Backfilled** for all 197 settled, 5 pending, and 2249 dropped
  historical rows. Result: every historical bet is series
  `KXATPCHALLENGERMATCH` (as expected, since the live `KalshiLoader` is
  hardcoded to that series).
- **`tourney_level` is preserved** in the row (still useful as a feature
  signal — the TML mode catches things like surface/round shape) but is
  no longer considered authoritative for tier. The auto-review process
  doc should treat `kalshi_series` as the source of truth for any future
  tier-based filter.
- v2.2's filter code path is left in place (just behind a `False` flag)
  so re-enabling it on the correct field is a small, reviewable patch
  next cycle rather than a re-implementation from scratch.

This is the smallest diff that (a) lands the rollback the auto-review
process needs and (b) hands the next reviewer correctly-labeled data so
the YES-on-Challenger question can be re-examined cleanly. No retrain.
No threshold changes. `MIN_EDGE` / `MAX_SPREAD` untouched.

**Why this isn't an auto-review change:** the bug was found by the user
inspecting the diagnostic, not by the 3-agent committee. The committee
*can't* find this class of bug because the labels are wrong in the same
direction in both the diagnostic's input and any cross-checks it would
do. Future similar issues are best caught the same way — by reviewing
the diagnostic before assuming its slices are correct.

---

## v2.4 — direction-asymmetry guard, re-enabled on canonical series — REVERTED

**Active:** `2026-05-31` only (same-day human revert in v2.5)
**Origin:** auto-review 2026-05-31 (3-agent committee, run under an explicit
user cadence override). See `data/analyses/2026-05-31/` for the diagnostic,
proposal, and devil's-advocate critique.

Re-enables the v2.2 direction-asymmetry guard that was reverted in v2.3 — but
gates on the **canonical `kalshi_series` field**, not the TML-mode `tourney_level`
that caused the v2.2 bug:

- **`yes_on_challenger` (corrected)**: drop the bet if the best-edge candidate
  is `direction == "YES"` AND `kalshi_series == "KXATPCHALLENGERMATCH"`. NO bets
  on Challengers and any bet on a future main-tour `KXATPMATCH` series are
  preserved.
  - Constant: `DROP_YES_ON_CHALLENGER = True` (flipped from `False`).
  - Gate now derives `series = _series_from_market_id(primary["market_id"])`
    instead of reading `level`. `_series_from_market_id` reads the market_id
    prefix off the live Kalshi quote — scan-time known, no TML coupling, no
    name-mode lookup, structurally immune to the Cordoba-style mislabel that
    sank v2.2.

Diagnostic that motivated this (n=240 settled, all `KXATPCHALLENGERMATCH`):
- **YES bets**: n=200, ROI **−25.2%**, win-rate 29.0% vs avg theo 52.8%.
  Wilson-95 CI [0.232, 0.356] lies entirely below avg theo — structural
  overconfidence, not noise.
- **NO bets**: n=40, ROI **+52.6%**, win-rate 52.5% vs avg theo 51.7% — well
  calibrated. Devil's-advocate confirmed this survives dropping the top 5
  winners (+25.8% on n=35), spreads across ~14 tournaments / 4 weeks, and is
  not stake-inflated (equal-weighted median per-bet ROI +58%).
- The sign has held across **all four** weekly diagnostics (2026-05-20: YES
  −29.2% / NO +47.5%; 2026-05-27 and 2026-05-31 unchanged). Acting on the
  n=200/n=40 split, NOT the noisy n=18 current-period slice (−29.6%, CI still
  covers avg theo) which the diagnostic flags as within-noise.

Why it's safe to ship now when v2.2 was pulled: v2.3 made `kalshi_series` the
single source of truth for tier and confirmed all 240 live rows are
`KXATPCHALLENGERMATCH` — there is no real ATP-250 cohort to protect (v2.2's
"clean +32.3% on n=21" was 21 mislabeled Cordoba Challengers). The plumbing
that made v2.2 unsafe is now in place; the filter lands on the bug-free field.

Volume cost (high, and the main risk): historically ~80% of placed bets are
YES-on-Challenger, so v2.4 is functionally a NO-only bot until a main-tour
series appears in the live ingest. Acceptable — the dropped cohort has negative
expectation — but **flagged for next review**: watch whether the kept NO cohort
holds its calibration out of sample, and whether flow collapses to near-zero.

Rollback: set `DROP_YES_ON_CHALLENGER = False` (single-constant flip).
`MIN_EDGE` / `MAX_SPREAD` untouched. No retrain. No threshold changes.

---

## v2.5 — revert v2.4; treat YES overconfidence as a model problem — CURRENT

**Active:** `2026-05-31` to present
**Origin:** human decision, same day as v2.4 (not an auto-review committee
change). Flips `DROP_YES_ON_CHALLENGER` back to `False`.

**No behavioural change vs v2.3.** v2.5 re-enables YES (back-the-favorite)
betting on Challengers — i.e. it removes the v2.4 guard. Behaviour is identical
to v2.3; the only difference is the corrected `kalshi_series`-gated guard code
introduced in v2.4 is left in place but dormant behind the `False` flag (a
one-line flip to re-enable, exactly as v2.3 left v2.2's code).

**Why revert a filter that demonstrably worked:**

The v2.4 guard was statistically sound — YES-on-Challenger ran −25.2% ROI on
n=200 (Wilson-95 CI excludes its 0.528 avg theo) vs NO at +52.6% on n=40, a sign
that held across four weekly diagnostics. The devil's-advocate approved it but
explicitly flagged that it *patches a calibration artifact at the execution
layer instead of fixing the model*. On reflection that is the deciding factor:

- The root cause is a **model deficiency** — the logistic-regression Theo
  systematically overrates the nominal favorite on thin Challenger fields (where
  the public-stats features are sparse and `rank_ratio_a` dominates). The right
  fix lives in the model: better calibration, surface-Elo features that decay to
  a prior for unknown players, or a retrain — not a blanket ban on a whole bet
  direction.
- With 100% of live flow on Challengers, the guard turned the system into a
  **NO-only bot**, which (a) discards ~80% of volume and (b) hides the
  overconfidence symptom we actually want to see and fix. Masking the metric we
  use to diagnose the model is counterproductive while model work is the focus.
- Forward priority is **model quality over execution-layer filtering**. We would
  rather keep placing YES bets, keep the loss signal visible, and fix the
  calibration that causes it.

`MIN_EDGE` / `MAX_SPREAD` untouched. No retrain in this change. Rollback (if we
ever want the guard back): set `DROP_YES_ON_CHALLENGER = True`.

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
