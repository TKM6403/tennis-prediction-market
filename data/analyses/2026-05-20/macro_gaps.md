# Macro gaps — 2026-05-20

Surfacing patterns the committee review can't fix from inside the gate. All
numbers from `data/paper_trades/dropped.csv` (n=2,309) and
`data/paper_trades/settled.csv` (n=197). Recent = `timestamp >= 2026-05-13`
(1,786 rows). Prior = 2026-05-06 to 2026-05-13 (523 rows).

## PnL opportunity gaps

Drop reasons, ranked by recent count:

| reason | recent 7d | lifetime | % of recent drops | WoW ratio |
|---|---|---|---|---|
| duplicate_match | 624 | 931 | 34.9% | 2.03x |
| tournament_not_in_tml | 518 | 570 | 29.0% | **9.96x** |
| missing_player_id | 287 | 355 | 16.1% | **4.22x** |
| low_player_coverage | 147 | 147 | 8.2% | NEW |
| below_min_edge | 144 | 205 | 8.1% | 2.36x |
| wide_spread | 36 | 57 | 2.0% | 1.71x |
| loose_mirror_sum | 16 | 16 | 0.9% | NEW |
| thin_tournament_history | 8 | 8 | 0.4% | NEW |
| yes_on_challenger | 2 | 2 | 0.1% | NEW (reverted in v2.3) |

Top-3 reasons account for **80% of all forgone markets last week**. The two
ballistic growers — `tournament_not_in_tml` (~10x WoW) and `missing_player_id`
(~4x WoW) — are 100% on `kalshi_series == KXATPCHALLENGERMATCH`. This is not
a model-side bleed, it is a data-coverage hole: Kalshi listed a wave of new
Challengers (Bengaluru 2/3, Cervia, Istanbul) that TML doesn't carry.
`low_player_coverage` (147, all NEW this week, also 100% Challenger and
concentrated in Istanbul) is the same iceberg surfacing through a different
gate.

`duplicate_match` (35% of drops) is mechanical — Kalshi lists each match
twice (one contract per side) so a 624 raw-row drop equals ~312 unique
matches; the gate is correct, but if `duplicate_match` ever leaks past the
dedup it would double our position count. Worth a tripwire.

## Data-source gaps

Tournaments live on Kalshi but absent from TML this week, by Kalshi-listing
volume:

- **Bengaluru 3** (198 drops) and Bengaluru 3 Qualification (38)
- **Cervia** (142) and Cervia Qualification (90)
- **Bengaluru 2** (50)
- **Istanbul Qualification** (40 missing-player + 70 low-coverage)
- **Oeiras 4** (17 missing-player)

All KXATPCHALLENGERMATCH. Either the TML slug map is missing these events
(see `e95cf65 audit: fix queensclub slug ...` for prior precedent — slug
drift is recurring) or TML genuinely doesn't have feed coverage for the
ITF-adjacent Challenger 50/75 tier.

Player resolver gaps — most common unresolvable Kalshi player IDs (recent):
`C0BB`, `J0D2`, `G0AO`, `P0HT`, `SY50`, `S0TI`, `S0IV`, `H0J1`, `K0AB`.
Each appears 5–11 times; these are real Challenger-tier players that the
Kalshi-to-TML name/ID join is failing on (`player_b_id=None` appears 6x —
Kalshi feed itself is occasionally null on the B-side).

## Label / inference inconsistencies

**v2.2-style bug, still latent in settled data — and now confirmed not
hypothetical.** The scanner is hardcoded to `series_tickers=
["KXATPCHALLENGERMATCH"]` (`src/paper_trader.py:393`,
`src/loaders/prediction_market_loader.py:302`), so by construction
**every market we have ever scanned, dropped, or settled is a Challenger**
— there is no ATP-250 cohort in this dataset, full stop. Confirmed in
data: 197 / 197 settled rows have `kalshi_series == KXATPCHALLENGERMATCH`.
But `tourney_level` from `_infer_surface_and_level` says C for 176 and
**"250" for 21 — all 21 are Cordoba**, where TML carries a same-name
"Cordoba" 250 historically and the inference snaps to it. This is the
same misclassification class that motivated the v2.2 revert. Crucially,
the diagnostic/proposal treated those 21 Cordoba rows as a **clean
ATP-250 cohort (+32.3% ROI)** and the critique used them as the headline
"preserved profitable" slice — that cohort does not exist. The +32.3%
is 21 mislabeled Challengers, and v2.3's own source-of-truth
(`kalshi_series`) has not been propagated back through the analysis
pipeline that produced the diagnostic.

Surface inference: 0 NaN this scan. Good. But surface is 100% Clay or
Hard with zero Grass/Carpet so we have no out-of-sample signal on the
inference path for non-Clay events.

`low_player_coverage` cases include `cov_a=22 cov_b=0` and `cov_a=43 cov_b=0`
patterns (10+ rows each) — one side has decades of history, the other
side returns 0 matches. That's a player-name join miss, not a real coverage
problem; the player exists in TML, we just aren't finding them.

## Repeating bug patterns

`src/paper_trader.py` has been touched by 7 non-auto commits in the last
month (914c669, dc65d0d, d7f680d, 3a3d831, b90f95d, 56a9a81, 47eeff1) —
two of those (56a9a81, 47eeff1) are a ship + immediate revert pair on the
same gate version, both about Challenger tier-labeling. Tier identification
specifically is now on its third source-of-truth (`tourney_level` from
joiner → `tourney_level` from `_infer_surface_and_level` → `kalshi_series`)
in three weeks. Each iteration found the previous source was wrong on some
subset.

Adjacent recurring areas: TML slug mapping (`e95cf65` slug audit,
`9c40e63` slug expansion), composite-key joining (`2c144c1` + `b9d56e3`
"merge predictions via composite key, not positional tml_match_id" — same
bug found twice). The join layer is fragile.

## Suggested follow-ups

- **(SEV-1)** The scanner only fetches `KXATPCHALLENGERMATCH`, so by
  construction there is no ATP-250 data in this dataset. The 21 "Cordoba
  ATP-250" rows are mislabeled Challengers, the diagnostic's +32.3%
  clean ATP-250 slice does not exist, and every downstream decision
  (including the v2.2 propose-then-revert cycle) was made against a
  phantom cohort. The analysis pipeline still consults `tourney_level`
  from `_infer_surface_and_level` instead of `kalshi_series` — that
  needs fixing before the next committee review.
- **(SEV-1)** ~45% of recent scanned markets (`tournament_not_in_tml` +
  `missing_player_id` + `low_player_coverage` = 952 / 1,786) are silently
  dropped on the Challenger circuit — investigate whether Bengaluru 2/3,
  Cervia, and Istanbul Qualification have TML coverage at all, or whether
  this is a slug/join failure that's leaving real opportunity on the table.
- **(SEV-1)** Tier source-of-truth is on its third rewrite in a month;
  pin down `kalshi_series == KXATPCHALLENGERMATCH ⇔ Challenger` as a
  hard invariant and add a scan-time assert that fails loudly if any
  downstream code disagrees.
- **(SEV-2)** Player-name resolver returns `cov_b=0` for players with
  obvious TML history (e.g. opposite-side `cov_a=43`); audit the
  Kalshi-ID → TML-name join for the 9 most-common unresolved IDs
  (`C0BB`, `J0D2`, `G0AO`, `P0HT`, `SY50`, `S0TI`, `S0IV`, `H0J1`, `K0AB`).
- **(SEV-2)** Add a tripwire on `duplicate_match` — if the dedup ever
  fails open, we will double-position every match.
- **(SEV-3)** Surface inference has never been exercised on Grass or
  Carpet in settled data; the v2.3 gate has no production evidence on
  ~25% of the annual calendar.
- **(SEV-3)** `src/paper_trader.py` is the highest-churn file in the
  repo; consider extracting the gate-stack into its own module so
  filter add/revert cycles stop touching the hot path.
