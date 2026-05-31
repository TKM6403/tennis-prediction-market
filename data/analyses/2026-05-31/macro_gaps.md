# Macro-gap briefing — 2026-05-31

Surfacing systemic, repo-wide patterns for the user. Not a bet-rule proposal.
Drop counts from `dropped.csv` (n=4509), label checks from `settled.csv` (n=240).

## PnL opportunity gaps

Drop reasons, sorted by count:

| reason | last 7d (≥05-24, n=2070) | lifetime (n=4509) |
|---|---|---|
| tournament_not_in_tml | 597 | 1225 (27%) |
| low_player_coverage | 526 | 684 (15%) |
| missing_player_id | 504 | 884 (20%) |
| thin_tournament_history | 152 | 160 (4%) |
| duplicate_match | 129 | 1070 (24%) |
| below_min_edge | 87 | 299 (7%) |
| wide_spread | 47 | 121 |
| loose_mirror_sum | 18 | 36 |

The top three data-availability reasons (`tournament_not_in_tml`,
`missing_player_id`, `low_player_coverage`) together account for **~62% of all
lifetime drops** and **~78% of last-week drops** — i.e. we forgo roughly four of
every five scanned markets purely because we can't resolve the tournament or a
player in TML, before any edge/spread logic even runs. These are coverage gaps,
not risk decisions.

Growing week-over-week (prior 7d 05-17→05-24 vs last 7d): `low_player_coverage`
**144 → 526 (3.6×)**, `missing_player_id` **201 → 504 (2.5×)**, and
`thin_tournament_history` **0 → 152**. The coverage gates are firing far harder
this week — consistent with the live ingest hitting a new batch of low-tier /
overseas Challenger events (see below).

## Data-source gaps

Kalshi-listed events with **no TML resolution at all** (`tournament_not_in_tml`):
the bulk is a cluster of lower-tier / non-European Challengers —
**Bengaluru 3 (222), Kosice (174), Cervia (169), Centurion (158), Bengaluru 2
(102)**, plus their Qualification draws. None resolve in TML, so every market on
them is auto-dropped.

`thin_tournament_history` is now dominated by **Chisinau (130)** — a tournament
that *is* in TML but with <3 calendar years of history.

Player-resolver gaps: `missing_player_id` (884 lifetime) is one side failing to
resolve to a TML id — recurring names include **Pucinelli de Almeida, Ferreira
Silva, Amit Vales**. Separately, `low_player_coverage` has 91 rows where a side
resolved to an id but returned **cov=0** (28 distinct players), e.g.
**Ulises Blanch, Tristan Boyer, Arjun Kadhe, Andrea Arnaboldi, Szymon Kielan** —
players who clearly exist and play, suggesting a name→id mismatch or a
short-name collision rather than a genuinely inactive player.

## Label / inference inconsistencies

- **Cordoba mislabel still present.** All **21** Cordoba rows in `settled.csv`
  still carry `tourney_level == "250"` while `kalshi_series ==
  "KXATPCHALLENGERMATCH"` — the exact v2.2-era defect. v2.3/v2.4 correctly route
  the *gate* through `kalshi_series`, so this no longer leaks bets, but the
  stale `tourney_level` field is still wrong in the logs and remains a trap for
  any future filter or diagnostic slice that reads `tourney_level`. It is the
  only series-vs-level disagreement in the book (21/240).
- Surface inference: no NaN surfaces in `settled.csv` — clean.
- `cov=0` for players who clearly exist (above) is an inference inconsistency in
  the resolver, not a market-thinness fact.

## Repeating bug patterns

`src/paper_trader.py` is by far the most-churned source file — **10 of the last
~11 src commits touch it**. The tier/Challenger-labeling region specifically has
been reworked **five times in a row**: v2.2 add (`56a9a81`) → v2.3 revert
(`47eeff1`) → tier-filtered lookup fix (`6e8b48b`) → canonical-field refactor
(`683c020`) → v2.4 re-enable (`570c593`). Two of these are explicitly tagged
`fix`/`revert`. That is **a single file touched by ≥3 fix/revert-class commits in
one month**, all circling the same concept (how a Challenger's tier is
determined). The churn has stabilized the *gate*, but the underlying
`tourney_level` field that caused it remains uncorrected (see above).

## Suggested follow-ups

- (SEV-3) The user should look at why ~62% lifetime / ~78% last-week of scanned
  markets drop on TML coverage (tournament/player resolution), since this caps
  addressable volume regardless of edge quality.
- (SEV-2) The user should look at the 28 players returning `cov=0` despite
  clearly being active (Blanch, Boyer, Kadhe, Arnaboldi, Kielan…) — a
  name→TML-id resolver miss would silently suppress real, possibly-profitable
  markets.
- (SEV-2) The user should look at whether the stale `tourney_level == "250"` on
  the 21 Cordoba rows should be backfilled/corrected, since any future
  level-based slice or filter reading that field inherits the v2.2 bug.
- (SEV-3) The user should look at the week-over-week 3.6× jump in
  `low_player_coverage` and the new `thin_tournament_history`/Chisinau cluster
  to confirm it's a benign new-event batch and not a regression in the resolver.
- (SEV-3) The user should look at consolidating the tier-determination logic in
  `paper_trader.py`, which has absorbed 5 consecutive reworks of the same idea.

No SEV-1 items: nothing here is currently *placing* losing bets. The v2.4 gate
reads the canonical `kalshi_series` field, so the residual Cordoba/`tourney_level`
defect and the coverage gaps cost forgone volume and latent correctness, not
live PnL.
