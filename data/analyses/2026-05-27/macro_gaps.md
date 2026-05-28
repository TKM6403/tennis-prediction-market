# Macro gaps — 2026-05-27

Surfacing systemic, repo-wide issues the committee can't fix from inside the
gate. Sources: `data/paper_trades/dropped.csv` (n=3,601), `settled.csv`
(n=227), this week's diagnostic / proposal / critique, `git log` for src/,
and last week's `2026-05-20/macro_gaps.md` for WoW comparison.

## PnL opportunity gaps

Drop reasons, this week (`timestamp > 2026-05-20`, n=1,550) and lifetime:

| reason | recent 7d | lifetime | % of recent | WoW change |
|---|---|---|---|---|
| tournament_not_in_tml | 482 | 940 | 31.1% | +19% (up from 406) |
| missing_player_id | 360 | 695 | 23.2% | +35% (up from 267) |
| low_player_coverage | 360 | 471 | 23.2% | **+224%** (up from 111) |
| duplicate_match | 143 | 1,004 | 9.2% | −74% (down from 554) |
| thin_tournament_history | 88 | 96 | 5.7% | **+1000%** (up from 8) |
| below_min_edge | 74 | 263 | 4.8% | −42% |
| wide_spread | 25 | 82 | 1.6% | −31% |
| loose_mirror_sum | 16 | 30 | 1.0% | +14% |
| yes_on_challenger | 2 | 2 | 0.1% | unchanged (v2.2 dormant) |

The top three reasons (`tournament_not_in_tml` + `missing_player_id` +
`low_player_coverage`) are **77.5% of all recent drops, ~1,202 / 1,550
scanned markets**. All three are growing WoW, and `low_player_coverage`
more than tripled. Compared to last week's macro_gaps (which already
flagged this iceberg as SEV-1), the data hole is not closing — it is
widening. `duplicate_match` is the only category that dropped, and that
is mechanical (fewer days of double-listed contracts settled into the
file this window, not a real change in dedup behaviour).

`thin_tournament_history` jumped from 8 → 88. Worth confirming whether
that is a real change in scanned events or a side effect of v2.3's
re-routing of tier checks through `kalshi_series`.

## Data-source gaps

Tournaments live on Kalshi but absent from TML this week, by drop
volume: **Kosice** (112), **Centurion** (99), **Cervia** (87),
**Bengaluru 3** (76), **Kosice Qualification** (73), **Centurion
Qualification** (35). Bengaluru 3 alone has 222 lifetime drops. These
are all `KXATPCHALLENGERMATCH` and either don't exist in our TML slug
map at all (slug drift, see commit `e95cf65` precedent) or sit at a
Challenger-50/75 tier TML doesn't carry.

Player-resolver concrete misses (recent, both sides counted):
**Pucinelli de Almeida / Dalla Valle / Saraiva Dos Santos / Dutra Da
Silva / Longwe-Smit / Hohmann / Koenig / Latinovic / Mukund / Roddick /
Shimanov / Cretu**. The pattern is striking: many of these surface as
short last-name strings on the b-side (`Dalla Valle`, `Mukund`,
`Hohmann`) where Kalshi gives a partial name and the resolver returns
`player_b_id=nan`. 193 of 360 recent `missing_player_id` rows are
b-side-only; only 11 are both-sides-missing. The Kalshi name normalizer
is the obvious suspect.

`low_player_coverage` `cov=0` on one side with `cov=22/56` on the other
(e.g. **Dimitar Kuzmanov vs Szymon Kielan**, **Nathaniel Lammons vs
Gabi Adrian Boitan**, **Ioannis Xilas vs Samuel Vincent Ruggeri**) is
the same join failure surfacing through a different gate — these
players exist in TML, we just aren't finding them.

## Label / inference inconsistencies

**The v2.2-style mislabel is still live in settled data.** 21 of 227
settled rows have `kalshi_series == KXATPCHALLENGERMATCH` but
`tourney_level == "250"` — all 21 are Cordoba, all Clay, the same
Cordoba-name-collision that motivated the v2.2 → v2.3 revert. The
2026-05-27 diagnostic.md correctly ignored `tourney_level` and used
`kalshi_series` (good); but the *underlying inference path*
(`_infer_surface_and_level` / its tier-filtered variant in `6e8b48b`)
is still emitting the wrong level for these 21 rows. The proposal/
critique correctly identified this and even names the Cordoba row that
ran ROI +165.9% on a single bet — a `kalshi_series=Challenger,
tourney_level=250` market that exists today and would mislead any
downstream filter that consulted `tourney_level`.

Three tier sources-of-truth now exist in the codebase
(`tourney_level` from joiner, `tourney_level` from
`_infer_surface_and_level`, `kalshi_series`) and they disagree on 21
settled rows. Anything new that looks at tier is forced to pick one;
nothing enforces consistency.

Surface inference: 0 NaN in settled, but 100% Clay/Hard — no Grass or
Carpet has ever been settled, so the inference path on grass-season
events (June onward) is unexercised.

## Repeating bug patterns

`src/paper_trader.py` has been touched by **8 non-auto commits in the
last month** (`9a10f5a, b90f95d, 3a3d831, 914c669, dc65d0d, d7f680d,
56a9a81, 47eeff1, 6e8b48b`). Of those, **three (56a9a81, 47eeff1,
6e8b48b) are tier-labelling fix/revert/fix on a 7-day window**. This is
exactly the "3+ fix commits in a month" pattern this report is supposed
to flag. The tier identification path specifically has been on its
third source-of-truth in three weeks.

Adjacent: the player_resolver hasn't been touched since `b90f95d`
(initial ship) despite generating 695 lifetime drops and the
audit-only `1525b56` commit ("9 'failing' Kalshi IDs are red herrings —
opponent-side issues drive drops") that explicitly identified the
opponent-side b-side resolver problem but shipped no fix.

## Suggested follow-ups

- **(SEV-1)** ~78% of recent scanned markets are dropped because TML
  / resolver coverage is missing (`tournament_not_in_tml` +
  `missing_player_id` + `low_player_coverage` = 1,202 / 1,550); the
  user should investigate whether Kosice, Centurion, Cervia, Bengaluru
  3 are slug-map omissions vs. real TML coverage gaps before another
  bet-rule cycle.
- **(SEV-1)** `low_player_coverage` more than tripled WoW (111 → 360);
  combined with 193/360 b-side-only `missing_player_id` rows, the
  Kalshi-to-TML name resolver looks like it is silently regressing —
  the user should look at the b-side / partial-last-name failure mode
  (Dalla Valle, Mukund, Hohmann, Koenig, Latinovic) as a single
  resolver bug, not nine separate player misses.
- **(SEV-1)** Tier source-of-truth is on its third rewrite in a month
  and `_infer_surface_and_level` still emits `tourney_level == "250"`
  on the 21 Cordoba Challenger rows; the user should investigate
  whether the inference path can be deleted outright and
  `kalshi_series` made the single canonical tier field with a hard
  invariant.
- **(SEV-2)** `thin_tournament_history` drops jumped 8 → 88 in one
  week; flag for the user to confirm this is real Challenger-calendar
  churn and not a v2.3 plumbing side-effect that quietly tightened the
  gate.
- **(SEV-2)** `src/paper_trader.py` has 8 non-auto commits / 3
  tier-labelling fixes in a month; the user should consider whether
  the gate stack belongs in its own module before the next filter
  iteration touches the hot path again.
- **(SEV-2)** No Grass or Carpet has ever settled; the surface
  inference path is about to be exercised on out-of-sample data in
  June and the user should pre-audit before the calendar flips.
- **(SEV-3)** `BET_RULES.md` reserves "minor" for v2.1 / v2.2 only
  but the rejected v2.4 proposal also called itself minor; the
  semver convention is drifting and the user may want to clarify it
  before the next non-rejected proposal lands.
