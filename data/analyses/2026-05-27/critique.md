# Critique — 2026-05-27

## Verdict

REJECT: the headline "YES inside gated cohort -6.3%" is a single-tournament, single-week blow-up (Vicenza, 10 bets settled 2026-05-25 → 2026-05-27). Strip Vicenza and the gated-YES slice flips to **+10.0% ROI on n=32**; within the *current* gate (v2.3) alone, YES ex-Vicenza is **+59.8% ROI on n=13 (8/13 wins)**. The filter as written would drop exactly the bets that are working right now.

## Reasoning

### 1. Overfitting / single-event blow-up

This is the disqualifying check. I reproduced every number in the proposal from `data/paper_trades/settled.csv` (227 rows). The proposal motivates itself by pointing at lifetime YES (n=187, -24.0%) and gated-cohort YES (n=42, -6.3%). The lifetime number is dominated by v1.0 (179/227 rows, -17.5% ROI on its own) — the gate stack has *already* mostly addressed that bleed. The relevant slice is the gated cohort.

Decomposition of the gated YES slice (v2.1+v2.3 combined):

| tournament                 | n | stake | pnl    | roi    |
|----------------------------|---|-------|--------|--------|
| Vicenza                    | 9 | 2.76  | -1.887 | -68.4% |
| Vicenza Qualification      | 1 | 0.30  | -0.315 | -104.9% |
| Valencia                   | 2 | 0.93  | -0.963 | -103.6% |
| Little Rock Qualification  | 1 | 0.53  | -0.547 | -103.3% |
| Bordeaux                   | 1 | 0.38  | -0.396 | -104.3% |
| Zagreb                     | 1 | 0.35  | -0.366 | -104.6% |
| Istanbul                   | 17| 6.17  | +0.574 | +9.3%  |
| Cordoba                    | 1 | 0.37  | +0.614 | +165.9%|
| Little Rock                | 9 | 3.54  | +2.315 | +65.4% |

Gated YES total: n=42, ROI -6.3%. **Drop Vicenza (10 bets, all settled 2026-05-25 → 2026-05-27, in the most recent 3 calendar days)** and the slice is **n=32, +10.0% ROI**. The proposal cites no multi-week persistence inside the gated cohort because none exists — the diagnostic itself flags Vicenza as "NEW this week, first time it crosses the n≥5 threshold" (`diagnostic.md` line 42).

Inside `v2.3` alone (current gate), YES is n=23 ROI **+9.0%** (`v2.3 by chosen_direction` reproduces directly from settled.csv). Strip Vicenza: n=13, ROI **+59.8%**, 8/13 wins, on $4.88 stake → +$2.92 pnl. The proposal would have dropped every one of those bets.

The persistent-failure-mode claim is therefore false at the only level the proposal can act on. The pre-cutoff YES bleed is v1.0's, and v1.0 is retired. The remaining v2.3 YES bleed is one tournament-week.

### 2. Lookahead / leakage

Clean on this axis. `_series_from_market_id` (`src/paper_trader.py:232`) parses the Kalshi ticker prefix off `market_id`, which is set at scan time from live Kalshi quotes. `best["market_id"]` is the field already on the candidate dict at line 731 / 744. No model output, no settled-row data, no future information. If the filter were otherwise justified, this implementation is sound.

### 3. Brittleness / repeat of past mistake

Verified: the proposal's claim that `_series_from_market_id` exists and is available at the guard site is correct. `_series_from_market_id` is defined at line 232; it is called at lines 443, 507, 517, 834, 960 — including the `kalshi_series` column written into every bet row (line 834) on the same `best["market_id"]` the proposal uses. So the field-reliability fix is real and is not a recurrence of the Cordoba TML-mode-lookup bug.

That said, "the bug is fixed" is necessary but not sufficient. The proposal correctly identifies that v2.2's *implementation* was buggy. It does not establish that v2.2's *thesis* (YES-on-Challenger is systematically broken) survives once the gate stack already in place is doing its job. See check (1) — it does not.

### 4. Worst case

The guard drops 187/227 = 82.4% of lifetime bets and 42/48 = 87.5% of gated-cohort bets. Most of that is academic (v1.0 retired bets we wouldn't place now). The forward-looking question: what does it drop from the v2.3 stream?

v2.3 YES is 23/26 = 88% of current bets. Of those 23, **13 are non-Vicenza and ran +59.8% ROI**. Drop those forward at the same rate of arrival (~3 weeks of v2.3 produced n=26 → roughly n=10/week) and the lost upside on the kept-set delta is on the order of **-$1 to -$2/week of pnl forfeited** plus a near-zeroing of book volume on Challenger-only weeks. The diagnostic confirms book is 100% Challenger.

NO-on-Challenger and main-tour bets: confirmed unaffected by the filter as proposed (`best["direction"] == "YES" AND chosen_series == "KXATPCHALLENGERMATCH"`). That part is correct. But "preservation of NO" is a hollow benefit: NO is 6/48 = 12.5% of the gated cohort and 40/227 = 17.6% of lifetime. The bot would become a tiny-volume NO-only Challenger bot, and any one bad NO week (n=6 base) wipes out the kept-set edge. The proposal acknowledges this risk (proposal.md §"Uncertainty" point 2) without pricing it.

Worst case bleed if wrong: not a bleed of capital, a bleed of *information*. A week with very few bets placed is a week where we can't tell if the gate stack is working. The next review is 7 days away. If the right diagnosis is "Vicenza is a thin/cursed Challenger that the `thin_tournament_history` gate should have caught and didn't" (the cursed-4 pattern, restated), then v2.4 papers over that miss and we lose another cycle to learn the actual lesson.

### 5. Selection bias on the proposing slice

YES-on-Challenger has been the focal "obvious bleed" for two weeks running. Persistence is evidence — but only if the persistence is in the slice that the new gate would actually act on. It is not. The gated-cohort persistence is one tournament-week deep. v1.0's persistence is moot because v1.0 is retired. The 3-agent committee that produced v2.2 was operating on Cordoba-mislabeled data; the proposing agent here is operating on correctly-labeled data but, judging by the way it leans on the lifetime n=187 figure and treats the gated n=42 as a confirmatory checkbox, has not done the within-gate tournament decomposition.

## Worst case bleed if wrong

Capital: small in absolute terms (lifetime book stake is $82.20; one week's stake under v2.3 is ~$3-5). Forward upside forfeited on the kept set, if v2.3 YES ex-Vicenza is the real signal at +$1.50/week pnl, is on the order of **-$5 to -$10 over the 7-day window to the next review**. In a book this small, that's the entire week's expected positive ROI.

The bigger cost is observational: with ~80%+ of candidates dropped on a Challenger-only book, weekly n collapses to ~2-4 bets. Confidence intervals at that n are too wide to update the gate stack again in 7 days, so a wrong v2.4 can compound by delaying its own correction.

## Conditions for re-evaluation

I would APPROVE a YES-on-Challenger-style guard if any of the following hold next cycle:

1. **Multi-tournament persistence inside v2.3+**: gated YES is negative across at least 3 distinct tournaments that aren't all within one calendar week. Right now it's negative on Vicenza and arithmetic noise (1-2 bet "tournaments"); Istanbul and Little Rock are sharply positive.
2. **A more targeted filter passes the same data**: the actual failure mode this week is Vicenza, which looks like the `thin_tournament_history` / cursed-tournament pattern reasserting. A v2.4 that tightens `MIN_TOURNEY_YEARS` from 3 to 4 (or adds a Challenger-specific player-coverage floor) and shows it would have dropped the Vicenza cluster while keeping Istanbul + Little Rock would be a much better use of this slot.
3. **The asymmetry returns once Vicenza settles out**: if next week's diagnostic still shows gated-YES negative *after* Vicenza is in the rear view, that's the multi-week persistence check #1 asks for and the prior strengthens materially. Re-propose then.
4. **Refinement, not blanket guard**: a YES-on-Challenger guard *additionally gated* on a thin-data signal (e.g. low player coverage near the threshold, low TML tournament history, or first-time-this-event) — i.e. a filter that names the mechanism rather than the symptom — would be evaluated on its own merits and likely approved if it cuts Vicenza-class events without cutting Istanbul/Little Rock.

Until one of those is true, the gate stack already in place looks like it's doing the work the proposal is trying to assign to a new filter, and the new filter would destroy the only positive signal in the live book.
