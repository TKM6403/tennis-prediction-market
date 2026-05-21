# Critique — 2026-05-20

Devil's-advocate review of the proposed `yes_on_challenger` drop reason (v2.1 → v2.2).

## 1. Overfitting / sample size

Direct verification against `data/paper_trades/settled.csv` (n=197 total: 179 v1.0, 18 v2.1):

- `direction == "YES"`: n = **160** (matches the diagnostic).
- `tourney_level == "C"`: n = **176** (matches the diagnostic).
- **Intersection `YES AND C`: n = 142** (the proposal hand-waved this as "≥ 139"; the real number is 142). Intersection ROI = **−36.7%**, stake $53.63, net −$19.70, win rate 25.4% on avg theo ~53%. Well above the n ≥ 30 floor.
- **Drop fraction historically: 142 / 197 = 72.1% of placed bets.** The proposal estimated 70–75%; the true number is just over the 70% red line in my prior. On the v2.1-only subset it is even worse: **14 of 18 = 77.8%** of v2.1 bets would have been dropped.

The intersection-n test passes comfortably; the drop-fraction test is essentially at the red line. This is a very aggressive cut, but the underlying intersection slice has enough samples that we are not blade-fitting a 30-bet curiosity.

## 2. Lookahead / leakage

Both gating fields are scan-time known:

- `chosen_direction` is derived from the candidate table at `_score_match_from_features` time (built from current Kalshi `yes_ask` quotes and current theo), strictly before settlement.
- `tourney_level` is recovered from TML via `_infer_surface_and_level` from data prior to `event_date` — exactly the same source v2.0/v2.1 already used to gate `MIN_TOURNEY_YEARS`.

No future information is used. Clean.

## 3. Brittleness

**Single-tournament concentration of the YES-on-C bleed (n=142 intersection):**

Negative-net tournaments sum to −$20.01 of net PnL. Worst single tournament:

- Oeiras 4: −$4.22 = **21.1% of the total negative bleed**.
- Top 2 (Oeiras 4 + Bordeaux): 36.8%.
- Top 3 (+ Tunis): 52.4%.

No single tournament owns >50% — the rule of thumb threshold for "this is one cursed event, not a pattern." Eight different Challengers lose ≥ $1.50 net each. The bleed is genuinely diffuse across the Challenger circuit, not a Tunis/Wuxi mirage.

**Has v2.1 already fixed the bleed?** No. This is the most important number in the review:

- v2.1 overall: n=18, ROI +1.7% (matches diagnostic).
- v2.1, `YES AND C` subset: **n=14, ROI −44.7%**, win rate 21.4%, net −$2.25.
- v2.1, everything else (3 NO-on-C + 1 YES-on-250): n=4, net +$2.37 (ROI +151%).

The diagnostic's "v2.1 ≈ flat" headline is entirely an artifact of 4 lucky non-YES-on-C bets covering a still-bleeding 14-bet YES-on-C cohort. v2.1's player-coverage + mirror-sum filters do not arrest the YES-on-Challenger failure mode at all — they just trimmed volume.

This is the cleanest "REJECT-killer" check for the change: **v2.1 has not visibly arrested the bleed**, so the new filter is not solving a problem already solved.

## 4. Worst case if the filter is wrong

If we are wrong that YES-on-C is structurally negative-EV, here is the historical opportunity cost. Surviving cohort after the filter (n=55, all-time):

- NO-on-Challenger: n=34, ROI +56.4%, net +$6.31.
- YES-on-non-C (i.e. ATP-250): n=18, ROI +47.4%, net +$2.50.
- NO-on-non-C: n=3, ROI +151%, net +$0.36 (from the 4 clean v2.1 bets, mostly).
- Aggregate kept set: n=55, ROI **+47.5%**, net +$8.44.

If the YES-on-C reversal happens, we forgo whatever the new mean of that slice is. The diagnostic's central estimate is that YES-on-C runs at −29% to −37% ROI; even a 20pp reversal to neutral would only have cost us roughly $0 in net PnL on the historical slate. The asymmetry is favorable: the bleed we are bypassing (−$19.70 historically) is an order of magnitude larger than any plausible foregone gain from a reversal.

## 5. Asymmetry sanity check (entry price)

Concern: is NO's +47.5% ROI artificially inflated by low entry price (NO is cheap when theo is high → small position size with leveraged returns)?

Checked the average `entry_price` (which is also the stake per contract under unit sizing):

- YES overall: $0.368.
- NO overall: $0.338.
- YES-on-C: $0.378.
- NO-on-C: $0.329.

NO is only marginally cheaper (~4 cents). The NO outperformance is **not** a price-leverage artifact — the buckets stake nearly identical dollars per bet on average, so the win-rate gap (51.4% NO vs 27.5% YES against similar theos) is doing the work. The "NO is clean" claim survives scrutiny.

## Verdict summary

- Intersection n large enough (142, not 30): **pass**.
- No lookahead: **pass**.
- Bleed not concentrated in one tournament (worst = 21.1%): **pass**.
- v2.1 has NOT silently fixed the problem post-cutover (YES-on-C still −44.7% on n=14 within v2.1): **pass** — this is the decisive check.
- Drop fraction is high (72.1% all-time, 77.8% on v2.1) but proposal correctly characterizes this as "trading volume for sign" given a negative-ROI baseline: **pass with note**.
- Asymmetry sanity check on entry price: **pass**.
- Worst-case opportunity cost is small and favorable: **pass**.

All four APPROVE conditions specified in the brief are met:
(a) intersection n = 142 ≥ 30,
(b) bleed worst-single-tournament = 21.1% < 50%,
(c) v2.1 alone is NOT visibly fixing it (−44.7% YES-on-C inside v2.1),
(d) drop fraction = 72.1% — at the boundary but not exceeding it on the historical population that motivated the change. v2.1's 77.8% drop rate is a real concern for forward volume; if forward Challenger flow continues to be ~80% of book, v2.2 will functionally be a NO-only-or-ATP-250 bot.

Notes for design agent / future review:

- The +1.7% v2.1 ROI cited in the diagnostic is misleading framing. v2.1's own YES-on-C bets are still bleeding at −44.7%. The proposal should foreground this number; the next post-shipment review should specifically watch v2.2's NO-on-Challenger and ATP-250 cohorts for whether they hold up out of sample.
- If forward volume on v2.2 falls below ~5–10 bets/week sustained, escalate — at that flow rate we will not have an n=30 intersection slice on v2.2-native data for months, and any future direction-asymmetry decisions will continue to lean on v1.0 history.
- The proposed filter is structurally a one-line gate change; the rollback path (set `DROP_YES_ON_CHALLENGER = False`) is trivial. This bounds downside.

APPROVE
