# Diagnostic — 2026-05-31

Settled bets: **n=240** (179 v1.0, 22 v2.1, 39 v2.3). Settle-timestamp range **2026-05-07 → 2026-05-31**. Overall ROI **−13.3%** (−$11.54 on $87.06 stake), win-rate **32.9%** (79/240) on avg `theo_chosen` **0.526**. Overall Wilson-95 win-rate CI [0.273, 0.391] sits entirely below avg_theo → book-level overconfidence persists. **All 240 rows are `kalshi_series=KXATPCHALLENGERMATCH`** — no main-tour cohort exists in live data, as v2.3 predicted. `tourney_level` still mislabels 21 Cordoba rows "250"; ignored per BET_RULES.

Method: "overconfidence" = Wilson-95 upper bound for empirical win-rate < avg_theo. Stake = `entry_price`.

## Headline — this period regressed

This period (settled since 2026-05-27): **n=18, ROI −29.6%, win-rate 27.8%** (5/18), avg_theo 0.517, net −$1.99. All 18 are v2.3 and **all 18 are YES** (every NO bet in the book settled before this window). This reverses last week's headline (+12.6% on n=31). Wilson CI [0.125, 0.509] still covers avg_theo, so on n=18 this is **likely noise**, not a confirmed regime change.

## By gate_version

| version | active? | n | wins | win_rate | avg_theo | ROI | net | flag |
|---|---|---|---|---|---|---|---|---|
| v1.0 | retired | 179 | 56 | 0.313 | 0.532 | **−17.5%** | −$11.37 | OVERCONF |
| v2.1 | superseded | 22 | 9 | 0.409 | 0.526 | **+0.8%** | +$0.07 | |
| v2.3 | **CURRENT** | 39 | 14 | 0.359 | 0.498 | **−1.8%** | −$0.24 | |

v2.3 grew 26→39 (+13 settles) and slipped from +20.6% to −1.8% lifetime; the 13–18 new bets ran negative. Combined gated (v2.1+v2.3) n=61, near break-even, no overconfidence flag — still the cleanest cohort.

## By direction (dominant signal)

| dir | n | win_rate | avg_theo | ROI | net | flag |
|---|---|---|---|---|---|---|
| YES | 200 | 0.290 | 0.528 | **−25.2%** | −$18.58 | OVERCONF |
| NO | 40 | 0.525 | 0.517 | **+52.6%** | +$7.04 | clean |

YES Wilson [0.232, 0.356] excludes 0.528. NO well-calibrated. Pattern unchanged from prior weeks. Within v2.3: YES n=36 ROI −10.5% (flagged); NO n=3 ROI +127% (n=3, ignore).

## By surface / theo / edge

| surface | n | win_rate | avg_theo | ROI | flag |
|---|---|---|---|---|---|
| Clay | 199 | 0.332 | 0.536 | −13.0% | OVERCONF |
| Hard | 41 | 0.317 | 0.479 | −14.6% | OVERCONF |

Hard regressed sharply from prior −4.6% (n=37) to −14.6% (n=41): the 5 new Hard settles (all Little Rock) went 0-for-5, ROI −104%.

| theo bucket | n | win_rate | avg_theo | ROI | flag |
|---|---|---|---|---|---|
| [0.5,0.6) | 49 | 0.510 | 0.552 | **+30.4%** | clean |
| [0.6,0.7) | 50 | 0.400 | 0.651 | −20.4% | OVERCONF |
| [0.7,0.8) | 23 | 0.391 | 0.743 | −32.0% | OVERCONF |
| [0.8,1.0) | 6 | 0.500 | 0.966 | −26.1% | OVERCONF (n=6) |
| [0.0,0.3) | 20 | 0.000 | 0.272 | −105.9% | OVERCONF (0-for-20) |

High-theo overconfidence persists; [0.5,0.6) remains the one profitable band.

| edge bucket (¢) | n | win_rate | avg_theo | ROI | flag |
|---|---|---|---|---|---|
| [5,7) | 27 | 0.481 | 0.506 | **+4.5%** | clean |
| [7,10) | 40 | 0.375 | 0.541 | −21.2% | OVERCONF |
| [10,15) | 57 | 0.281 | 0.459 | −19.8% | OVERCONF |
| [15+) | 116 | 0.302 | 0.559 | −11.9% | OVERCONF |

Smallest-edge bucket still the only non-losing one — the "phantom edge at high edge" pattern holds.

## Tournaments (n ≥ 5)

| tournament | n | win_rate | avg_theo | ROI | flag |
|---|---|---|---|---|---|
| Wuxi | 10 | 0.000 | 0.406 | **−104.5%** | 0-for-10 |
| Tunis | 21 | 0.238 | 0.637 | −45.5% | OVERCONF |
| Francavilla | 12 | 0.167 | 0.488 | −45.5% | OVERCONF |
| Vicenza | 17 | 0.235 | 0.493 | −35.6% | OVERCONF |
| Oeiras 4 | 22 | 0.273 | 0.559 | −33.7% | OVERCONF |
| Zagreb | 24 | 0.333 | 0.506 | −11.8% | |
| Bordeaux | 26 | 0.385 | 0.597 | −11.6% | OVERCONF |
| Valencia | 24 | 0.333 | 0.488 | −9.7% | |
| Istanbul | 17 | 0.412 | 0.493 | +9.3% | |
| Little Rock | 13 | 0.462 | 0.502 | **+16.0%** | down from +65% |
| Santos | 10 | 0.600 | 0.657 | +25.2% | |
| Cordoba | 21 | 0.429 | 0.477 | +32.3% | (mislabeled "250") |
| Brazzaville | 9 | 0.333 | 0.395 | +37.6% | |

## Change vs prior diagnostic (2026-05-27)

- **Lifetime ROI worsened** −11.5% → −13.3% (13 new settles all negative).
- **v2.3 flipped** +20.6% → −1.8%; the gated-cohort optimism from last week did not hold.
- **Little Rock cooled** +65.4% (n=9) → +16.0% (n=13); the new 0-for-5 erased most of the edge — confirms last week's outlier was small-n luck.
- **Vicenza** crossed into a clearer bleeder: −40.7%(n=10) → −35.6%(n=17), still flagged.
- Cursed cluster (Wuxi/Tunis/Francavilla/Oeiras 4) unchanged — no new settles, gate mechanically excluding them.
- YES/NO asymmetry and high-theo/high-edge overconfidence all unchanged.

## Caveat

Only 18 bets settled since the last review, **all YES, all v2.3, all Clay/Hard Challenger** — no NO bets settled this window, so the period figure cannot speak to the YES/NO asymmetry. The −29.6% period ROI and the v2.3 flip both rest on n≤18 / ~$6.7 stake and are within noise; treat as a non-confirmation of last week's positive signal, not a confirmed reversal.
