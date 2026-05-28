# Diagnostic — 2026-05-27

Settled bets: **n=227** (179 v1.0, 22 v2.1, 26 v2.3). Overall ROI **−11.5%** (−$9.49 on $82.20 stake), win-rate **33.5%** on avg `theo_chosen` **0.528**. Versus last week (n=197, ROI −15.8%): lifetime ROI improved ~4pp, driven entirely by the 30 new post-cutoff settles. **All 227 rows are `kalshi_series=KXATPCHALLENGERMATCH`** — there is no main-tour cohort in the live data, exactly as v2.3 predicted. `tourney_level` shows 21 mis-labeled "250" rows (all the Cordoba ATP-250-name collision identified in v2.3); per CLAUDE.md / BET_RULES.md we ignore that column for tier.

Methodology: "overconfidence" = Wilson 95% CI upper bound for empirical win-rate falls strictly below `avg_theo`.

## Headline

This week (timestamp_settled > 2026-05-20): **n=31, ROI +12.6%, win-rate 41.9%** on stake $11.13 (net +$1.40). Of these, 26 are v2.3 and 5 are v2.1. v1.0 contributed 0 this week — the gated cohort is now driving the book. First positive weekly ROI in the dataset.

## By gate_version

| version | active? | n | wins | win_rate | avg_theo | ROI | net_pnl |
|---|---|---|---|---|---|---|---|
| v1.0 | retired | 179 | 56 | 0.313 | 0.532 | **−17.5%** | −$11.37 |
| v2.1 | superseded | 22 | 9 | 0.409 | 0.526 | **+0.8%** | +$0.07 |
| v2.3 | **CURRENT** | 26 | 11 | 0.423 | 0.496 | **+20.6%** | +$1.81 |

Combined post-gate (v2.1+v2.3) is n=48, ROI **+10.8%**, win-rate 41.7%, gap to avg_theo essentially zero (0.417 vs 0.510 — Wilson CI [0.286, 0.557] covers avg_theo, no overconfidence flag). v1.0 alone retains the historical −17.5% bleed and the calibration gap.

## By tier (kalshi_series)

100% of settled bets are **KXATPCHALLENGERMATCH** (n=227). There is no main-tour split to report. The 21 "tourney_level=250" Cordoba rows are series-Challenger and are counted as Challenger throughout. This confirms the v2.3 post-mortem: any future tier-based filter must key on `kalshi_series`, not `tourney_level`.

## By surface / theo / edge / direction

**Direction (the dominant signal):** lifetime YES n=187, ROI **−24.0%**, win-rate 29.4% vs avg_theo 53.0% (Wilson [0.233, 0.363] excludes 0.530 — strong overconfidence). NO n=40, ROI **+52.6%**, win-rate 52.5% vs avg_theo 51.7% (clean). Pattern from 2026-05-20 (YES −29.2% / NO +47.5%) holds. **Within the gated post-cutoff cohort**, the asymmetry narrows: YES n=42 ROI −6.3%, NO n=6 ROI +138.4% (n small but consistent direction).

**Surface:** Clay n=190 ROI −12.8% (Wilson [0.269, 0.401] excludes avg_theo 0.537 — overconfident). Hard n=37 ROI −4.6%, no overconfidence flag. **Hard improved sharply this week:** lifetime Hard moved from −56.0% (n=20) at 2026-05-20 to −4.6% (n=37); all 17 new Hard settles are v2.3 and ran ROI +34.7% (10/18 wins on YES Hard alone).

**Theo buckets:** the high-theo overconfidence persists but is much smaller in absolute n than before. [0.70, 0.80) n=22 ROI −28.1% (avg_theo 0.744 vs win-rate 0.409 — flagged). [0.80, 1.00] n=6 ROI −26.1% (avg_theo 0.966 vs 0.500 — flagged, but n=6). Low-theo [0.20, 0.30) n=17 went 0-for-17, ROI −105.9% — also overconfident (predicted 27.9% wins, got zero).

**Edge buckets:** smallest-edge **[5, 7)¢ is the only profitable bucket** (n=24, ROI +16.1%, win-rate 54.2%). All three higher-edge buckets lose, with [7, 10)¢ worst at −25.0%. Same "phantom edge at high edge" pattern from v1.0.

## Tournament outliers (n ≥ 5)

| tournament | n | win_rate | avg_theo | ROI | flag |
|---|---|---|---|---|---|
| Wuxi | 10 | 0.000 | 0.406 | **−104.5%** | 0-for-10, persistent |
| Tunis | 21 | 0.238 | 0.637 | −45.5% | overconfident |
| Francavilla | 12 | 0.167 | 0.488 | −45.5% | overconfident |
| Vicenza | 10 | 0.200 | 0.479 | **−40.7%** | NEW this week |
| Oeiras 4 | 22 | 0.273 | 0.559 | −33.7% | overconfident |
| Bordeaux | 26 | 0.385 | 0.597 | −11.6% | overconfident |
| Zagreb | 24 | 0.333 | 0.506 | −11.8% | |
| Valencia | 24 | 0.333 | 0.488 | −9.7% | |
| Istanbul | 17 | 0.412 | 0.493 | **+9.3%** | flipped positive (was −15.6%) |
| Cordoba | 21 | 0.429 | 0.477 | +32.3% | clean (mis-labeled "250") |
| Santos | 10 | 0.600 | 0.657 | +25.2% | |
| Brazzaville | 9 | 0.333 | 0.395 | +37.6% | |
| **Little Rock** | 9 | 0.667 | 0.519 | **+65.4%** | NEW this week, top |

## Overconfidence flags

Slices where Wilson-95 upper bound < avg_theo:

- direction=YES (n=187, gap −0.236)
- surface=Clay (n=190, gap −0.205)
- gate=v1.0 (n=179, gap −0.219)
- theo=[0.20, 0.30) (n=17, gap −0.279) and theo=[0.70, 0.80) (n=22, gap −0.335)
- edge=[7, 10), [10, 15), [15+] all flagged
- Tournaments: Wuxi, Tunis, Francavilla, Oeiras 4, Bordeaux

**Not flagged:** every v2.1 and v2.3 slice taken alone, NO direction, Hard surface, edge=[5, 7), and the profitable tournaments. The gated cohort's calibration looks reasonable; the v1.0 backlog is the remaining bleed.

## Continuity with prior weeks

What 2026-05-20 flagged, status now:

- **YES-bleed**: still real lifetime (−24.0% vs −29.2%), but **materially smaller inside v2.3** (YES ROI −6.3% on n=42). The direction-asymmetry committee finding from last week looks like it was driven primarily by v1.0; gated YES bets are not (yet) reproducing the pattern at the same magnitude.
- **Challenger=full book**: confirmed by `kalshi_series` — 227/227.
- **Cursed-tournament cluster** (Wuxi/Tunis/Oeiras 4/Francavilla): all four are still the worst non-this-week tournaments and none received new settles, so v2.1's `thin_tournament_history` gate is mechanically excluding them — persistent failure mode that the active gate already addresses.
- **High-theo overconfidence** (≥0.75 was −48.6% on n=15 last week): now [0.70, 0.80) n=22 ROI −28.1% and [0.80, 1.00] n=6 ROI −26.1%. Still flagged, still small-n, still v1.0-dominated.
- **Hard-surface bleed** from Istanbul (−15.6% on n=10 last week): **reverted** — Istanbul is now +9.3% on n=17, and lifetime Hard is −4.6%. v2.3's new Hard bets on Little Rock + Istanbul are the source of this week's positive ROI.
- **New persistent bleeder**: Vicenza (n=10, −40.7%, 2-for-10) — first time it crosses the n≥5 threshold.

Caveats: v2.3 n=26 and v2.1 n=22 are individually below conventional confidence floors. The headline +12.6% weekly ROI is on n=31 / $11 stake — directionally encouraging but not statistically significant.
