# Diagnostic — 2026-05-20

Settled bets: n=197 (179 v1.0, 18 v2.1). Overall win rate 32.0% on avg_theo 0.53, ROI **−15.8%** (−$11.25 on $71.42 stake). No prior diagnostic in `data/analyses/` — this is the first.

Note on methodology: "overconfidence" flag = slice has n ≥ 15 **and** `win_rate − avg_theo_chosen ≤ 0` **and** the normal-approx z on that gap is ≤ −1.0. Strict z ≤ −2.0 is treated as high-confidence.

## Headline finding

**YES bets on Challenger clay with the v1.0 gate are the entire bleed.** Direction-wise, YES is −29.2% ROI on n=160 with win rate 27.5% vs avg_theo 53.3% (gap −0.258, z = −7.3). NO bets are profitable (+47.5% ROI on n=37, win rate 51.4% almost exactly matching avg_theo 51.9% — well-calibrated and edge-positive). The same asymmetry from the v1.0 post-mortem persists: when the model says "take the favorite YES," it loses; when it says "take the underdog via NO," it doesn't.

## Worst high-n slices (priority order)

| slice | n | win_rate | avg_theo | ROI | gap | z |
|---|---|---|---|---|---|---|
| direction=YES | 160 | 0.275 | 0.533 | −0.292 | −0.258 | −7.30 |
| tourney_level=C (Challenger) | 176 | 0.307 | 0.537 | −0.206 | −0.230 | −6.61 |
| surface=Clay | 177 | 0.339 | 0.541 | −0.119 | −0.202 | −5.69 |
| theo_chosen <0.55 | 104 | 0.221 | 0.403 | −0.188 | −0.182 | −4.47 |
| edge ≥0.20 | 59 | 0.322 | 0.598 | −0.040 | −0.276 | −4.53 |
| theo_chosen ≥0.75 | 15 | 0.333 | 0.848 | −0.486 | −0.515 | −4.23 |
| surface=Hard | 20 | 0.150 | 0.432 | −0.560 | −0.282 | −3.54 |

Clean (not flagged): tourney_level 250 (n=21, ROI +32.3%, gap −0.048), direction=NO (above), and tournament=Cordoba (n=21, +32.3%) / Santos (n=10) / Brazzaville (n=9).

## Theo / edge buckets

Calibration is broken across every theo bucket — the model is overconfident at the top **and** bottom. The ≥0.75 bucket (n=15) is the most extreme: predicted 84.8% win rate, actual 33.3% → ROI −48.6%. Even the supposedly-large-edge ≥0.20 bucket (n=59, avg_edge 27.7¢) is only break-even-ish at −4% ROI; the "edge" is mostly fake, exactly the v1.0 failure mode. The middle edge band (0.10–0.15, n=41) is the worst edge bucket at −33.4% ROI.

## By tournament (n ≥ 5)

Worst bleeders are the familiar "cursed Challenger" cluster:

| tournament | n | ROI | gap |
|---|---|---|---|
| Tunis | 21 | −0.455 | −0.399 |
| Oeiras 4 | 22 | −0.337 | −0.286 |
| Wuxi | 10 | −1.045 | −0.406 |
| Francavilla | 12 | −0.455 | −0.321 |
| Bordeaux | 26 | −0.116 | −0.212 |
| Zagreb | 24 | −0.118 | −0.173 |
| Valencia | 24 | −0.097 | −0.155 |

Tunis is uniquely toxic — model is most confident here (avg_theo 0.637, avg_edge 22.8¢) and loses worst. Wuxi is 0-for-10. Cordoba is the lone profitable ATP-250 stop.

## Since last review (2026-05-15)

n=36 settled (18 v1.0, 18 v2.1). Combined ROI is roughly flat (−$0.46 net on $13.90 stake). **v2.1's 18 bets returned +1.7% ROI (n=18, 38.9% wins on avg_theo 50.9%)** — too small to conclude, but no longer obviously bleeding and the theo-vs-realized gap shrank from −0.22 (v1.0) to −0.12. v2.1 has not yet placed a YES-favorite-clay-Challenger bet at the extreme theo ≥0.75 / edge ≥0.20 corner where v1.0 lost the most, so its sample is mechanically protected from the worst v1.0 regions. Hard-surface post-cutoff (n=10, all from one tournament, Istanbul) lost −15.6%.

## Caveats

- v2.1 n=18 is below the n=15 confidence floor for *any* sub-slice; treat all v2.1-specific numbers as descriptive only.
- 89.8% of the dataset is Clay Challenger — surface/level/direction signals are entangled with the dominant cohort.
- One Wuxi z-score blew up numerically (win_rate=0 → SE=0); ignore the z, the −104.5% ROI on n=10 is the real signal.
