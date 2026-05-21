# Proposal — 2026-05-20

## Problem

The diagnostic's strongest, structurally-grounded finding is that **YES bets on
Challenger events are the entire bleed**: direction=YES is n=160, ROI −29.2%,
win-rate-vs-theo gap −0.258 (z = −7.30), and tourney_level=C is n=176, ROI
−20.6% (z = −6.61). NO bets are clean (n=37, +47.5% ROI, near-perfect
calibration) and ATP-250 events are clean (n=21, +32.3% ROI, gap −0.048). The
two failure axes (YES, Challenger) are highly entangled with each other but
*not* with the clean slices — so the safest single knob is the intersection:
drop YES bets on Challenger events.

## Change

Add a new drop reason `REASON_YES_ON_CHALLENGER` in `src/paper_trader.py` and
enforce it inside `_score_match_from_features` immediately after the
`best = max(candidates, key=...)` pick and before `MIN_EDGE` is checked. The
filter looks at the *chosen* candidate only: if the best edge is on the YES
side AND `level == "C"`, drop the group.

**File: `src/paper_trader.py`**

Add a constant near the existing gate constants:

```python
# Direction-asymmetry guard for Challenger tier. n=176 settled bets on
# Challenger have ROI −20.6% (gap −0.230, z=−6.61); within that bucket
# YES bets bleed −29.2% (n=160) while NO bets earn +47.5% (n=37) and
# calibrate cleanly. ATP-250+ events are unaffected (n=21, ROI +32.3%).
# Mechanism is that public-stats features systematically overrate the
# nominal favorite on thin Challenger fields, so favorite-YES picks
# inherit the bias; NO-side picks (underdog or unfavored side) don't.
DROP_YES_ON_CHALLENGER = True
```

Add a reason code near the others:

```python
REASON_YES_ON_CHALL  = "yes_on_challenger"
```

Add this block in `_score_match_from_features`, right after `best = max(...)`
and before the `MIN_EDGE` check (i.e. between current lines 709 and 712):

```python
        # Direction-asymmetry guard on Challenger tier. See BET_RULES.md v2.2.
        # Drops the bet if our best candidate is a YES bet on a Challenger.
        # NO bets on Challenger and ANY bet on ATP-250+ are unaffected.
        if DROP_YES_ON_CHALLENGER and best["direction"] == "YES" and level == "C":
            return self._drop_group(
                group, ts, REASON_YES_ON_CHALL,
                f"best={best['direction']} on Challenger "
                f"(theo={best['theo']:.3f}, edge={best['edge']:.3f})",
            )
```

Bump the version constant:

```python
GATE_VERSION          = "v2.2"
```

No other constants change. `MIN_EDGE` and `MAX_SPREAD` are untouched. No
retraining. No new features. No new dependencies.

## New version

`v2.1` → **`v2.2`**. This is a new filter inside the existing gate family
(same model, same features, same data sources) per the bump rules in
`BET_RULES.md`.

## Expected impact

- **Targeted failure mode**: YES-on-Challenger bets. n=160 historical, ROI
  −29.2%. Removes the direction-asymmetry bleed completely on the dominant
  cohort.
- **Rough ROI bound**: counterfactually removing all 160 YES-on-Challenger
  bets from the historical n=197 settled set leaves n≈37 NO-on-Challenger
  (+47.5%) + n≈21 ATP-250 mostly-YES (+32.3%) + a few ATP-250 NO bets. The
  surviving cohort had aggregate ROI roughly +35–45% on n≈58. After
  shipping, expected steady-state ROI on the kept set is in the
  **+5% to +20%** range with wide bounds (we expect some regression because
  v1.0's clean slices were small and partly lucky); the strong claim is
  only that v2.2 should not be ROI-negative, not that it will match the
  historical +35%.
- **Slices it touches**: every YES-on-Challenger pick — i.e. the Tunis /
  Oeiras 4 / Wuxi / Francavilla / Bordeaux / Zagreb / Valencia bleeders
  whenever they show up as favorite-YES, plus all the overconfident-
  favorite theo ≥0.75 picks (15/15 of which were YES on Challenger).
- **Slices it does NOT touch**:
  - NO bets on Challenger (n=37, +47.5% — preserved).
  - ATP-250+ events (n=21, +32.3% — preserved; the lone profitable cohort
    we don't want to choke off).
  - All v2.1 player-coverage / mirror-sum / tournament-history / imputation
    gates remain stacked on top, so the kept set is still narrowed by
    those.
- **Volume cost**: high — historically ~81% of placed bets are YES and
  ~89% are Challenger, so the intersection is likely ~70–75% of flow. The
  rule trades volume for sign. This is the correct trade given a negative
  ROI baseline; betting less is not a cost when the bets we'd drop have
  negative expectation.

## Sample-size justification

- **Target slice n is large.** The two slices being addressed are n=160
  (direction=YES) and n=176 (tourney_level=C). Both clear the n=15
  diagnostic floor by an order of magnitude. The intersection is at least
  max(160, 176) − 21 ≈ 139 (since only ~21 bets are ATP-250). z-scores
  −7.30 and −6.61 are far beyond noise.
- **Not optimized against a thin slice.** Deliberately *not* using
  theo_chosen ≥0.75 (n=15, ROI −48.6%) as the filter axis even though its
  per-bet bleed is worst, because n=15 is below the devil's-advocate
  threshold. Also not using surface=Hard (n=20) for the same reason. The
  filter axes chosen are the two highest-n axes in the report.
- **Structural, not noise.** The YES/NO asymmetry was the headline failure
  of the v1.0 post-mortem (−29% YES on n=141 vs +37% NO on n=34) AND it
  has persisted through v2.0 and v2.1 gates that didn't touch direction.
  That's two independent observations of the same asymmetry across
  different filter regimes — strong evidence it's a systematic feature-
  side bias (public stats overrate nominal favorites on thin fields), not
  a one-day blow-up. Tunis alone is n=21 with −45.5% ROI, so the bleed
  isn't concentrated in a single tournament.
- **Not lookahead.** The filter uses only `best["direction"]` (from the
  candidate table built from current Kalshi quotes) and `level` (from TML
  prior to event_date via `_infer_surface_and_level`). No future
  information.
- **Not chasing v2.1's small sample.** v2.1 is n=18 (descriptive only). The
  filter is justified on the v1.0 n=179 cohort where the YES/Challenger
  bleed is overwhelming. If v2.1's own filters happen to already trim the
  bleed in practice, this rule is a belt-and-suspenders; if they don't,
  this rule still bites. Either way the n is sufficient to act on.
- **One knob, one digit bump.** No multi-filter compound change, no
  threshold change to existing constants, no model retrain. Minimum viable
  diff.
