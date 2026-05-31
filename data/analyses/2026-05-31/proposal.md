# Proposal — 2026-05-31 (DESIGN agent)

## Problem

The dominant, most robustly-sampled failure mode in this week's diagnostic is the
**YES/NO direction asymmetry**:

| dir | n | win_rate | avg_theo | ROI | net |
|---|---|---|---|---|---|
| YES | 200 | 0.290 | 0.528 | **−25.2%** | −$18.58 |
| NO  | 40 | 0.525 | 0.517 | **+52.6%** | +$7.04 |

YES Wilson-95 CI [0.232, 0.356] excludes its avg_theo of 0.528 — the model is
systematically overconfident on the favorite-YES side. NO bets are well
calibrated. This pattern has held across every weekly diagnostic (cf. v2.2
diagnostic 2026-05-20: YES n=160 −29.2% vs NO n=37 +47.5%) and is the single
best-sampled, most persistent signal we have.

This is exactly the failure mode v2.2 (`yes_on_challenger`) targeted. v2.2 was
**reverted not because the signal was wrong, but because the gating field was
wrong**: it gated on `tourney_level == "C"`, a TML name-mode lookup that
mislabeled name-collision Challengers (Cordoba → "250"). v2.3 fixed precisely
this by making `kalshi_series` the canonical tier source and confirming **all
240 live rows are `KXATPCHALLENGERMATCH`** — there is no real ATP-250 cohort to
protect. The plumbing that made v2.2 unsafe is now in place; the filter can be
re-enabled correctly.

## Change

Re-enable the direction-asymmetry guard, but gate on the **`kalshi_series`**
field (now canonical per v2.3) rather than the unreliable `tourney_level`.

**Edit 1 — flip the flag** in `src/paper_trader.py` (line 116):

```python
DROP_YES_ON_CHALLENGER = False
```
→
```python
DROP_YES_ON_CHALLENGER = True
```

**Edit 2 — gate on the canonical series field**, not `level`, in
`_score_match_from_features` (`src/paper_trader.py`, lines 791–796). The
`primary` mirror is already in scope in this method via `meta["primary"]`;
derive the series from it with the existing `_series_from_market_id()` helper:

Current (lines 791–796):
```python
if DROP_YES_ON_CHALLENGER and best["direction"] == "YES" and level == "C":
    return self._drop_group(
        group, ts, REASON_YES_ON_CHALL,
        f"best={best['direction']} on Challenger "
        f"(theo={best['theo']:.3f}, edge={best['edge']:.3f})",
    )
```
New:
```python
series = _series_from_market_id(primary["market_id"])
if DROP_YES_ON_CHALLENGER and best["direction"] == "YES" \
        and series == "KXATPCHALLENGERMATCH":
    return self._drop_group(
        group, ts, REASON_YES_ON_CHALL,
        f"best={best['direction']} on {series} "
        f"(theo={best['theo']:.3f}, edge={best['edge']:.3f})",
    )
```

This is a single new active filter on an existing reason code
(`REASON_YES_ON_CHALL`, already defined) gating on a scan-time-known,
no-lookahead field. No constant thresholds change; `MIN_EDGE` and `MAX_SPREAD`
are untouched.

**Edit 3 — bump the version** in `src/paper_trader.py` (line 157):
```python
GATE_VERSION = "v2.3"
```
→
```python
GATE_VERSION = "v2.4"
```

(Code is shown for the committee; per instructions I am NOT editing any source
files in this step — only writing this proposal.)

## New version

**v2.4** — minor bump (re-enabling/adding a filter inside an existing gate
family; same model, same features, same data sources). Single minor digit, no
major bump.

## Expected impact

Failure mode addressed: YES-side overconfidence on Challenger markets — the
−25.2% ROI YES cohort. With all live flow on `KXATPCHALLENGERMATCH`, this filter
drops every best-edge YES candidate and keeps only NO candidates (and, if a
main-tour `KXATPMATCH` series ever appears, YES on those is preserved).

- **Direct effect:** removes the YES cohort (n=200, −25.2%, −$18.58) from future
  flow; retained NO cohort ran +52.6% (n=40, +$7.04).
- **Expected forward ROI on kept bets:** centered near the NO cohort's +52.6%,
  but I bound this conservatively. The NO sample is small (n=40) and its point
  estimate is inflated by variance, so the honest expectation is **roughly
  break-even to strongly positive**: bounds **[0%, +50%]**, most-likely band
  **+10% to +30%**. The robust claim is directional: we stop placing a cohort
  with a CI that excludes its own theo (a structurally losing book) and keep the
  only calibrated one.
- **Volume cost (high, and the main risk):** historically ~80% of placed bets
  are YES-on-Challenger, so v2.4 functionally becomes a NO-on-Challenger bot.
  Acceptable — betting less is not a cost when the dropped cohort has negative
  expectation — but flagged for next review, matching the v2.2 caveat.
- **Opportunity cost if the YES bleed reverses out of sample:** small. Even a
  20pp reversal of the dropped slice only zeroes it out; it does not exceed the
  historical bleed prevented.

This is the same mechanism v2.2 proposed, now landed on the correct,
bug-free field — so it inherits v2.2's expected-impact analysis without v2.2's
mislabeling defect.

## Sample-size justification

- **Relevant slice is large and clean:** YES n=200 vs NO n=40 — the
  best-sampled split in the diagnostic. YES Wilson-95 CI [0.232, 0.356] lies
  entirely below avg_theo 0.528 (z ≈ −7 in the equivalent 2026-05-20 cut),
  so the overconfidence is not noise.
- **Persistence across reviews:** the same sign and rough magnitude appears in
  every prior diagnostic (2026-05-20: YES −29.2% / NO +47.5%; 2026-05-27:
  unchanged; today: −25.2% / +52.6%). A signal that survives four weekly cuts
  is not overfit to one slice.
- **Deliberately NOT chasing the noisy slice:** the −29.6% current-period
  headline rests on n=18 (all YES, all v2.3, Wilson CI [0.125, 0.509] still
  covers avg_theo) and the v2.3 flip on +13 settles. Per the diagnostic's own
  caveat these are within noise. This proposal acts on the n=200/n=40 direction
  split instead — ~5–11× better sampled than anything in the recent window.
- **Caveat on the kept side:** NO n=40 is itself modest, which is why the
  expected-ROI bound is widened to include break-even rather than asserting
  +52.6%. The decision rests on the YES side being a proven loser (large n,
  CI-excluded), not on the NO side being a proven winner.
