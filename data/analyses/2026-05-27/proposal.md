# Proposal — 2026-05-27

## Problem

The direction-asymmetry failure mode flagged by the 2026-05-20 diagnostic is
still the single largest source of bleed in the book, and it is now sittable
on the correctly-labeled `kalshi_series` field that v2.3 plumbed.

From `diagnostic.md`:

> **Direction (the dominant signal):** lifetime YES n=187, ROI **−24.0%**,
> win-rate 29.4% vs avg_theo 53.0% (Wilson [0.233, 0.363] excludes 0.530 —
> strong overconfidence). NO n=40, ROI **+52.6%**, win-rate 52.5% vs
> avg_theo 51.7% (clean).

And:

> **Within the gated post-cutoff cohort**, the asymmetry narrows: YES n=42
> ROI −6.3%, NO n=6 ROI +138.4% (n small but consistent direction).

The asymmetry is the only overconfidence flag that survives the v2.1+v2.3
filter stack: every v2.1/v2.3 slice taken alone is calibrated, but YES as a
direction is *not* — even inside the gated cohort YES is still negative
while NO is sharply positive. Mechanism (carried over from v2.2's
post-mortem in `BET_RULES.md`): public-stats features overrate the nominal
favorite on thin Challenger fields; YES picks (typically backing that
favorite) inherit the bias, NO picks (typically fading it) don't.

The full book is 227/227 `kalshi_series == "KXATPCHALLENGERMATCH"` — there
is no ATP-250+ cohort to preserve — and v2.3 explicitly notes that the v2.2
filter code path is "left in place (just behind a `False` flag) so
re-enabling it on the correct field is a small, reviewable patch next
cycle."

## Change

**File:** `src/paper_trader.py`

**One-line semantic change:** re-enable the YES-on-Challenger direction
guard, but gate on the v2.3-plumbed `kalshi_series` ticker instead of the
TML-inferred `tourney_level` field that caused the v2.2 Cordoba bug.

Concretely, two edits, in the same function (`_score_match_from_features`):

1. Flip the existing flag from `False` to `True`:

   ```python
   DROP_YES_ON_CHALLENGER = True  # re-enabled in v2.4 on kalshi_series
   ```

2. Replace the guard's tier check so it reads from the chosen market's
   `kalshi_series` ticker (already computed via `_series_from_market_id`
   elsewhere in the same function for the bet row), not from the
   TML-mode-derived `level` local. The guard currently reads:

   ```python
   if DROP_YES_ON_CHALLENGER and best["direction"] == "YES" and level == "C":
   ```

   New form:

   ```python
   chosen_series = _series_from_market_id(best["market_id"])
   if (DROP_YES_ON_CHALLENGER
           and best["direction"] == "YES"
           and chosen_series == "KXATPCHALLENGERMATCH"):
   ```

   The `reason_detail` string should include `chosen_series` so the
   dropped.csv audit trail makes the gating field obvious.

No new constant, no new feature, no new dependency. The
`REASON_YES_ON_CHALL` reason code already exists from v2.2. `kalshi_series`
is a string the scanner already derives at scan time from the live Kalshi
market_id — no TML lookup, no lookahead, and not subject to the Cordoba
name-collision class of bug that killed v2.2.

## New version

`GATE_VERSION = "v2.4"` (minor bump).

Justified as minor because:

- No model retrain, no feature-set change, no new data source — the
  bump-rules in `BET_RULES.md` reserve major bumps for those.
- The filter family (`DROP_YES_ON_CHALLENGER`) and the reason code
  (`REASON_YES_ON_CHALL`) already exist in the codebase from v2.2; this is
  re-enablement of an existing dormant filter on a corrected field, not a
  novel filter family.
- Only one version digit moves (v2.3 → v2.4).

## Expected impact

**Targeted failure mode:** direction-asymmetry / YES-on-Challenger bleed.

**Mechanical effect on the live book:** since 227/227 settled bets are
`KXATPCHALLENGERMATCH`, the guard reduces to "drop the bet whenever the
best-edge candidate is YES." Lifetime that would have dropped 187/227 =
**82.4%** of placed bets, keeping the n=40 NO-side slice that ran +52.6%
ROI. Within just the gated cohort it would drop 42/48 = 87.5%, keeping the
n=6 NO slice (+138.4% ROI).

**Expected ROI delta on kept set:** the n=40 lifetime NO slice ran ROI
+52.6%; the n=6 gated-cohort NO slice ran +138.4%. Both are small samples,
so I won't pretend either point estimate is the forward-ROI prediction.
Honest bound:

- **Optimistic:** kept set continues to look like the historical NO slice
  → ROI in the +20% to +50% range on the bets that survive. That would
  flip the book from −11.5% lifetime to clearly positive on the new bets.
- **Central:** the NO-side advantage is partly a regression effect that
  shrinks as the gated cohort grows. Forward ROI on kept set in the
  +5% to +20% range — still meaningfully above the current gated-cohort
  combined +10.8%.
- **Pessimistic:** NO ROI compresses toward zero as the favorite-fading
  edge gets arbed away or as the mechanism doesn't carry to new
  tournaments. Forward ROI on kept set near 0% — *still strictly better
  than the −6.3% the YES side ran inside the gated cohort*, because the
  bets we're dropping had negative expectation.

**Volume cost:** very high. Expect ~80%+ of candidate bets to be dropped.
This is the same trade-off explicitly accepted in v2.2's design note:
"betting less is not a cost when the bets we'd drop have negative
expectation." The risk is not lost upside, it's slow accumulation of
forward evidence — we will need a longer window to confirm or reject the
direction.

**Uncertainty I'm honest about:**

1. The headline "narrowing" inside the gated cohort (YES n=42 ROI −6.3%)
   is one interpretation; another is that the v2.1+v2.3 filters are
   *already* eating most of the YES bias and v2.4 will mostly be cutting
   bets that were on track to break even. If that's true, ROI delta on
   kept set could be smaller than the lifetime YES vs NO gap suggests.
2. The NO slice is small (n=40 lifetime, n=6 inside the gated cohort) and
   its variance is real. A single bad week of Challenger NO picks could
   wipe out the entire historical edge.
3. We have not directly observed `kalshi_series != "KXATPCHALLENGERMATCH"`
   in any settled or pending bet. The guard is written to be tier-safe
   (it only drops on Challenger), so if and when the loader is widened to
   `KXATPMATCH` etc., this filter does not silently leak — but we also
   have no out-of-sample evidence that the asymmetry is Challenger-
   specific vs. universal. v2.4 takes the conservative position that the
   Challenger-only hypothesis from v2.2 is still the best-supported.

## Sample-size justification

The committee floor is n ≥ 30 in the targeted slice; the diagnostic flags
this is binding because v2.3 alone is n=26 and v2.1 alone is n=22.

The slice this proposal *targets* is YES-on-Challenger, and the relevant
n is the YES side of the lifetime book:

- **YES (lifetime, all gates):** n=187, win-rate 29.4%, ROI −24.0%,
  avg_theo 53.0%. Wilson 95% CI on the win-rate is [0.233, 0.363] — the
  *upper bound* (0.363) sits 0.17 below avg_theo (0.530). This is not a
  marginal overconfidence flag; it is a 16+ percentage-point gap with the
  CI strictly excluding the model's predicted rate. n=187 is well above
  any conventional floor.

- **NO (the slice we *preserve*):** n=40, win-rate 52.5%, ROI +52.6%,
  avg_theo 51.7% (Wilson covers avg_theo, no overconfidence flag). n=40
  also clears the n=30 floor.

- **YES inside the gated cohort (v2.1+v2.3 combined):** n=42, ROI −6.3%.
  This slice is what skeptics will (correctly) point to as "the gates
  already mostly fixed it." n=42 clears the floor but is much smaller in
  effect size than the lifetime YES slice.

The prior is also strong, independent of n:

- The direction-asymmetry pattern was the focal finding of the
  2026-05-20 diagnostic and was already vetted by a 3-agent committee.
- The mechanism is named in `BET_RULES.md` v2.2 ("public-stats features
  systematically overrate the nominal favorite on thin Challenger
  fields") and matches the broader tennis-modelling prior that public
  features lag behind market-makers' private info on lower tiers.
- The reason v2.2 was reverted was a *field bug* (TML name-collision on
  Cordoba), not a flaw in the underlying signal. The diagnostic on
  correctly-labeled data still shows the same asymmetry. This proposal
  is the exact "re-examine the YES-on-Challenger question on
  correctly-labeled data" path that the v2.3 BET_RULES entry teed up.

The combination of n=187 on the targeted slice, a 16pp+ overconfidence
gap, a pre-registered hypothesis from the prior week, a named mechanism,
and the fact that the audit trail for re-enablement was deliberately left
in the code for this cycle makes the n cost of being wrong genuinely
small — and the prior strong enough that I'd defend the proposal even if
the gated-cohort slice alone (n=42) were the only evidence.
