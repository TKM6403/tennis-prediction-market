# Devil's-Advocate critique — 2026-05-31

APPROVE

## What I verified (pandas, settled.csv n=240)

**Series field is clean and canonical.** All 240 settled rows are
`kalshi_series == "KXATPCHALLENGERMATCH"`. The proposal gates on this field,
which is populated at scan time from the `market_id` prefix via
`_series_from_market_id()` — no TML name-mode lookup, no `tourney_level`
coupling, no lookahead. This is the *exact* defect that killed v2.2: v2.2 gated
on `tourney_level == "C"`, which the TML mode mislabeled "250" for Cordoba
(name collision with the Argentina ATP-250). The v2.4 patch does not touch
`tourney_level` at all. The leakage objection that justified the v2.2 revert is
genuinely closed. **No-lookahead: confirmed** — direction and series are both
known at scan time.

**Direction split reproduces exactly.** YES n=200, win 0.290, avg_theo 0.528,
ROI −25.2%, Wilson-95 [0.232, 0.356] — entirely below theo. NO n=40, win 0.525,
avg_theo 0.517, ROI +52.6%, Wilson-95 [0.375, 0.671].

**Is NO just lucky? No.** This was my primary suspicion and it failed.
- Equal-weighted mean per-bet ROI on NO is **+61.3%** (median **+57.6%**) —
  *higher* than the dollar-weighted +52.6%, so the headline is not inflated by a
  few big-stake wins.
- Leave-one-out: dropping the top 1/3/5 NO winners leaves +46.8% / +36.0% /
  +25.8% (n=35). The top 5 are 55% of gross profit but the residual 35 bets are
  still strongly positive. Not a handful of correlated wins.
- NO calibration is honest: win-rate 0.525 ≈ avg_theo 0.517 (sits *inside* its
  Wilson interval). The NO book is well-calibrated; n=40 clears the 30 floor.
- No single tournament/week carries it: NO spreads across ~14 tournaments and
  4 settle-weeks, the biggest single-tournament contributor (Bordeaux n=7) is
  itself +62%.

**Is YES a tail artifact? No — it's structural.** YES equal-weighted median
per-bet ROI is **−104%** (the median YES bet is a total loss). The bleed is the
body of the distribution, not the tail. This is the strongest reason to act:
we are removing a cohort whose *typical* outcome is a wipeout, not one with a
fat left tail.

## The real objection, and why it does not block

This is v2.2 resurrected, and a blanket "never bet YES on Challenger" patches a
*model* calibration artifact with a direction ban rather than fixing the model
(the planned Elo/line-move retrain). v2.4 drops ~80% of flow (v2.3 flow is 92%
YES) and turns the system into a NO-only Challenger bot on n=40 of evidence.
That is real curve-fitting-to-the-sign-of-PnL risk and I weighed rejecting on it.

It does not block for three reasons. (1) The v2.2 revert rationale in BET_RULES
is explicit: "reverted not because the signal was wrong, but because the gating
field was wrong," and v2.3 was shipped *specifically* to hand the next reviewer
correctly-labeled data to re-enable this. v2.4 is that intended re-enable. (2)
The signal is the best-sampled, most persistent finding in the book — same sign
and magnitude across four weekly cuts (2026-05-20 YES −29.2%/NO +47.5%; today
−25.2%/+52.6%). (3) Worst-case opportunity cost is bounded: even a 20pp reversal
of the dropped YES slice only zeroes it, and the kept NO book is independently
calibrated, so v2.4 does not leave the bot betting into an equally-bad NO book.

Betting less is not a cost when the dropped cohort's median outcome is a total
loss. Ship it — but the design agent's expected-ROI bound [0%, +50%] is the
honest framing; do not market this as "+52%". Flag for next review: NO volume is
thin (n=40) and the proper fix remains the model retrain, not a permanent ban.
