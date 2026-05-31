# Model-Research Agent — Charter

**Status:** DRAFT charter for discussion. No code yet. This doc defines what
the agent is *for* and the rules it must obey, the same way
`REVIEW_PROCESS.md` governs the weekly bet-rule committee.

---

## Why this exists

We run two feedback loops over the trading system, and they have different
jobs:

| Loop | Owns | Cadence | Touches the model? |
|---|---|---|---|
| **Execution-review committee** (`REVIEW_PROCESS.md`) | bet-selection rules — thresholds, filters, drop guards | weekly | **No** |
| **Model-research agent** (this doc) | the Theo itself — features, data sources, model class, calibration | weekly, synced to the committee | **Yes (proposes only)** |

The two loops are **phase-locked to the weekly review**. Each week the research
agent ingests the committee's fresh output (diagnostic / proposal / critique /
macro-gaps), forms a model hypothesis, and stands up a **challenger model that
shadow-trades for the coming week**. The *next* weekly review is the evaluation
checkpoint: it scores the challenger against the live champion on a full week of
forward, out-of-sample data and decides whether to recommend promotion.

The committee can only ever *paper over* a bad model. The canonical example is
the v2.4 → v2.5 episode in `BET_RULES.md`: YES-on-Challenger bets lost money
because the model **overrates favorites on thin Challenger fields**. The
committee's answer was to ban those bets; the human reverted it because that
masks the symptom instead of fixing it. **Fixing it is this agent's entire
job.** The committee tunes *where and whether* to bet; the research agent tunes
*what the model believes*.

---

## Prime directive: no lookahead, ever

This is the headline rule and it overrides everything else, including the
agent's freedom to experiment. It is the existing repo rule (`CLAUDE.md` Hard
Rule #1) extended to cover **any** data source the agent brings in:

> Every feature the agent proposes must be reconstructable using only
> information that provably existed **strictly before the match started**, with
> a stored timestamp to prove it. Strict `<` on the relevant cutoff. No
> exceptions, no "it's probably fine."

Because the agent is allowed **open season on free data sources** (TML, Kalshi
order book / line movement / volume, Twitter/X sentiment, news, anything free),
this rule does the heavy lifting. Exotic feeds are exactly where leakage hides:

- A tweet's *content* timestamp is not its *ingestion* timestamp — the feature
  must use only tweets posted before `match_start_time`.
- A "closing line move" is only usable up to our scan time, not the true close.
- Any scraped value must be snapshotted point-in-time, not re-fetched live at
  backtest time (which would silently pull post-match state).

If a feature cannot demonstrate point-in-time integrity, it does not ship —
regardless of how good its backtest looks.

### The devil's-advocate has a hard leakage veto

Every research proposal is routed through the **devil's-advocate agent** (the
same committee role from `REVIEW_PROCESS.md`, or a leakage-specialized variant)
**specifically to audit for lookahead/leakage**, before the proposal ever
reaches the human. Its job here is narrow and absolute:

- For each new feature/data source, demand the point-in-time reconstruction
  story and the timestamp that backs it.
- Probe for subtle leakage: target leakage, train/test contamination across the
  time split, live-refetch of scraped data, survivorship in the scraped set.
- Verdict `APPROVE` / `MODIFY` / `REJECT` with **veto power**. A proposal the
  devil's-advocate rejects on leakage grounds is dead — it does not go to the
  human as a recommendation.

This is the safety valve that makes "open season on data" survivable.

---

## Objective function: calibration-first

What the agent is allowed to call "better":

1. **Primary: probabilistic calibration loss** — log-loss (the metric
   `train.py` already selects on), plus ECE and Brier on a held-out,
   out-of-time period. See `LITERATURE_REVIEW.md` (Walsh & Joshi 2024: picking
   models by calibration made +34.7%, by accuracy lost 35.2%).
2. **Secondary, sanity-check only: backtest ROI** under the *current* bet rules.
   ROI is reported with bounds but is never the primary target — optimizing it
   directly is the easiest way to overfit and to reward leakage.
3. Never accuracy. Never in-sample fit.

A candidate model must beat the incumbent on the primary calibration metric on
an **out-of-time** validation window (walk-forward, mimicking production) before
ROI is even discussed.

---

## Autonomy: propose-only

The agent **never** ships a model change. It:

- Works in an **isolated sandbox** (its own git worktree/branch). It must
  **never overwrite the live model pickle** (`data/processed/model_augmented_beta.pkl`)
  or touch `src/paper_trader.py`'s bet-selection logic.
- Experiments freely *inside the sandbox*: add features, pull data, swap the
  model class entirely (logistic regression → gradient boosting → calibrated
  ensemble → whatever it can justify). The architecture is not sacred; the
  objective and the no-lookahead rule are.
- Returns a **written proposal + evidence** to the human. The human decides
  whether to retrain and ship. (This mirrors the v2.5 human-gate the user just
  exercised.)

---

## A/B testing: champion vs challenger (shadow)

The agent validates a model change by **shadow A/B test against the live model
over one weekly cycle**, never by backtest alone.

- **Champion (A):** the current live model. It alone places the real paper bets.
- **Challenger (B):** the candidate the research agent built this week. It runs
  in **shadow mode** — on every scan, the scan path also loads the active
  challenger pickle, computes its Theo on the *same markets at the same
  timestamps* the champion saw, and logs what it *would* have bet. **The
  challenger never places a real bet** and never affects the live system.
- After a full week, the **next weekly review** scores champion vs challenger
  on the same matches: calibration-first (log-loss / ECE / Brier on resolved
  outcomes), then would-be ROI as a secondary check.

Why shadow A/B is the linchpin, not a nice-to-have:

- **It is the strongest no-lookahead guarantee we have.** The challenger commits
  its prediction *before* the match resolves and that prediction is logged
  immutably. A leaky feature that looked great in backtest will fall apart in
  the forward shadow test, because there is no future to peek at. This is the
  prime directive enforced by reality, not by audit.
- **It matches the real objective.** Same markets, same scan times, same
  liquidity the champion actually faced — no idealized backtest fills.

Champion is only swapped for a challenger by **explicit human approval** after
the challenger wins its shadow week. Promotion is never automatic.

Default to **one active challenger per week** so its result is cleanly
attributable; the sandbox may explore many candidates but only the single best
is promoted to the live shadow slot. (Multiple parallel challengers is a
possible later extension.)

---

## The research loop

```
[1] Problem intake   ← THIS week's committee output (diagnostic.md,
                       proposal.md, critique.md, macro_gaps.md) + live betting
                       results. Pick ONE failure rooted in the model.
        ↓
[2] Hypothesis register  ← write the hypothesis + research rationale/citation
                            to the audit trail BEFORE experimenting
                            (pre-registration; guards against p-hacking).
        ↓
[3] Experiment (sandbox) ← new features / new data / new model class, capped at
                            the per-cycle iteration budget (below). New features
                            follow CLAUDE.md #2 (spec docstring + dummy-data
                            __main__). Uses the single date-based split in
                            train.py — never peeks at the test window.
        ↓
[4] Backtest screen  ← calibration-first metrics on an out-of-time window;
                       walk-forward. ROI as secondary sanity check. Best single
                       candidate advances.
        ↓
[5] Leakage audit    ← devil's-advocate hard veto on lookahead before deploy.
        ↓
[6] Shadow deploy    ← promote the survivor to the live shadow slot; it
                       shadow-trades alongside the champion for one week.
        ↓
[7] Next weekly review ← score challenger vs champion on the forward week.
        ↓
[8] Proposal to human ← hypothesis, what changed, forward calibration delta +
                        ROI delta with bounds, leakage verdict, new
                        dependencies, retrain/promote instructions, rollback.
                        Human decides whether to promote the challenger.
```

---

## Storage & audit trail

- **Candidate pickles are stored locally and gitignored** (e.g.
  `data/research/candidates/<challenger_id>.pkl`), alongside the live
  `data/processed/model_augmented_beta.pkl` which the agent never overwrites.
  Pickles are data; per `CLAUDE.md` #4 they are never committed.
- **The change history IS committed** — a tracked ledger under `data/research/`
  (analogous to `data/analyses/`) recording, for every challenger: the
  hypothesis + citation, what changed (features / data / model class), the
  backtest screen, the leakage verdict, the forward shadow A/B result, and the
  human's promote/reject decision. This is the permanent record of
  what-was-tried-and-why, so dead ends aren't re-explored and promotions are
  auditable.
- **Shadow prediction logs** (the challenger's would-be bets) live under
  `data/research/shadow/<challenger_id>.csv` — gitignored like the other
  paper-trade CSVs; only the weekly scored summary lands in the tracked ledger.

## Iteration budget

A reasonable per-cycle cap to start (tune later): the sandbox explores **up to
~15 candidate experiments per weekly cycle**, of which **exactly one** is
promoted to the live shadow slot. The cap exists so a research cycle is bounded
and cheap; raise it once the loop has proven its leakage discipline.

---

## Hard constraints (non-negotiable)

- **No lookahead, ever** — the prime directive above. Devil's-advocate veto.
- **Propose-only** — never writes the live model pickle, never auto-retrains
  into production, never ships without human approval.
- **Stays in its lane** — does not touch `MIN_EDGE`, `MAX_SPREAD`, or any
  bet-selection rule. Those belong to the committee. It only changes the model
  / features / data.
- **New features follow `CLAUDE.md` #2** — spec docstring (what/why/inputs/
  outputs/dummy example) + runnable `__main__` block, before touching real data.
- **New dependencies are surfaced, not silently added** (`CLAUDE.md` #3). Open
  season on data implies new deps (scrapers, NLP) — each one is named in the
  proposal with what it does and why stdlib/existing deps can't, for the human
  to approve.
- **No data committed** (`CLAUDE.md` #4). Scraped corpora, snapshots, and
  feature caches stay gitignored.
- **One split, in `train.py`** (`CLAUDE.md` #5). The time-based train/val/test
  convention is not duplicated or worked around.
- **Retrain is always a separate, human-approved step.** The agent may produce a
  candidate artifact in its sandbox; promoting it to live is the human's call.

---

## Relationship to the committee

- **Shared feedback signal:** both loops read the same diagnostic / PnL /
  macro-gap output.
- **Shared devil's-advocate:** the committee uses it to gate bet-rule
  overfitting; the research loop uses it to gate model leakage.
- **Clean lane split:** committee = execution layer (weekly, bet rules);
  research agent = model layer (slower, the Theo). Neither writes the other's
  artifacts.

---

## Resolved decisions

1. **Trigger/cadence:** phase-locked to the weekly review. Each cycle ingests
   the committee's output, deploys one shadow challenger, and is scored by the
   next review (the A/B section above).
2. **Validation:** forward shadow A/B (champion vs challenger), not backtest
   alone. Backtest is only the in-sandbox screening step.
3. **Pickle storage:** local, gitignored candidate pickles; live model never
   overwritten. Tracked change-history ledger under `data/research/`.
4. **Experiment budget:** ~15 candidate experiments per cycle, one promoted to
   the shadow slot. Tunable.
5. **Audit trail:** tracked ledger in `data/research/` recording hypothesis,
   change, backtest, leakage verdict, forward A/B result, and human decision.

## Build status

- **Shadow plumbing — IMPLEMENTED.** Hooked directly into the live scan/settle
  path in `src/paper_trader.py` (safe: it's all paper money, no production
  risk). When `data/research/active_challenger.json` names a challenger, every
  scan re-scores the same markets/timestamps/features with the challenger model
  and logs would-be bets to a gitignored `data/research/shadow/<id>.csv`; the
  settle pass resolves them with the same logic the champion uses. Absent/null
  registry → complete no-op (champion logs byte-for-byte unchanged). The
  challenger runs the identical bet gates — only the model differs.

## Still open (decide before first real cycle)

- **Cold start:** the very first cycle has no challenger that's been shadowing
  yet, so it can only *deploy* one, not *evaluate* one. Default plan: seed the
  first challenger with the **surface-Elo feature** already on the planned-next
  list in `BET_RULES.md`, to exercise the loop end-to-end on a known, wanted
  experiment rather than something free-styled. (Change this if you'd rather the
  agent pick its own first hypothesis off the diagnostic.)
- **The research agent itself** is not built yet — only the shadow *plumbing*
  it will use. The agent (hypothesis register, sandbox training, the weekly
  scoring/proposal step) is the next thing to build on top of this.
```
