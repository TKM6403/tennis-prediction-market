# Auto-Review Process

**Trigger:** in any Claude Code session, ask "run auto-review" (or `/auto-review` if the
slash command is wired up). Works from laptop or phone via claude.ai/code as long as
the session has access to this repo.

**Goal:** every ~week, run a 3-agent committee that analyzes paper-trading data and
proposes ONE concrete change to the bet rules. A devil's-advocate agent gates the
proposal. If approved, the change is pushed directly to main.

---

## Pipeline overview

```
[1] git pull → check data freshness        ───╮
[2] Diagnostic agent (writes diagnostic.md)   │
                ↓                              │
[3] Design agent (writes proposal.md)         │── all saved to
                ↓                              │   data/analyses/YYYY-MM-DD/
[4] Devil's-advocate agent (writes critique)  │
                ↓                              │
        APPROVE? ─── yes → [5] apply + push   │
                  └── no  → archive, no push  │
[6] Brief report to user                    ──╯
```

---

## Phase 1: Data freshness check

Before spawning agents, the orchestrator should:

1. `git pull --rebase origin main`
2. Read `data/paper_trades/settled.csv`
3. Find the date of the last review (highest dated subdir in `data/analyses/`).
   If no prior reviews, set last_review to `2026-05-15` (when v2.1 shipped).
4. Count settled bets with `timestamp_settled` > last_review.
5. **If new bets < 20:** stop. Tell user "not enough new data — last review was X,
   only Y new settled bets since." Do not spawn agents.

This prevents over-fitting to noise from a tiny week.

---

## Phase 2: Diagnostic agent

**Spawn type:** `general-purpose`, in background or foreground.

**Inputs to give the agent in its prompt:**
- Path to `data/paper_trades/settled.csv`
- Path to `BET_RULES.md` (for version history & current filters)
- Path to `src/paper_trader.py` (current gate code)
- Path to `src/ml/train.py` (AUGMENTED_FEATURES list)

**The agent's job:**
- Compute PnL / ROI / win rate broken down by:
  - `gate_version` (v1.0 vs v2.0 vs v2.1 vs newer)
  - surface, tourney_level, theo bucket, edge bucket, direction (YES/NO)
  - tournament (n >= 5 only)
- For each slice, flag overconfidence (CI on win rate vs avg `theo_chosen`)
- Compare current-period numbers to the prior diagnostic in `data/analyses/`
  if available
- Write a structured ~500-word report to
  `data/analyses/YYYY-MM-DD/diagnostic.md`

**Constraints:**
- Only diagnose. Do not propose changes. Do not edit code.

---

## Phase 3: Design agent

**Inputs:**
- `data/analyses/YYYY-MM-DD/diagnostic.md` (just written)
- `BET_RULES.md`
- `src/paper_trader.py`
- `src/ml/features/feature_engineer.py`

**The agent's job:**
- Propose **at most ONE** change addressing the most concrete failure mode in
  the diagnostic.
- Write the proposal to `data/analyses/YYYY-MM-DD/proposal.md`.

**Allowed change types:**
- Threshold tweak on an existing constant
  (e.g., `MIN_PLAYER_COVERAGE 15 → 12`)
- New filter (a new drop reason in `_score_match_from_features`)
- New feature (added to `AUGMENTED_FEATURES` and `_assemble_features`,
  but ONLY if the agent also writes the computation logic in
  `feature_engineer.py` AND notes that retraining is required separately)

**Disallowed (hard-coded refusals):**
- Changing `MIN_EDGE` or `MAX_SPREAD` (safety constants, not bet-quality knobs)
- Retraining the model in this workflow (separate higher-risk task)
- Deleting any data
- Force-push or amending bot commits
- Bumping multiple version digits in one change (one minor or one major, not both)

**Proposal format (required sections):**
- "## Problem" — which diagnostic finding this addresses
- "## Change" — exact constant/code change
- "## New version" — proposed `GATE_VERSION` (e.g., v2.1 → v2.2 for filters/thresholds,
  v2.x → v3.0 for new features)
- "## Expected impact" — which failure mode, expected ROI delta with bounds
- "## Sample-size justification" — n in the relevant slice, why this isn't overfit

---

## Phase 4: Devil's-advocate agent

**Inputs:**
- `data/analyses/YYYY-MM-DD/diagnostic.md`
- `data/analyses/YYYY-MM-DD/proposal.md`
- `data/paper_trades/settled.csv`
- `BET_RULES.md` (to check version history for similar past changes)

**The agent's job:** critically evaluate the proposal. Specifically check:

1. **Overfitting**: is the sample size in the targeted slice adequate?
   - Rule of thumb: if the slice the proposal optimizes against has fewer than
     30 settled bets, lean toward REJECT.
   - If the proposal cites a single-day blow-up as motivation, lean toward REJECT.
2. **Lookahead / data leakage**: does the proposed feature or threshold use
   information not available at scan time?
3. **Brittleness**: would this change work as well 3 months from now, or is it
   chasing a noise spike? Check whether the failing slice has been failing
   consistently across prior diagnostics.
4. **Worst case**: if this is wrong, how bad is the bleed?

**Verdict (write to `data/analyses/YYYY-MM-DD/critique.md`):**
- `APPROVE` — change is sound, ship it
- `MODIFY: <specific feedback>` — let the design agent re-run with the feedback (max one re-roll)
- `REJECT: <reason>` — archive, no change this week

**Critical rule:** the devil's advocate has veto power. Never apply a change
without an `APPROVE` verdict.

---

## Phase 5: Apply the change (only if APPROVED)

The orchestrator (not a sub-agent) does the actual edits:

1. Edit `src/paper_trader.py` (or `src/ml/features/feature_engineer.py`) per
   the proposal.
2. Bump `GATE_VERSION` constant in `src/paper_trader.py`.
3. Add a new section to `BET_RULES.md` documenting the change. Format matches
   existing entries (cutoff timestamp, commit, what it does, why).
4. Smoke test:
   - `python -c "import src.paper_trader"` (no import errors)
   - `python -m src.paper_trader --scan-only` (no crash on a live run)
5. Stage everything:
   - `git add src/paper_trader.py BET_RULES.md data/analyses/YYYY-MM-DD/`
   - Plus any other files the change touched
6. Commit:
   ```
   auto-review vX.Y: <one-line change summary>

   <diagnostic finding in 1-2 sentences>
   <proposal change in 1-2 sentences>
   <devil's-advocate APPROVE rationale in 1 line>
   ```
7. `git pull --rebase origin main` (in case the CI bot pushed during the review)
8. `git push origin main`

---

## Phase 6: Final report to user

Brief (~6-10 lines):
- Last review date and # new bets since
- Diagnostic finding (1 sentence)
- Design proposal (1 sentence)
- Devil's-advocate verdict and reasoning (1 sentence)
- What was pushed (commit hash + `BET_RULES.md` link) OR "no change this week"
- Pointer to `data/analyses/YYYY-MM-DD/` for full audit trail

---

## Failure modes & what to do

- **Smoke test fails after applying change:** revert the working tree edits,
  do NOT commit. Report to user: "proposal X passed devil's advocate but
  failed smoke test — needs human review."
- **Devil's-advocate rejects twice in a row** (after one MODIFY iteration):
  archive both proposals, no change. Tell user explicitly so they can decide.
- **Git push race with the CI bot:** the rebase in step 7 handles it. If
  rebase has conflicts on `data/paper_trades/*`, `git checkout --theirs` and
  let our CI regenerate.

---

## Audit trail

Every review writes 3 markdown files to `data/analyses/YYYY-MM-DD/`:

```
data/analyses/2026-05-24/
  ├─ diagnostic.md   ← what's happening
  ├─ proposal.md     ← what design agent proposed
  └─ critique.md     ← what devil's-advocate said + verdict
```

These are tracked in git. The orchestrator should `git add` them in Phase 5
(if approved) or as a separate commit (if rejected — even rejected analyses
are kept for the audit trail).

---

## Honest caveats

- The diagnostic agent will tend to find "patterns" even in noise. Devil's
  advocate is the last line of defense; if it gets sloppy, bad changes ship.
- The first 3-4 auto-reviews you should still spot-check the resulting
  `BET_RULES.md` entries before trusting the chain.
- The agent committee has no awareness of in-flight matches. If a tournament
  is just starting and our v2.1 model is hot for that field, an auto-review
  mid-week could ship a change that worsens it. Lean toward weekly cadence,
  not daily.
