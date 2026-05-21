---
description: Run the 3-agent paper-trading review (diagnostic + design + devil's-advocate) plus a macro-gap surfacing agent, and ship a gate-version bump if approved.
---

# /auto-review

You are coordinating an autonomous review of the tennis paper-trading system. Follow the playbook at `docs/REVIEW_PROCESS.md` in this repo step-by-step.

Short version:

1. **Pull** `origin/main` (rebase if needed).
2. **Cadence + data freshness check** (BOTH must hold — else stop):
   - **At least 7 days** since the most recent dated subdir in `data/analyses/` (if none exists, fall back to `2026-05-15`). If less than 7 days have passed, stop and tell the user `"last review was X (N days ago) — auto-review is rate-limited to once per week. Run /auto-review again after <date>."`
   - **At least 20 new settled bets** with `timestamp_settled` newer than the last review date. If fewer, stop and tell the user `"week elapsed but only Y new settled bets since last review — not enough data."`
   - Only proceed when both gates pass. The user can override by saying so explicitly in the same message ("force a review anyway") — do not infer override silently.
3. **Spawn the 3 committee agents in sequence**, using `general-purpose` subagent_type:

   - **Diagnostic agent**: reads `data/paper_trades/settled.csv` + `BET_RULES.md` + `src/paper_trader.py`. Writes a structured PnL/category breakdown to `data/analyses/{YYYY-MM-DD}/diagnostic.md`. **Diagnose only — no proposals.** Tell the agent to use `kalshi_series` as the source of truth for tier, not `tourney_level` (which is a TML-mode lookup and can mislabel Challengers — see v2.3 in `BET_RULES.md`).

   - **Design agent**: reads the diagnostic + `BET_RULES.md` + the relevant source files. Proposes ONE concrete change (threshold tweak, new filter, or new feature). Writes to `data/analyses/{YYYY-MM-DD}/proposal.md`. Must include: ## Problem, ## Change, ## New version, ## Expected impact, ## Sample-size justification.

   - **Devil's-advocate agent**: reads diagnostic + proposal + `settled.csv` + `BET_RULES.md`. Critically evaluates for overfitting, lookahead, brittleness, worst case. Writes verdict (`APPROVE` / `MODIFY: <feedback>` / `REJECT: <reason>`) to `data/analyses/{YYYY-MM-DD}/critique.md`. Has hard veto power.

4. **If APPROVE:**
   - Apply the proposed change (`src/paper_trader.py` and/or `src/ml/features/feature_engineer.py`).
   - Bump `GATE_VERSION`.
   - Add a new section to `BET_RULES.md`.
   - Smoke test: `python -c "import src.paper_trader"` then `python -m src.paper_trader --scan-only`.
   - `git add` the touched code + `BET_RULES.md` + `data/analyses/{YYYY-MM-DD}/`.
   - Commit with `auto-review vX.Y: <summary>` and the diagnostic / proposal / verdict each as a sentence.
   - `git pull --rebase origin main` (CI bot races).
   - `git push origin main`.

5. **If REJECT or MODIFY-and-REJECT-again:**
   - Save the 3 analysis files anyway (they're the audit trail).
   - Commit them with `auto-review {date}: no change (rejected by devil's-advocate)`.
   - Push.

6. **Spawn the macro-gap agent** (runs regardless of the committee's verdict, AFTER the apply/archive step so it can also read the committee output). Use `general-purpose` subagent_type. Its job is to surface systemic, repo-wide patterns to the human — NOT to propose bet-rule changes and NOT to ship code. Inputs:
   - `data/paper_trades/dropped.csv` (~2000+ rows; understand which markets we forgo and why)
   - `data/paper_trades/settled.csv` (now carries `kalshi_series` per v2.3)
   - `data/analyses/{YYYY-MM-DD}/diagnostic.md` + `proposal.md` + `critique.md` (this week's committee output)
   - Recent `git log --oneline -20 -- src/` (look for fixup/revert/bugfix patterns)
   - `BET_RULES.md` (version history — has any pattern recurred?)

   Writes to `data/analyses/{YYYY-MM-DD}/macro_gaps.md` — a 300-600 word briefing aimed at the user. Sections:
   - `## PnL opportunity gaps` — drop reasons sorted by count (last 7 days AND lifetime). For the top 2–3, estimate what fraction of scanned markets we forgo and call out any reason that's growing week-over-week.
   - `## Data-source gaps` — tournaments / players appearing in Kalshi but not resolving in TML or the player resolver. Cite specific examples by name where useful.
   - `## Label / inference inconsistencies` — places where two scan-time fields disagree (e.g. `kalshi_series == "KXATPCHALLENGERMATCH"` but `tourney_level != "C"`; surface-inference returning NaN; coverage-counts of 0 when the player clearly exists). Flag anything that looks like a v2.2-style hidden bug.
   - `## Repeating bug patterns` — anything in recent git history that looks like a re-fix of the same area. If a single file has been touched by 3+ "fix" / "bug" / "revert" commits in the last month, surface it.
   - `## Suggested follow-ups` — bulleted list. Each item is one sentence and tagged `(SEV-1/2/3)`. SEV-1 = likely currently losing PnL; SEV-2 = correctness bug latent; SEV-3 = nice-to-have. NO code edits, NO concrete proposals beyond "the user should look at X". This is *surfacing*, not designing.

   The macro-gap agent must not edit code, must not bump GATE_VERSION, must not write to `BET_RULES.md`, must not commit. It only writes `macro_gaps.md`.

   After it finishes, append a separate commit `auto-review {date}: macro-gap briefing` containing only `data/analyses/{YYYY-MM-DD}/macro_gaps.md`, then push.

7. **Report to user** (8–12 lines):
   - Days since last review + # new bets
   - Diagnostic finding (1 sentence)
   - Design proposal (1 sentence)
   - Devil's-advocate verdict (1 sentence)
   - What was pushed (commit hashes) OR "no change this week"
   - **Macro-gap top surfacings**: the SEV-1 items from the macro-gap agent verbatim (1–3 lines)
   - Pointer to `data/analyses/{YYYY-MM-DD}/` for the full audit trail (diagnostic / proposal / critique / macro_gaps).

## Hard constraints

- NEVER change `MIN_EDGE` or `MAX_SPREAD` (safety constants).
- NEVER retrain the model in this workflow.
- NEVER force-push or amend bot commits.
- NEVER apply a change without devil's-advocate `APPROVE`.
- NEVER bypass the 7-day cadence gate unless the user explicitly types an override in the same message.
- The macro-gap agent NEVER ships code, edits gate logic, or bumps the version. It surfaces only.
- If smoke test fails after applying, revert the working tree and report to user — do not commit a broken change.

For the full reasoning behind each phase, read `docs/REVIEW_PROCESS.md`.
