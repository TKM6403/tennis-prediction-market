---
description: Run the 3-agent paper-trading review (diagnostic + design + devil's-advocate) and ship a gate-version bump if approved.
---

# /auto-review

You are coordinating an autonomous review of the tennis paper-trading system. Follow the playbook at `docs/REVIEW_PROCESS.md` in this repo step-by-step.

Short version:

1. **Pull** `origin/main` (rebase if needed).
2. **Data freshness check.** Find the highest dated subdir in `data/analyses/` (the last review date — assume `2026-05-15` if none). Count settled bets with `timestamp_settled` newer than that. If fewer than 20, stop and tell the user "not enough new data, last review was X, only Y new bets."
3. **Spawn 3 agents in sequence**, using `general-purpose` subagent_type:

   - **Diagnostic agent**: reads `data/paper_trades/settled.csv` + `BET_RULES.md` + `src/paper_trader.py`. Writes a structured PnL/category breakdown to `data/analyses/{YYYY-MM-DD}/diagnostic.md`. **Diagnose only — no proposals.**

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

6. **Report to user** (6-10 lines): days since last review, # new bets, diagnostic finding, proposal, verdict, commit hash if any, link to `data/analyses/{YYYY-MM-DD}/`.

## Hard constraints

- NEVER change `MIN_EDGE` or `MAX_SPREAD` (safety constants).
- NEVER retrain the model in this workflow.
- NEVER force-push or amend bot commits.
- NEVER apply a change without devil's-advocate `APPROVE`.
- If smoke test fails after applying, revert the working tree and report to user — do not commit a broken change.

For the full reasoning behind each phase, read `docs/REVIEW_PROCESS.md`.
