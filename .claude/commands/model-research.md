# /model-research

You are coordinating one weekly cycle of the **model-research agent** — the
propose-only loop that tries to improve the Theo model itself, as opposed to the
bet-selection rules (those belong to `/auto-review`). Read
`docs/MODEL_RESEARCH_AGENT.md` first; it is the charter and these instructions
implement it. The two loops are phase-locked: run this AFTER the week's
`/auto-review` committee, so this cycle can read its output.

## What this loop does, in one breath

Ingest the committee's output → (A) score last week's shadow challenger against
the champion and decide its fate → (B) form ONE new model hypothesis → train a
candidate → leakage-veto it → deploy the survivor as the new shadow challenger →
report to the user. It never ships a model to production; promotion is a
separate human-approved step.

## Hard constraints (read before anything)

- **Propose-only.** NEVER overwrite the live champion pickle
  (`data/processed/model_augmented_beta.pkl`). NEVER promote a challenger to
  champion — that is a separate, explicit human decision outside this loop.
- **Calibration-first.** Select and judge candidates on log-loss / ECE on the
  out-of-time split and the forward shadow week — NEVER accuracy or in-sample
  fit. (LITERATURE_REVIEW.md, Walsh & Joshi 2024.)
- **No lookahead, ever.** The devil's-advocate leakage critique has a HARD VETO.
  NEVER deploy a candidate to the shadow slot without an `APPROVE`.
- **Stay in lane.** NEVER touch `MIN_EDGE`, `MAX_SPREAD`, the bet rules, or
  `GATE_VERSION` — that is `/auto-review`'s job. This loop only changes the model.
- **One challenger at a time** in the live shadow slot, for clean attribution.
- **v1 candidate surface.** Candidates vary model class / calibration method /
  regularization / feature-SUBSET over the existing 15 `AUGMENTED_FEATURES` —
  these are shadow-testable today. A genuinely NEW feature (e.g. surface-Elo)
  also needs inference-side plumbing in `matches_to_feature_matrix` + the shadow
  path before the loop can test it; treat that as a separate build, not an
  in-loop experiment.
- **The agent surfaces a candidate; it does not retrain the champion.** Promoting
  a winning challenger (rebuilding the live pickle) is human-approved follow-up.

## Phase 0 — Pull + intake

1. `git pull --rebase origin main` (stash local notebook edits first if it blocks).
2. Set `DATE` = today (`YYYY-MM-DD`). `mkdir -p data/research/$DATE`.
3. Read the latest committee output: the highest dated `data/analyses/<d>/`
   (`diagnostic.md`, `proposal.md`, `critique.md`, `macro_gaps.md`).
4. Read `data/research/LEDGER.md` if it exists (history of past challengers).

## Phase 1 — Evaluate the active shadow challenger (the A/B checkpoint)

Only if `data/research/active_challenger.json` names a challenger AND its shadow
log has settled rows.

1. Read `active_challenger.json` → `<cid>`. Read
   `data/research/shadow/<cid>_settled.csv` (the challenger's resolved would-be
   bets) and the champion's `data/paper_trades/settled.csv` over the same
   forward window (since `registered_at`).
2. Score **calibration-first**: compute log-loss and ECE of challenger
   `theo_chosen` vs realized `bet_won` on the shadow bets, and the same for the
   champion on its bets over that window. Report would-be ROI as a secondary
   sanity check only.
3. Write `data/research/$DATE/evaluation.md`: did the challenger beat the champion
   on the FORWARD week (calibration first)? Verdict is one of:
   - `PROMOTE-CANDIDATE` — challenger clearly better-calibrated forward; recommend
     to the human that they retrain/swap the champion (NOT done here).
   - `KEEP-SHADOWING` — promising but n too small / mixed; leave it in the slot.
   - `RETIRE` — no better (or worse); free the slot for a new challenger.
4. Append the outcome to `data/research/LEDGER.md`.

If there is no active challenger, note that and continue.

## Phase 2 — Form ONE hypothesis (research/design subagent)

Spawn a `general-purpose` subagent. Inputs: the committee output (esp. the
diagnostic's losing slices + `macro_gaps.md`), `LITERATURE_REVIEW.md`,
`src/ml/train.py` (the 15 `AUGMENTED_FEATURES`, the calibration choices), and the
ledger. Its job:

- Propose **exactly ONE** model hypothesis within the v1 candidate surface
  (model class / calibration / regularization / feature-subset). Research-backed:
  cite a paper or a concrete diagnostic finding, not a vibe.
- Write `data/research/$DATE/hypothesis.md` with: `## Problem` (which model
  failure, tied to the diagnostic), `## Hypothesis` (what change + why it should
  improve calibration), `## Candidate spec` (exact `train_candidate` args:
  `--model`, `--cal`, `--C`/hyperparams, `--features`), `## Why no lookahead`
  (point-in-time story for every input it uses).

Constraint to the subagent: propose only; do not train or edit code.

## Phase 3 — Train the candidate (orchestrator)

Run the harness with the spec from `hypothesis.md`:

```
python -m src.ml.research.train_candidate --id <DATE>_<short> \
    --model <m> --cal <c> [--C <C>] [--features ...] \
    --hypothesis "<one-line>"
```

(Use a fresh venv if the local interpreter lacks deps — see how `/auto-review`
smoke-tests.) This writes the candidate pickle + `<id>.metrics.json` (both
gitignored). Read the metrics: this is the **backtest screen**. If the candidate
is clearly WORSE than the champion on out-of-time log-loss AND ECE, do not
deploy — note it in the ledger and stop this cycle (or, once, ask Phase 2 to
re-roll). A neutral/better backtest advances to Phase 4 — the forward shadow week
is the real judge.

## Phase 4 — Leakage critique (devil's-advocate, HARD VETO)

Spawn a `general-purpose` subagent as the leakage devil's-advocate. Inputs:
`hypothesis.md`, the candidate spec + `metrics.json`, `src/ml/train.py` +
`src/ml/features/feature_engineer.py` (how each feature is computed), and
`docs/MODEL_RESEARCH_AGENT.md` (the prime directive). Its job: hunt for
lookahead / leakage / train-test contamination / a too-good backtest that smells
like leakage. Verdict (write `data/research/$DATE/leakage_critique.md`, first
line exactly one of):

- `APPROVE`
- `MODIFY: <feedback>` (allow ONE Phase 2→3 re-roll)
- `REJECT: <reason>`

NEVER proceed to Phase 5 without `APPROVE`.

## Phase 5 — Deploy to the shadow slot (only on APPROVE)

1. `python -m src.ml.research.train_candidate ... --register` (or call
   `register_challenger(id, hypothesis)`) to write `active_challenger.json`. The
   challenger now shadow-trades for the coming week; the next `/model-research`
   cycle scores it (Phase 1).
2. Append the new challenger to `data/research/LEDGER.md` (hypothesis, spec,
   backtest screen, leakage verdict, deployed-on date).
3. `git add data/research/$DATE/ data/research/active_challenger.json
   data/research/LEDGER.md` (NOT the pickle — gitignored). Commit:
   `model-research $DATE: shadow <id> (<one-line hypothesis>)`. Pull-rebase, push.

If `REJECT`, still commit the audit files (`hypothesis.md`,
`leakage_critique.md`, `evaluation.md`) + ledger note as
`model-research $DATE: no deploy (leakage veto)`. Push.

## Phase 6 — Report to user (8–12 lines)

- Prior challenger A/B verdict (Phase 1), with the forward calibration delta.
- New hypothesis in one sentence.
- Backtest screen: candidate vs champion out-of-time log-loss / ECE.
- Leakage verdict.
- What is now shadowing (or "slot empty — rejected").
- Reminder that promoting a winner to champion is a separate human-approved step.
- Pointer to `data/research/$DATE/` and `data/research/LEDGER.md`.

## Notes

- Cold start (first ever cycle): there is no prior challenger to evaluate
  (skip Phase 1). For the first hypothesis, prefer a known same-feature variant
  (e.g. a calibration-method or regularization A/B) so the loop is exercised on
  something immediately shadow-testable; surface-Elo and other NEW features wait
  on inference-side plumbing.
- The shadow plumbing is in `src/paper_trader.py` (champion path is unaffected
  when no challenger is registered). The harness is
  `src/ml/research/train_candidate.py`.
