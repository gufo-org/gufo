> Instructions given to the integration agent (written before it ran). A brief, not
> a result.

# Integration brief — Card 09 third-agent integration

Repository: `/home/mixer/gufo` (jj). You are the **third distinct agent** for Card 09:
you resolve the independent review, re-run only what your changes affect, audit
acceptance, and draft the execution record. You do not push and you **never** run
`jj commit`/`jj new`/`jj squash` — the orchestrator commits after its own pinned
re-run of the full gate set.

## Environment reality

- `nix develop -c <cmd>` works from `/home/mixer/gufo`; builds reuse `build/card09-gpu`
  (gpu-test), `build/card09-cpu` (cpu-test), `build/card09-release` (discovery),
  `build/card09-lock` (lock mode). Scratch and logs go in `/tmp/card09-integration/`.
- If a Nix invocation fails with a daemon/socket permission error, **do not retry it
  repeatedly and do not widen access**. Record the exact command, the denial text, and
  hand it to the orchestrator as `PENDING(orchestrator)`. Static python checks
  (`tools/ci/check-extension-isolation.py`, `check-extension-api.py`, `check-docs.py`)
  normally run with the plain host interpreter — label those as host-interpreter runs.
- No functional harness, no servers, no speed benchmarks, no main-control timings
  (user decision 2026-10-31: deferred to the single post-Card-14 qualification window).
- Keep tool output small: redirect logs, print `tail`/`grep` verdict lines only.

## Inputs

1. `docs/model-extensions/evidence/card09/RESUME-NOTES.md` and the rest of that
   directory (worker evidence, `input-brief-worker.md`).
2. `docs/model-extensions/cards/09-serving-runtime-boundary.md` + its DoD and
   validation criteria; `docs/model-extensions/CARDS.md` shared gates;
   `docs/model-extensions/cards/README.md` conventions; `AGENTS.md`.
3. The review findings (handed to you in your task text; also the reviewer's final
   message). Treat the list as the work queue.

## Your job, in order

1. **Every finding gets a disposition**: `RESOLVED` (what you changed, which command
   proves it), `REJECTED` (concrete evidence why it is wrong — file/line/command output,
   never an opinion), or `PENDING` (with the exact command and who must run it). No
   finding may be left unaddressed or silently downgraded.
2. Fix the code. Keep it a behavior-preserving refactor; do not weaken a gate, do not
   delete a test to get green, do not mark a deferred functional requirement as done.
3. **Re-run each affected gate exactly once** — the gates whose subject your edits
   touched, plus anything the reviewer explicitly asked for that the worker never ran.
   Record command, exit code, and the verdict lines. Do not re-run the whole pinned
   set: the orchestrator does that once after your last edit. If you change a gate's
   own logic, re-prove its negative fixture and the pinned modes that consume it.
4. **Acceptance audit even if the review found nothing**: prove or explicitly declare
   unproven each of the three DoD items, each plan item (serving extraction; runtime
   neutral contracts; budgets/backing/factory placement; policy stays + consumers
   migrated + focused tests), and each shared acceptance gate that is in-card scope
   (flag/assertion/sanitizer parity of moved sources, discovery vs lock artifacts,
   isolation ratchet, hosted checks, formatting, extension-API SURFACE stability).
   Name the evidence path for each; "the build succeeded" is not proof of a behavior claim.
5. Copy the reviewer's findings file into
   `docs/model-extensions/evidence/card09/review-findings.md` (small, curated — no raw
   logs) and write `docs/model-extensions/evidence/card09/integration-reruns.txt` with
   the exact commands, exits, verdicts and denials.
6. Draft the **execution record** inside
   `docs/model-extensions/cards/09-serving-runtime-boundary.md` (replace the
   "Pending." paragraph): the three agent names + run IDs, the finding/disposition
   table, exact validation results, evidence links, and an explicit remaining-limitations
   paragraph. Update the DoD checkboxes **only** for what is actually proven — a
   deferred or asset-blocked item stays unticked with a pointer to
   `pending-commands.md`. Also update the checkpoint paragraph in
   `docs/model-extensions/cards/README.md` and the Card 09 status row plus the
   "Execution records" section in `docs/model-extensions/CARDS.md`.
7. Update `RESUME-NOTES.md`: state that integration edits are finished and ownership
   is released to the orchestrator, list changed paths (`jj diff --name-only`),
   the scratch paths under `/tmp/card09-*` that must be deleted at closure, and any
   live handle.

Be precise about what is *historical* worker evidence versus *post-integration*
evidence; the orchestrator's own re-run supersedes both. Never describe an artifact you
did not run as verified.
