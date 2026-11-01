> Instructions given to the integration agent (written before it ran).
> A brief, not a result.

# Integrator brief — Card 14 review integration

Repository: `/home/mixer/gufo` (jj working copy `@`, parent commit `qupqlxsk 642af9d6`).
You are the **integration agent** (`codex-exec-writer`) for Card 14. You have
**sole editing ownership** of the shared worktree while you run.

## Absolute constraints

1. **Never run `jj commit`, `jj new`, `jj squash`, `git commit`, or anything that
   changes revision state.** The orchestrator commits. Read-only commands (`jj st`,
   `jj diff`, `jj log`) are permitted.
2. `docs/model-extensions/RFC.md` must stay **byte-identical and uncommitted**.
   Do not read or write `HANDOFF.md` (deleted; its decisions now live in `CARDS.md` and `cards/README.md`).
3. **Scope discipline**: during this card run only unit/identity/CPU/hosted/static
   gates. Functional harness runs, main-control timings, and speed benchmarks are
   deferred to the post-Card-14 qualification window.
4. Keep tool output small: redirect logs to `/tmp/card14-integration/` and inspect
   with `head`/`tail`/`grep`. Scratch in `/tmp/card14-*` only. No file > 1 MB under `docs/`.
5. Host RAM: check `free -g` before heavy operations (keep >= 15 GiB headroom). Do not
   stop llama-swap services.

## Tasks

### 1. Resolve Review Findings
- **F1 (Nit)**: In `tools/ci/generate-infrastructure.py:556-557`, remove the dead / unreachable lines after `return 1`:
  ```python
  print("PASS: Infrastructure files are up to date with model metadata.")
  return 0
  ```
- **F2 (Minor)**: In `src/models/qwen_image_21/extension.json`, remove `"fixtures"` and `"functional"`. In `src/models/minimax_h3/extension.json`, remove `"functional"`.
  Run `python3 tools/ci/generate-infrastructure.py --check` (or `--write` if any target files changed, then verify `--check` exits 0 with zero drift).
  Verify schema with `nix develop -c python3 tests/extensions/metadata_test.py --schema`.

### 2. Rerun Affected Gates ONCE
Rerun the affected gates and record exact commands, exit codes, and output summaries in
`docs/model-extensions/evidence/card14/integration-reruns.txt`:
1. `python3 tools/ci/generate-infrastructure.py --check`
2. `python3 -m unittest tests/tools/generate_infrastructure_test.py`
3. `nix develop -c python3 tests/extensions/metadata_test.py --schema`
4. `python3 tools/ci/check-docs.py`
5. `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py`
6. `nix develop -c cmake --build --preset cpu-test --target check-pr`
7. `nix build .#checks.x86_64-linux.pr`
8. `nix eval --raw .#packages.x86_64-linux.default.drvPath` (derivation hash invariance)

### 3. Create Evidence Index
Create `docs/model-extensions/evidence/card14/README.md` listing provenance, all three agent IDs, all evidence files with descriptions, and the gate verification summary.

### 4. Update Execution Record & Ledgers
- Update `docs/model-extensions/cards/14-generated-infrastructure.md` execution record:
  - Mark status as integrated, ready for orchestrator closure.
  - Mark all 4 Definition of Done checkboxes `[x]`.
  - Detailed Execution Record with agent IDs, finding resolutions, exact gate commands and results, and deferred items.
- Update `docs/model-extensions/CARDS.md` and `docs/model-extensions/cards/README.md` to reflect Card 14 completion and the stop of the current implementation milestone.
- Update `docs/model-extensions/evidence/card14/RESUME-NOTES.md` with final notes and release editing ownership to orchestrator.
