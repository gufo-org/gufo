# Model extensions: new-chat handoff

Current user-authorized milestone: complete cards 01–14, then stop. Cards
15–29 remain documented future work. Use the three-agent workflow for every
card: implementation, independent review, then a third agent integrating
review comments and finishing verification. Complete cards sequentially.

## Authoritative starting state

- Workspace: `/home/mixer/gufo`; use `jj` and read `AGENTS.md`.
- Cards 01–05 are complete through all three agents. Card 05's implementation
  parent is `381c1648`, `feat(extensions): add cmake discovery system`; its
  review/integration follow-ups sit in the working copy `@`
  ("feat(extensions): card 5 follow-ups") for the orchestrator to commit.
- Card 05 used three distinct agents: implementation `/root/card05_implement`,
  independent review pi `reviewer` (static) plus pi `delegate` (execution pass),
  and integration pi `worker`. Findings, dispositions, exact commands and
  remaining gates are in
  `docs/model-extensions/evidence/card05/review-and-integration.md` and
  [its evidence index](evidence/card05/README.md). Next is **Card 06**
  (model-specific speculation out of `src/core/speculative/`), a text-path card:
  it needs the affected text profiles plus `long-context` and `cache`, the
  Card 02 identity goldens, and matched per-request timings against a main
  control.
- Editing ownership returns to the orchestrator when the Card 05 integration
  agent finishes; start Card 06 only after that and after the commit. All
  build/test jobs and the model download have finished; no old jobs need
  restarting.
- The current diff, except the original RFC, contains task-owned Card 05 work
  plus this handoff and the documentation-only card split. Inspect `jj status` and the actual files before relying
  on this checkpoint. Do not discard or reset pending changes.
- User-owned `docs/model-extensions/RFC.md` was added before implementation.
  Preserve it unchanged and uncommitted. Its SHA-256 is
  `2ce2becbc0bf0f844702b624de6827c2bae6eba921ab8a1ea883659fa2f96448`.
  Commit only task-owned paths after review integration.

## Read before continuing

- [CARDS.md](CARDS.md): sequential index, dependencies and shared evidence gates.
  [cards/README.md](cards/README.md) defines the workflow; each numbered card owns
  its detailed plan, acceptance criteria and execution record. Stop after Card 14
  is actually verified.
- [RFC.md](RFC.md): original specification and migration requirements.
- [INFRASTRUCTURE.md](INFRASTRUCTURE.md): implemented metadata/build/API
  decisions and truthful legacy-model bootstrap.
- [SPIKE.md](SPIKE.md): isolated Flash-Next spike decisions and limitations.
  The spike source remains in `/tmp/gufo-card03-spike`; do not merge it blindly.
  Its functional checks passed, but production performance parity remains
  **unproven**. Original failed comparisons and later inconclusive controls
  are retained; no later migration gate is waived by this investigation.

## Model files and machine

The user freed disk space and explicitly authorized downloads into `./models`.
That ignored path already symlinks to `/persist/models`. Both pinned DeepSeek
files have downloaded and passed complete SHA-256 verification:

- Directory: `models/deepseek-v4-flash/`.
- Target: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`,
  86,720,111,488 bytes; revision `1cd7b564460821938add0475a60b942c409295e0`;
  SHA-256 `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`.
- Sidecar: `DeepSeek-V4-Flash-DSpark-support-0731.gguf`, 5,989,114,272 bytes;
  revision `e7f04037032990db0346398d249baf9fb9df1ccc`;
  SHA-256 `7e319924541db3f7a163ed7e11d7532a70d48228ab59d36cb81e1d4511885360`.
- Local download provenance: `models/deepseek-v4-flash/download-verification.json`.
  Pinned identities also exist in the model's tracked quality artifacts.
- Flash-Next and Qwen 27B GGUF files were found under `./models`; revalidate
  actual paths and drafts before each affected run. Do not redownload verified
  files unnecessarily or delete unrelated model/cache data.
- Local machine: Strix Halo `gfx1151`, 125 GiB RAM. Current pinned Nix toolchain:
  GCC 16.2, HIP Clang 24, ROCm 10.1. `release` intentionally uses
  `RelWithDebInfo` with `BUILD_TESTING=OFF`. Use the repository's pinned build
  instructions; qualify actual flags and dependencies, not directory names.

## Verification and known limits

Card 05's implementation checks include real default/lock CPU executables and
CLI provenance, 12/12 focused CTests, a final 4/4 affected-fixture check, nine
discovery groups, formatting, platform-only builds, and Nix evaluation/source
admission checks. Independent review and integration reproduced them, added four
discovery cases and one selector-failure case, and recorded the results, so Card
05's infrastructure evidence is current; what remains outstanding is the future
lock-mode HIP and Windows evidence, not this card's checks.

Apply the repository's optimization skill for profiling/kernel work and the
affected numerical/functional/speed gates for later migrations. Preserve exact
cache identities, actual modes/drafts, model/toolchain/settings/cache-history
provenance and every result. Timings are per request and phase at 5% and 10 ms;
investigate and fix confirmed regressions. Inconclusive observations and
missing evidence never become unqualified passes. Test correctness on the
candidate; use matched main controls for affected timings, without duplicate
unrelated model sweeps. Default and actual shipped lock-mode HIP artifacts
need separate evidence. Windows workflow wiring is not Windows execution.

The existing whole-workspace documentation check reports 164 broken links
under the unrelated `hrx-system/` tree; task-scoped documentation checks pass.
Preserve unrelated work rather than changing it to suppress those diagnostics.

Keep the ledger and compact per-card evidence current. Continue autonomously
through Card 14 after Card 05 review/integration, and report the exact completed
scope and remaining limitations when that milestone is achieved.
