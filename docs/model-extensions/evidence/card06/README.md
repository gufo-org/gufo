# Card 06 evidence index — move model-owned speculation out of core

Candidate: jj working copy @ `kppqwqnv` (parent `cfc9802f` "docs(extensions): record card 05 closure verification").
Main-control checkout for matched timing: `/tmp/gufo-main-ctl` @ detached `5d66de8cd98e7c09ed4b6ee0af0c195adc3f861d` (bookmark `main`).
Machine: Strix Halo gfx1151, 125 GiB RAM. Toolchain: repo pinned `nix develop` (GCC 16.2, HIP Clang 24, ROCm 10.1) — both arms build under the candidate's `nix develop` environment.

Build directories (fresh, task-owned): `build/card06-gpu` (gpu-test: RelWithDebInfo, HIP ON, BUILD_TESTING ON),
`build/card06-release` (candidate production, lock-mode), `/tmp/gufo-main-ctl/build/card06-main-release` (main control production).

Status table (updated as each check completes; PENDING lists the exact remaining command):

| # | Validation item | Status | Evidence |
| --- | --- | --- | --- |
| 1 | gpu-test configure + affected targets build | Historical configure retained; build exit unretained, re-verification PENDING | configure-gpu-test.txt, build-gpu-test.txt |
| 2 | Identity goldens (qwen27b, flash-next, deepseek) exact bytes, zero skips | PASS 3/3 | unit-identity-goldens.txt |
| 3a | speculative unit tests (mock): speculative_verification_test, speculative_corpus_test | PASS 2/2 | unit-speculative-verification.txt |
| 3b | qwen_dflash_gpu_test on gfx1151 (real target+draft, snapshot/persistent restore, rollback replay) | PASS 12.25s | unit-qwen-dflash-gpu-initial.txt |
| 3c | qwen38_flash_next.prompt_lookup / .verification_sampling / .mtp_sampling, qwen_dflash_upload_lifetime_test, qwen_dflash_noncausal_attention_ops_test | PASS 5/5 (gfx1151) | unit-gpu-queue.txt |
| 3c2 | qwen_mtp_gpu_test (27B MTP draft backend) | PENDING — asset missing: no 27B-compatible MTP draft under ./models (only the Flash-Next-shared MTP GGUF exists; loading it errors `Missing Qwen MTP tensor: token_embd.weight`). Flash-Next MTP profile and prompt_lookup.py qualification remain deferred; no completed end-to-end pass is claimed here. Command: `GUFO_QWEN27B_MODEL=<27B gguf> GUFO_QWEN27B_MTP_MODEL=<27B MTP gguf> nix develop -c ctest --test-dir build/card06-gpu -R '^qwen_mtp_gpu_test$' --output-on-failure` | unit-gpu-queue.txt |
| 3d | ds4.target / ds4.dspark / ds4.serving (DeepSeek numerical quality + serving, local models) | SKIPPED BY USER DECISION (host capacity, 2026-10-31) — NOT a pass. In-card substitute: static path analysis + weight-free DeepSeek CPU tests `ds4.sampling/ds4.template/ds4.cli/ds4.load` + identity golden — 5/5 passed, zero skips. GPU kernel quality tests (ds4.markov/compressor/router/quantization/projections/attention) need HIP and were out of the weight-free substitute scope. Re-verify in the post-Card-14 window per deepseek-static-analysis.md. | unit-ds4-cpu.txt, deepseek-static-analysis.md |
| 3e | DeepSeek static correctness analysis (user-directed substitute) | PASS (static only, explicitly not runtime serving evidence) | deepseek-static-analysis.md |
| 3f | Identity trio re-run after final doc edits | PASS 3/3 exact bytes | unit-identity-goldens.txt (re-run 2026-10-31: `ctest --test-dir build/card06-gpu -R '^(qwen27b_continuation_identity_test|qwen38_flash_next_continuation_identity_test|deepseek_v4_flash_continuation_identity_test)$'` → 100% tests passed out of 3) |
| 6c | Task-scoped documentation check | PASS (0 errors outside pre-existing 164-error hrx-system baseline; fixed one self-inflicted HANDOFF.md link) | docs-check.txt |
| 4 | check-format.py | PASS (727 files) | check-format.txt |
| 5 | Extension API drift (SURFACE unchanged) | PASS (6 paths) | check-extension-api.txt |
| 6 | Core-boundary isolation gate + unit coverage | PASS (gate exit 0; 6/6 tests) | check-core-boundary.txt |
| 6b | Compile-graph audit (core TU zero model includes; model TU in gufo_core via model CMakeLists; flags) | PASS | hip-build-audit.md |
| 7 | Functional matrix (Qwen27B Q4/Q8 {off,dflash2}, Flash-Next {off,mtp}; per-config groups standard/single-session/cache-pressure/cache-stalls) | DEFERRED to the post-Card-14 qualification window by user decision 2026-10-31 — PENDING, not a pass. Historical partial attempt retained in verdicts only: all `run.py` suites that completed passed; main-control arms rc=0; Flash-Next MTP compare failed with HARNESS command mismatch (`follow-up changes comparison_command`), zero measurements (`measurements=[]`): this is no timing signal; historical candidate arm exit codes were 1/2/2/2 — see compare-flash-next-q4-mtp.json and pipeline-status.txt. The raw report tree and the DeepSeek harness-bug rows were deleted by the user (context pollution); pipeline halted. Exact rerun pattern: `tests/functional/run.py` per the harness README with `--output /tmp/cardNN-*` (caches always `/tmp/<test-specific-dir>`, deleted afterwards — see cards/README.md Validation conventions). | compare-flash-next-q4-mtp.json, functional-impact-plan-pr.txt, pipeline-status.txt |
| 7b | Suite scope decision | `impact.py --tier pr` plan retained for all 8 rows (functional-impact-plan-pr.txt); full-tier impact selects every suite (~47/row) which is the devlab/release matrix, not this PR's affected set; deferred release-only suites are listed in the plan file. `metrics` was re-added explicitly although the PR tier defers it. | functional-impact-plan*.txt |
| 8 | prompt_lookup.py Flash-Next MTP + Qwen27B DFlash2 accepted-copy evidence | DEFERRED to post-Card-14 window — PENDING. Command: `nix develop -c python3 tests/functional/prompt_lookup.py --output /tmp/card06-lookup-<tag> -- <serve cmd with --speculative mtp --mtp-model models/qwen38-flash-next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf>` (and the Qwen27B `--speculative dflash2 --dflash-model models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf` variant) |
| 9 | Matched main-control timings + compare.py per profile/mode (5% and 10 ms per request/phase) | DEFERRED to post-Card-14 window — PENDING (arms prepared: build/card06-release/gufo vs /tmp/gufo-main-ctl/build/card06-main-release/gufo, same pinned dev shell) |
| 10 | Standard speed benchmark (model-bench.py, Gufo-side continuity for affected modes) | DEFERRED to post-Card-14 window — PENDING; exact commands in notes below |
| 11 | HIP build audit: gufo_core compile graph, per-source flags, moved-TU ISA | PASS (graph, flags and per-function ISA all PASS — adapter 22/22 fixed-key methods: 16 identical, 4 address-encoding-only diffs, 2 destructors + 1 ctor presence-verified; class verbatim at source level; engine 59 identical + 69 alias-rename call/jump encoding differences backed by empty normalized source diff) | hip-build-audit.md, isa-audit-output.txt |
| 12 | nix build .#checks.x86_64-linux.pr (LAST, after all code/doc edits incl. review/integration) | PASS exit 0 (orchestrator, after final edits; fresh check-format/check-pr-tests/check-pr derivations, not cached). Card 05 snapshot caveat applies. Final user-mandated `codex exec review` stamp: exit 0, no critical/major findings. | hosted-pr-check.txt, integration-gates.txt |

## Implementation summary (what changed and why)

- `src/core/speculative/draft_backend.hpp`: qwen tokenizer include replaced by
  `src/core/sampling.hpp`; `tokenization::TokenId` → neutral `sampling::TokenId`
  (same `std::uint32_t` contract the sampler already uses).
- `src/core/speculative/speculative_verifier.{hpp,cpp}`: all `src/models/qwen`
  includes removed. `QwenGpuSpeculativeTarget` and the two
  `hip::QwenGpuExecutor` constructors moved out; neutral owning
  `std::unique_ptr<ISpeculativeTargetExecutor>` constructors added.
  `SpeculativeOptions::dflash_policy` (never read by the engine) removed;
  a neutral `speculative::GenerationOptions{max_new_tokens, sampling}` replaces
  the borrowed `models::GenerationOptions` in `Generate`.
- NEW `src/models/qwen/hip/qwen_speculative_verifier.{hpp,cpp}`: model-owned
  `QwenGpuSpeculativeTarget` adapter (moved verbatim) +
  `hip::CreateQwenGpuSpeculativeVerifier()` factories (owning/borrowing draft).
  Compiled into `gufo_core` through `src/models/qwen/CMakeLists.txt`
  `target_sources` — model-owned placement, same target, same flags;
  card 07 extracts the library.
- Serving: `inference_backend.{cpp,hpp}` use the factory; DFlash policy now
  flows `TextSpeculativeConfig -> QwenTextRunner::dflash_policy_ -> DFlash draft
  config + identity writer`. `continuation_identity.{hpp,cpp}`:
  `QwenCompatibilityIdentity` takes `DFlashDraftPolicy` as an explicit input;
  emitted identity bytes unchanged (goldens pass).
- `prompt.cpp`, `bench.cpp`, `dflash_test.cpp`: construct through the factory;
  `test_speculative_verification.cpp` uses only core types.
- Isolation ratchet: `tools/ci/check-extension-isolation.py`
  `--fail-on-core-model-includes` fails if any `src/core/**` file includes a
  `src/models/**` or `src/families/**` header; new CTest
  `extension_core_model_includes_gate`; unit coverage in
  `tests/tools/check_extension_isolation_test.py`.

## Deferral and remaining exact commands

- The detached queue was halted 2026-10-31 per user instruction (functional
  work deferred to the post-Card-14 window; DeepSeek runtime dropped).
  `pipeline-status.txt` ends with the `STATUS DEFERRED` line; no queue process
  remains and no server/scratch directories survive.
- Standard speed benchmark, per docs/models/*/BENCHMARKS.md (to run on the
  candidate production binary after the queue; Gufo-side continuity is the
  gate, reference-project columns are not required by this card):
  `nix develop -c python3 tools/bench/model-bench.py --model qwen3.8-27b --gguf $Q4 --dflash $DF run --target gufo --table <affected-table-id>`
  (same pattern for `qwen3.8-flash-next --mtp $MTP` and `deepseek-v4-flash`
  with `--dspark $DSS`); table ids from `docs/models/<model>/artifacts/bench.json`.

## Third-agent integration (2026-10-31)

See `integration-gates.txt` for exact commands and exits. Required Nix identity,
unit and formatting reruns are PENDING: sandbox denies the Nix daemon socket.
An existing store CTest fallback passed the affected units 8/8 and identity trio
3/3 (full output in
`unit-identity-goldens.txt`); it does not replace pinned-environment verification.
API check and --write pass (6 paths), SURFACE unchanged; isolation passes; RFC
SHA matches. Compare verdict excerpts are `compare-*.txt`, alongside their JSON.
Historical build exit was not retained; `build-gpu-test.txt` records that limit.
