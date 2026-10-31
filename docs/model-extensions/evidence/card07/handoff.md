# Card 07 implementation handoff

**This agent has finished editing and releases editing ownership of `/home/mixer/gufo`.**
Nothing is committed or staged: `jj status` shows the whole change set in the working copy
`@` (change `zwllvvnv`), parent `kppqwqnv 11af44f9` (Card 06). No bookmark, remote or
unrelated work was touched; `docs/model-extensions/RFC.md` is byte-identical
(`2ce2becb…96448`) and uncommitted.

Two implementation agents contributed (the first died on infrastructure after the moves
and builds, before writing any evidence; the second finished the gate set, the gate
registration, the fixtures and this record). Reviewers should read
`docs/model-extensions/evidence/card05/review-and-integration.md` and the Card 06 record for
the surrounding contracts, then this directory.

## What changed (282 files; grouped)

New family layer — `src/families/qwen/` (93 files):

- `README.md` (library table + manifest schema), `AGENTS.md` (the control-token catalogue
  and Python-mirror rules, moved here because the header is now family-owned),
  `CMakeLists.txt` (adds the six subdirectories, HIP-guarded where needed).
- `control_tokens/` — `control_tokens.hpp` (INTERFACE `gufo_qwen_control_tokens`).
- `hip_common/` — `continuation_hooks.hpp`, `execution_policy.hpp`, `attention_policy.hpp`,
  `recurrent_state.hpp`, `gemm_route.hpp`, `quant_ops.hpp`, `rope.hpp` (INTERFACE
  `gufo_qwen_hip_common`): the layer below both kernels and vision that breaks the
  token→vision-RoPE and kernel→recurrent-state edges.
- `config/` — `qwen_config.{hpp,cpp}`: free `gufo::models::qwen::ExtractQwenConfig(GgufReader
  const&, std::string*)`; `GgufReader::ExtractModelConfig` deleted from
  `src/core/gguf_reader.{hpp,cpp}` (111 removed lines, 0 added to core). Every string
  literal is preserved byte-for-byte (verified by extracting all literals from the removed
  block and the new file: identical sets apart from the two include paths).
- `text/` — `tokenizer.{hpp,cpp}`, `chat_template.{hpp,cpp}` and the pinned
  `reference/chat_template.jinja` fixture.
- `kernels/` — `ops.hpp`, `ops/*.hpp` and 38 kernel TUs, with the wave64 /
  `-fno-slp-vectorize` / `-gline-tables-only` / iterative-ilp properties re-anchored to
  `gufo_qwen_kernels` (comments kept).
- `vision/` — `encoder.{hpp,hip}`, `device_input.{hpp,hip}`, `image_cache.{hpp,cpp}`,
  `prompt.{hpp,cpp}`.
- `*/library.json` — the six manifests (`schema`, `name`, `target`, `kind`, `purpose`,
  `depends`); the gate validates field shape and include graph, without CMake cross-checking.

Qwen 27B library (92 files under `src/models/qwen/`):

- `CMakeLists.txt` now defines `gufo_qwen27b` instead of `target_sources(gufo_core …)`,
  names its family dependencies explicitly, and re-anchors its per-source HIP options.
- `AGENTS.md` is now a short model-directory file pointing at the family file (no
  duplicated token catalogue).
- Remaining sources keep their paths; includes rewritten to `src/families/qwen/...`.

Core and build wiring:

- `src/core/gguf_reader.{hpp,cpp}` — Qwen config extraction removed (CPU core no longer
  holds Qwen validation flow).
- `src/core/quant/ggml_dequant.hpp` — comment-only path reference update.
- `CMakeLists.txt` — `add_subdirectory(src/families/qwen)`, new `extension_family_graph_gate`
  CTest running `check-extension-isolation.py --fail-on-family-graph`, and relinking of
  `gufo_llm_cli`, `gufo_http`, `gufo`, `gufo_kernel_bench_support`, `gufo-kernel-bench`,
  `tune_hipblaslt`, `wmma_layout_test` and the affected test targets.
- `cmake/Checks.cmake` — the new gate added to `gufo_pr_tests`.
- `tools/ci/check-extension-isolation.py` — `QWEN_PREFIXES` (header closure now covers
  `src/families/qwen/**` and `src/models/qwen/**`), `family_library()`,
  `family_analysis()`, `cycle_violations()` and `--fail-on-family-graph`;
  `--fail-on-core-model-includes` behaviour unchanged.
- `tests/tools/check_extension_isolation_test.py` — 8 new tests (this resume): clean family
  graph, undeclared *and* unused dependency, unknown dependency, cycle, self-dependency,
  family→model leak, unmanifested family file, manifest name/target/kind/purpose
  validation, plus repository-level "family graph holds" and "both Qwen trees inventoried"
  and a control-token header↔Python-mirror link test. 14 tests pass.

Other: `tests/families/qwen/{CMakeLists.txt,qwen_config_test.cpp}` (config cases moved out
of `tests/core/gguf_reader_test.cpp`, all assertions retained), include rewrites across
`tests/`, `tools/`, `docs/models/**`, and `tests/functional/{impact.json,devlab.json,
test_impact.py,test_devlab.py}` path-pattern updates (registries only; the harnesses were
not run).

## Deviations from the brief, with reasons

- `ModelConfig::IsValidQwen()` stayed in `src/core/model_config.hpp`. The brief allowed
  moving it only if the extractor and tests were its sole callers; `src/models/qwen/
  continuation_layout.cpp` calls it too, and `ModelConfig` must remain in core until
  Card 10. Moving it would have created a second validation entry point. The extracted
  function calls `config.IsValidQwen()` exactly as before.
- `gufo_qwen_kernels`/`gufo_qwen_vision` expose HIPCUB/ROCPRIM as `SYSTEM PUBLIC` where
  `gufo_core` had `SYSTEM PRIVATE`, because family headers are now compiled by several
  targets. Verified to add nothing to the moved sources' own command lines.
- `src/models/qwen/` was not renamed to a model-specific directory (Card 20 scope, stated
  in `src/models/qwen/AGENTS.md`).

## Gate results (details and exact commands in the sibling files)

PASS — G1 configure; G2 gpu/cpu/release builds (exit 0, incremental rebuild after format);
G3 identity goldens 3/3 exact bytes; G4 affected selection 82/82 with 8 asset-gated skips;
G4b real-27B GPU units `qwen27b_target_test` (orchestrator rerun with absolute assets: Passed
91.67 s, exit 0 — F1 closed), `qwen_dflash_gpu_test` (7.3 s),
`qwen_dflash_upload_lifetime_test` (3.2 s), `qwen_vision_prepare_test`; G4c
`qwen27b.cli-options` + `cli_test.py --scope sampling` (48 generations, AR/fixed/adaptive);
G5 CPU-only libraries + 16 CPU tests + HIP-free compile/link proof; G6 production
cache/`compile_commands.json` audit and per-source option survival; G7 ISA parity
(84/84 sources flag-clean; 2438/2461 host functions byte-equal; the 23 differences are
`__FILE__`-length `.L.str` offsets and per-TU `__hip_cuid_*` references, no opcode or
control-flow change); G7b kernel scratch 2692/2692 identical in both build modes;
G8 formatting (730 files); G9 extension API SURFACE unchanged (6 paths) → no ext-api label;
G10 isolation both gates + interim edge list; G10b family gate + 14 fixture tests;
G10c `check-docs.py` zero errors; G11 `gufo diagnose` on both binaries.

FAIL, pre-existing and reproduced on the parent revision's own binary (byte-identical FAIL
list): registered `kernel_resources_test` against the stale `tools/ci/kernel-resources.json`
(30 kernels over budget on both sides, zero scratch deltas from this card). Needs an owner
decision, not a Card 07 baseline edit — see `pending-commands.md` §8.

PENDING (never claimed as passes): the functional matrix and `long-context`/`cache`
histories, `prompt_lookup.py`, matched per-request/phase main-control timings, CLI
`prompt`/`bench` matched controls, standard speed benchmark, shipped lock-mode HIP artifact,
Windows native build/run, hosted `nix build .#checks.x86_64-linux.pr`, and the
asset-blocked MTP/`qwen27b.cli`-full/`qwen_vision_serving_test` rows — all with literal
commands in `pending-commands.md`.

## What a reviewer should look at specifically

1. `src/families/qwen/*/library.json` vs `src/families/qwen/*/CMakeLists.txt`
   vs `src/models/qwen/CMakeLists.txt`: does each manifest's `depends` match both the real
   include edges and the CMake link edges? (The gate audits include edges only; CMake wiring needs manual review.)
2. `src/families/qwen/kernels/CMakeLists.txt` per-source options — are all five special
   compilations bound to the right target with the original comments intact?
3. `tools/ci/check-extension-isolation.py::family_analysis` — is the enforcement strong
   enough (declared == real in both directions, cycles, family→model ban, header
   ownership), and is the new CTest wired into `gufo_pr_tests` correctly?
4. `src/core/gguf_reader.cpp` deletion vs `src/families/qwen/config/qwen_config.cpp` —
   independent semantic check of extraction/validation and error strings.
5. `tests/functional/impact.json` and `devlab.json` path updates: moved sources must still
   escalate the right suites (registry text only, never executed here).
6. `docs/model-extensions/evidence/card07/kernel-resources.txt` — the pre-existing baseline
   failure needs an owner, not a silent allowance change.

Implementation died before cleanup. Integration salvaged comparator sources and removed `/tmp/card07-work` and its own scratch at 2026-10-31T12:33:50.399297+00:00. `/tmp/card07-orch` remains orchestrator-owned; see integration-reruns.txt.

Integration corrections: G4b target is PENDING, never PASS. G7 full device section parity: 39 objects, 117 section comparisons identical (device-isa.txt). Manifest target/kind validation checks field shape only; CMake target/kind/link edges are NOT cross-checked by this lexical gate.

F5 CPU consequence: Flash-Next engine.hpp includes text and vision unconditionally, while the CPU library compiles only config/layout/ngram/rope/weights (engine.cpp is HIP-only). CPU config/oracle users work; including engine.hpp is not a supported CPU engine API because encoder.hpp pulls HIP headers. Wiring is intentionally left unchanged within this extraction; no CPU inference support is claimed.
