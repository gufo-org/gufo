# HIP build audit (validation item 11)

Audited from `build/card06-gpu` (gpu-test: `CMAKE_BUILD_TYPE=RelWithDebInfo`,
`ENGINE_ENABLE_HIP=ON`, `BUILD_TESTING=ON`) and
`build/card06-release` (production: RelWithDebInfo, HIP ON, tests OFF,
`GUFO_EXTENSIONS_FROM_LOCK=ON` → `gufo: extension mode=lock; included=[];
excluded=[]`). Toolchain: repo-pinned `nix develop` (GCC 16.2, HIP Clang 24,
ROCm 10.1); the main-control arm builds under the same candidate dev shell.

## (a) gufo_core compile graph after the move

`compile_commands.json` + ninja graph (captured 2026-10-31):

- `src/core/speculative/speculative_verifier.cpp` → compiled into `gufo_core`
  (`CMakeFiles/gufo_core.dir/...`, root CMakeLists HIP block, unchanged entry)
  with flags `-O2 -g -DNDEBUG`, `-DENGINE_ENABLE_HIP=1`,
  `-include src/core/hip/libstdcxx_noinline.hpp`.
  `g++ -fsyntax-only -H` with the exact compile command (inside
  `nix develop`) exits 0 with **zero** `/src/models/` header inclusions — the
  core speculative TU no longer pulls any Qwen header. Before this card the
  same TU pulled `models/qwen/{dflash_policy,generator,tokenizer,hip/executor}.hpp`.
- `src/models/qwen/hip/qwen_speculative_verifier.cpp` (NEW, model-owned source)
  → also compiled into `gufo_core`, registered through
  `src/models/qwen/CMakeLists.txt` `target_sources(gufo_core PRIVATE ...)`
  under `ENGINE_ENABLE_HIP`, exactly where every other Qwen 27B source lives.
  Same target ⇒ same per-target flags, includes, definitions, assertions and
  sanitizer state as its parent sources; no per-source COMPILE_OPTIONS were
  added or removed (the old root-listing entry carried none).
  Its `-fsyntax-only -H` trace lists 19 `/src/models/` inclusions (expected:
  this is model code binding `hip::QwenGpuExecutor`).
- `src/core/speculative/{draft_backend,prompt_lookup}.hpp` are header-only and
  model-free; `grep` over the compile graph confirms no other core source
  includes `src/models/**` — this is now machine-enforced by the new
  `--fail-on-core-model-includes` gate (see below), so the boundary fails if
  violated again.

Interpretation for the card wording "the moved code must no longer be compiled
into gufo_core as a core source": the moved Qwen-specific translation unit is
no longer listed as a core (src/core) source anywhere; it is registered by the
model's own CMakeLists. It still links into the `gufo_core` archive because
Card 07 (RFC step 5) is the card that extracts Qwen 27B from `gufo_core` into
its own library; this card deliberately does not pre-run that extraction.

## (b) Per-source flags / definitions / sanitizer state

Diffed compile_commands entries: the new model TU's command equals the old
core TU's command except for the file/object paths (same `-O2 -g -DNDEBUG`,
same defines list, same `-include libstdcxx_noinline.hpp`, same warning set;
no sanitizer flags are configured in these presets — `cpu-sanitizer` is a
separate preset and was not part of the shipped path). No source-specific
`COMPILE_OPTIONS`/`COMPILE_DEFINITIONS` property existed for
`speculative_verifier.cpp` (root `CMakeLists.txt` sets such properties only
for MiniMax `dit.hip`), so nothing needed replicating.

## (c) ISA comparison in stripped NDEBUG production objects — PASS

Compared production objects (both trees, same pinned dev shell):
main `/tmp/gufo-main-ctl/build/card06-main-release/...speculative_verifier.cpp.o`
vs candidate `build/card06-release/...speculative_verifier.cpp.o` (engine TU)
and `build/card06-release/...src/models/qwen/hip/qwen_speculative_verifier.cpp.o`
(moved adapter TU). Raw per-function census: `isa-audit-output.txt`; the final
per-function comparison scripts re-ran `objdump -d --no-show-raw-insn` with
normalization only for (i) the `gufo::speculative (anonymous)` → `gufo::hip
(anonymous)` namespace relocation and (ii) object-position-dependent renderings
(branch-target immediates and multi-byte NOP padding encodings).

- **Moved adapter:** 22/22 fixed-key methods compared: 16 identical,
  4 address-encoding-only diffs, 2 destructors + 1 ctor presence-verified;
  class verified verbatim at source level (see deepseek-static-analysis.md
  item 3). The retained census does not prove all 27 bodies equal.
- **Core engine TU:** 59 identical bodies and 69 changed bodies. The latter
  classification is attributed to alias-rename call/jump encodings and moved
  adapter call boundaries, backed by the empty normalized source diff after
  the TokenId alias rename and adapter/constructor exclusion (independently
  confirmed in review; deepseek-static-analysis.md item 2). This is a
  source-level semantic proof, not a retained per-hunk ISA proof for all 69.
  Added/removed symbols include the owning constructor and neutral
  GenerationOptions variants; no stronger body-equality claim is made.
- **Kernel resources:** no `.hip`/kernel source moved, flag-changed or
  re-registered (compile-graph audit above; root HIP source list unchanged
  apart from deleting the core verifier entry, which is not a kernel). No
  code-object size or kernel-resource property of any untouched TU changed.

Verdict: static/source equivalence supported; adapter ISA coverage is limited
to the retained fixed-key comparisons above. Engine semantics are backed by
the empty normalized source diff and 59 identical bodies; no kernels moved.

## Boundary now enforced in tools/ci/check-extension-isolation.py

- New `core_boundary_violations()` + `--fail-on-core-model-includes`: fails
  with path:line if any `src/core/**` file includes a header resolving under
  `src/models/**` or `src/families/**` (literal or relative spellings).
- CTest `extension_core_model_includes_gate` (labels `cpu;repository`) runs
  the flag over the repository.
- `tests/tools/check_extension_isolation_test.py` adds a fixture test
  (models + families + relative include blocked; core→core and cli/tests
  edges unaffected) and a repository-level assertion.
- Other layers (cli/cache/tests) stay report-only until their cards, matching
  the RFC ratchet ("becomes failing ... in the PR that migrates it").
