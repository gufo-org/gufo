Orchestrator input brief — planned commands, none of these are results.

# Card 08 — worker brief (implementation, sole editing owner)

Repo: `/home/mixer/gufo` (jj working copy `@` = empty child of `zwllvvnv ec59bed0`
"refactor(extensions): extract acyclic qwen family libraries"). You are the ONLY
writer in this worktree. **Do not commit, do not push, do not `jj new`/`abandon`.**

## Read first (in this order)

1. `AGENTS.md`, `docs/model-extensions/CARDS.md` (shared acceptance gates + the
   2026-10-31 functional deferral decision), `docs/model-extensions/cards/README.md`
   (validation conventions).
2. `docs/model-extensions/cards/08-h3-eval-and-test-targets.md` — your contract.
3. `docs/model-extensions/RFC.md` §"Target graph", §"Metadata and build manifests",
   §"Tests, benchmarks, tools and documentation", and the step-5 row of the
   implementation-plan table. **RFC.md must stay byte-identical**
   (`sha256 2ce2becbc0bf0f844702b624de6827c2bae6eba921ab8a1ea883659fa2f96448`);
   never edit or stage it.
4. `docs/model-extensions/cards/07-qwen-family-libraries.md` (execution record:
   what is already proven, what is deferred) and
   `docs/model-extensions/evidence/card07/orchestrator-gates.txt` (the exact pinned
   gate command shapes you will mirror) plus
   `docs/model-extensions/evidence/card07/pending-commands.md` (the PENDING format).

## Hard constraints

- **In-card scope only**: unit tests (incl. the 3 identity goldens), weight-free
  model tests, CPU/hosted checks, formatting, static audits. **No
  `tests/functional/run.py`, no `prompt_lookup.py`, no speed benchmark, no
  main-control timings.** Record every deferred functional command verbatim in
  `docs/model-extensions/evidence/card08/pending-commands.md` as PENDING — never as
  a pass.
- Scratch lives in `/tmp/card08-*` **only** (build scratch, dumps, parity scratch).
  Every `/tmp/card08-*` directory you create must be deleted when its step finishes
  — failures included. Do not leave scratch behind.
- Evidence under `docs/model-extensions/evidence/card08/` stays small: curated
  commands, verdict lines, compact tables, short excerpts. **No file over 1 MB**, no
  raw build/server logs, no report trees. Redirect long build output to a `/tmp`
  file and record only the verdict lines.
- Keep your own tool output small: `head`/`tail`/`grep`/filters, never `cat` of a
  large file or a full `ninja` log.
- Host RAM: before any GPU-heavy run, `free -g` and require ≥15 GiB available.
  The llama-swap services stay running; if you truly need them paused, ask ONCE via
  `contact_supervisor` (`reason: need_decision`) and proceed either way without
  re-asking.
- Run everything from the repo root in the pinned dev shell (`nix develop -c ...`).
  `nix develop` works in this session (verified).

## Baseline state you must not destroy

Pre-existing, fully built **parent-revision** build directories (they are your
"before" evidence; **never build into them**):

| Dir | Config | Role |
| --- | --- | --- |
| `build/card07-gpu` | RelWithDebInfo, HIP ON, BUILD_TESTING ON, GUFO_BUILD_TOOLS ON | candidate test/binary baseline, 257 registered tests |
| `build/card07-cpu` | RelWithDebInfo, HIP OFF, BUILD_TESTING ON | CPU-exclusion baseline |
| `build/card07-release` | RelWithDebInfo, HIP ON, BUILD_TESTING OFF | production ISA / kernel-resource baseline |

Fast candidate path (do this **after** your edits, not before):
`cp -a build/card07-<x> build/card08-<x>` (preserves mtimes), then
`nix develop -c cmake -S . -B build/card08-<x>` to reconfigure (cached args), then
`nix develop -c cmake --build build/card08-<x> --target <affected-target> --parallel 8`
and finally the full `gufo` for the production gates. Copying preserves the baseline;
building in place does not. Do not delete the `card07-*` directories.

## Work items

### 1. Extract MiniMax H3 into its own model library

Today `gufo_core` compiles 13 H3 CPU sources (`CMakeLists.txt` lines ~323-335), the
9 HIP sources under `ENGINE_ENABLE_HIP` (lines ~376-388) and carries
`GUFO_FFMPEG_EXECUTABLE`/`GUFO_FFPROBE_EXECUTABLE` as **private** defines of
`gufo_core`; `dit.hip` has a per-source appended `-O2`.

- Create `src/models/minimax_h3/CMakeLists.txt` with `add_library(gufo_minimax_h3 STATIC …)`
  holding exactly those CPU sources (including the `$<$<PLATFORM_ID:Windows>:…>` media
  pair), the HIP sources + the `dit.hip` `-O2` under `ENGINE_ENABLE_HIP`, the
  `GUFO_FFMPEG_EXECUTABLE`/`GUFO_FFPROBE_EXECUTABLE` defines, the
  `${CMAKE_SOURCE_DIR}` include and the PUBLIC link interface the sources actually need
  (they currently inherit core's ICU/PNG/JPEG/WEBP/CURL/OpenSSL/OpenMP/HIP edges).
  **A target placed before the warnings/sanitizers block (`CMakeLists.txt` ~189-202)
  silently loses `-Wall -Wextra -Wpedantic` and sanitizer flags** — the existing model
  `add_subdirectory()` calls at lines 174-187 are in that position, which is why the
  DeepSeek/MMQ objects are unsanitized today. Place the H3 `add_subdirectory()` where
  the compile environment matches core's (i.e. after that block, next to `gufo_core`),
  unless you record an intentional, justified difference.
- Relink consumers that used to get H3 through `gufo_core`: `gufo_http`, `gufo`
  (its `GUFO_H3_SOURCE_MANIFEST` define stays working — H3 test targets also pass that
  define today at lines ~1256/1282/1298/1325), and every root-registered H3 test
  (`minimax_h3_*`, `video_cli_test`, `video_jobs_test`, `video_api_test`).
- Add `src/models/minimax_h3/library.json` (same shape as
  `src/families/qwen/config/library.json`: schema/name/target/kind/purpose/depends) plus
  the ownership fields from item 5.
- Nothing else may reference `src/models/minimax_h3/*` sources except that model's own
  `CMakeLists.txt` and its own test targets.

### 2. Comparison support and eval targets

- `src/testing/compare/logit_comparator.cpp` is in `gufo_core` today and is used by
  **production** `src/cli/bench/bench.cpp` plus `tests/models/qwen/cpu/model_oracle_test.cpp`,
  `tests/models/qwen27b/target_test.cpp`, `tests/testing/test_common.hpp`. Give it its
  own target (e.g. `gufo_compare`) + `library.json`, linked by the consumers; it must
  stay buildable in production and CPU builds.
- `src/eval/{dataset,extract,regrade}.cpp` → an eval target (e.g. `gufo_eval`) +
  `library.json`. `src/eval/http_client.cpp` stays compiled into the `gufo` executable
  (eval remains a core HTTP client). Consumers: `gufo`, `tests/quality/eval_regrade_test.cpp`,
  `tests/models/deepseek_v4_flash/eval_test.cpp`.

### 3. Move the eval dataset to a core fixture

`tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json` → `tests/fixtures/eval/antirez-ds4.json`
(the `tests/fixtures/<topic>/` convention already used by
`tests/fixtures/minimax_h3/…`). Update every reference: root `CMakeLists.txt`
(`GUFO_EVAL_SOURCE_DATA` ~line 468 and the `install(FILES …)` rule ~line 527),
`tests/models/deepseek_v4_flash/CMakeLists.txt`,
`tests/models/deepseek_v4_flash/{dataset_test.py,eval_test.cpp}`, and the Nix
production source filter in `.devops/nix/package.nix` (replace the
`tests` → `tests/models` → … → `antirez-ds4.json` admission chain with the single new
core path — the production derivation must not admit any other model test file).
`git rm`/`jj` the old path; do not copy both. Verify with
`grep -rn "antirez-ds4" $(git ls-files)` afterwards (historical card evidence files keep
the old path and are correct as-is).

### 4. CPU-only DeepSeek text target, no duplicate sources

`gufo_deepseek_v4_flash` (HIP, model CMakeLists line ~48) compiles `chat_template.cpp`
and `runtime/sampling.cpp`; `ds4_sampling_test` and `ds4_chat_template_test`
(`tests/models/deepseek_v4_flash/CMakeLists.txt` lines 12-27) recompile those same
files, and `openai_chat_test` (root lines 667-672) recompiles `chat_template.cpp`,
`tool_calls.cpp` **and** `src/cli/serve/openai_chat.cpp` (which `gufo_http` owns too).

- Create a CPU-only DeepSeek text target (e.g. `gufo_deepseek_v4_flash_text`) with
  exactly `chat_template.cpp` + `runtime/sampling.cpp`; `gufo_deepseek_v4_flash` links
  it instead of compiling them. It must **not** compile `tool_calls.cpp` (still owned by
  `gufo_http` until step 10) and must not add any HTTP tool-syntax migration.
  Verified for you: `chat_template.cpp` only uses header-only `constexpr`
  `DeepSeekToolSyntax` constants, so it has **no link dependency on `tool_calls.cpp`**.
- `ds4_sampling_test` / `ds4_chat_template_test` link the text target instead of
  compiling sources. For `openai_chat_test`, remove its duplicated
  `chat_template.cpp`/`tool_calls.cpp`/`openai_chat.cpp` sources and get them from
  `gufo_http` (+ the text target) — one implementation per source is the DoD.
- Flag/assertion state: those files are currently compiled **without** warnings or
  sanitizers inside the DeepSeek library (its `add_subdirectory` at line 174 precedes the
  block at 189-202) but **with** `-Wall -Wextra` + sanitizers + `-UNDEBUG` when compiled
  by the test executables (`add_compile_options(-UNDEBUG)` at ~line 547). Choose one
  state per moved source, prefer preserving the strongest state its consumers had,
  and record the old/new per-source flags, assertion (`NDEBUG`/`-UNDEBUG`) and sanitizer
  state in `flags-and-assertions.md`. Do not silently change the DeepSeek HIP library's
  other objects.

### 5. Inventoried ownership for every model-dependent core test/tool

There are still **zero** `extension.json` manifests (models register in Cards 11-14), so
`-DGUFO_EXTENSIONS=…` exclusion is vacuous today; ownership must be explicit anyway.

- Produce the inventory of every test/tool/executable target registered **outside** the
  model directories (root `CMakeLists.txt` + `tests/{core,cli,quality,kernels,benchmarks,tools,testing,extensions,diagnostics}`)
  whose sources or link line touch a model/family library. Starting point (verified, root file):
  `gufo`, `gufo-kernel-bench`, `gufo_http`, `gufo_llm_cli`,
  `chat_template_hf_token_golden_test`, `exact_token_test`, `hipblaslt_runtime_cache_test`,
  `openai_chat_test`, `persistent_deltanet_test`, `text_model_runner_test`,
  `tune_hipblaslt`, the `qwen3_tts_*` root-registered tests, plus the H3/eval/compare ones
  you just touched (`minimax_h3_*`, `video_*`, `eval_*`). Re-derive it from the CMake
  files; do not trust this list blindly.
- Give each entry an **owner** (the model/family/support library manifest that lists it)
  or an explicit **condition** (`if(TARGET …)` on the owning library, with the reason),
  so that removing one model library cannot leave an unresolved reference in an
  unrelated tool. Move model-specific test registration into the owning model directory
  only where that is a pure move; `tests/functional/` stays put (AGENTS.md).
- Extend `tools/ci/check-extension-isolation.py` with a **new failing gate**
  (e.g. `--fail-on-target-ownership`) plus at least one fixture test: parse the CMake
  files for `add_executable`/`target_link_libraries` (handle the `${GUFO_QWEN_LIBRARIES}`
  indirection) and require (a) every model-dependent test/tool target is claimed by the
  owning manifest, (b) every claimed target exists, (c) no model source path is compiled
  outside its own directory except a recorded, justified condition. Register the new gate
  in the `cpu;repository` CTest group next to the existing isolation gates, and keep the
  existing `--fail-on-core-model-includes` / `--fail-on-family-graph` behaviour intact.
- Add `library.json` manifests for the new support libraries (`src/testing/compare`,
  `src/eval`) and the H3 + DeepSeek model libraries, with purpose and depends.

### 6. Gates to run (record each one's exact command, exit code and verdict line)

Static / host (cheap, run first and re-run after the last edit):

```sh
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
python3 tools/ci/check-extension-api.py --write && python3 tools/ci/check-extension-api.py
python3 tools/ci/check-docs.py
python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-family-graph
python3 tools/ci/check-extension-isolation.py --fail-on-target-ownership    # your new gate
```

Builds and tests (use `build/card08-*`; build affected targets first, then the full
`gufo` in `card08-gpu` and `card08-release`):

```sh
nix develop -c cmake -S . -B build/card08-gpu && nix develop -c cmake --build build/card08-gpu --parallel 8
nix develop -c cmake -S . -B build/card08-cpu && nix develop -c cmake --build build/card08-cpu --parallel 8   # exclusion case: HIP model libs absent
nix develop -c cmake -S . -B build/card08-release && nix develop -c cmake --build build/card08-release --parallel 4
nix develop -c ctest --test-dir build/card08-gpu -R '^(qwen27b_continuation_identity_test|qwen38_flash_next_continuation_identity_test|deepseek_v4_flash_continuation_identity_test)$' --output-on-failure
nix develop -c ctest --test-dir build/card08-gpu -R '^(minimax_h3_.*|h3_.*|video_.*|eval_.*|ds4.*|strix_video_help|check_.*|extension_.*|json.*|sampling_test|logit_sampler_test|openai_chat_test|logprobs_test|quote_tracker_test)$' --output-on-failure
nix develop -c ctest --test-dir build/card08-cpu  -R '^(minimax_h3_.*|eval_.*|ds4.*|video_.*|json.*|sampling_test|openai_chat_test)$' --output-on-failure
```

A `SKIP_RETURN_CODE 77` skip (missing weights/fixtures) is **PENDING evidence, not a
pass** — list the skipped names explicitly. H3 weights are absent under `./models`
today, so the `minimax_h3_*_hip_test` / denoiser / dit / vae oracle rows will skip:
record them in `pending-commands.md` with their exact commands.

Moved-HIP parity (the Card 07 tools are hard-coded to `src/models/qwen/`; generalize
them with an explicit prefix/`--model` argument whose default reproduces the Card 07
invocation, and re-run Card 07's command verbatim against
`build/card06-release`/`build/card07-release` to prove you did not break it):

```sh
python3 tools/prof/card07_parity.py --before build/card07-release/compile_commands.json \
  --after build/card08-release/compile_commands.json --mode flags \
  --prefix src/models/minimax_h3/ --scratch /tmp/card08-parity
nix develop -c python3 tools/prof/card07_parity.py … --mode device --scratch /tmp/card08-parity
nix develop -c python3 tools/prof/card07_host_parity.py --root $PWD \
  --before build/card07-release/compile_commands.json --after build/card08-release/compile_commands.json
nix develop -c python3 tools/ci/check-kernel-resources.py build/card08-release/gufo --compare build/card07-release/gufo
```

Expect `0 worse, 0 better, N same` for kernel resources; the pre-existing
`kernel_resources_test` 30-over-budget baseline drift is owned outside this card
(documented in Card 07 §8) — do not widen the baseline.

Not yours: `nix build .#checks.x86_64-linux.pr` (needs a committed tree — the
orchestrator runs it) and Windows native execution (no Windows host: record the exact
`BUILD-WINDOWS.md` commands as PENDING, like Card 07 §6).

## Evidence to write incrementally (as each gate completes — crash-resumable)

`docs/model-extensions/evidence/card08/`:
`README.md` (provenance: revisions, toolchain, build dirs, asset paths, host),
`plan-and-scope.md`, `configure-*.txt`, `build-*.txt`, `unit-*.txt`,
`flags-and-assertions.md` (old/new flags + NDEBUG/sanitizer per moved source),
`device-isa.txt`, `host-parity.txt`, `kernel-resources.txt`,
`target-ownership.md` (the inventory + owner/condition per target),
`nix-dataset-admission.txt`, `isolation.txt`, `pending-commands.md`,
`codex-review.txt` (orchestrator writes this one).

Also maintain `/tmp/card08-run/RESUME-NOTES.md` (copy it to
`docs/model-extensions/evidence/card08/RESUME-NOTES.md` when you hand off): what is
done, what is in flight, exact remaining commands, live build directories. If a build
dies, say so with its `/tmp` log path and verdict lines rather than guessing.

## Handoff (your final message + RESUME-NOTES.md)

Explicitly state: files changed (grouped by work item), each gate with command + exit +
verdict, evidence paths, what is PENDING with its command, limitations and risks, any
live job handles, and the sentence "editing ownership released". Do not commit.
