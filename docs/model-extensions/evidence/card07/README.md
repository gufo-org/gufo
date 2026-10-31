# Card 07 evidence index — Extract acyclic Qwen family libraries

Candidate: jj working copy `@` change `zwllvvnv` (commits move as the working copy is
edited; last observed `6b9217ec`), parent `@-` = `kppqwqnv 11af44f9`
"refactor(extensions): move qwen-specific speculation into model-owned code" (Card 06).
Nothing is committed or staged: this directory is written from the working copy only.
`docs/model-extensions/RFC.md` untouched and byte-identical:
`2ce2becbc0bf0f844702b624de6827c2bae6eba921ab8a1ea883659fa2f96448`.

Machine: AMD Ryzen AI MAX+ 395 (Strix Halo), gfx1151, 20 CUs, 125 GiB RAM
(112–113 GiB free before every GPU step; llama-swap services left running).
Toolchain: repo-pinned `nix develop` — GCC 16.2.0 host, `nixpkgs-AMD` clang 24.0.0
(ROCm/therock 10.1), ROCm 10.1.0, `CMAKE_HIP_ARCHITECTURES=gfx1151`.
Build directories (task-owned): `build/card07-gpu` (gpu-test), `build/card07-cpu`
(cpu-test), `build/card07-release` (production `release` preset, discovery mode).
Binary sha256: `build/card07-release/gufo`
`c4879b4f3026d636bcdaf003bb06b427308be70f4c09ddbdd2867f81d8c498d7`,
`build/card07-gpu/gufo` `84f25732b5963e9daba064d3f4ded7fdebe7fcc8de7976ca0bbe74778543f30c`,
`build/card07-cpu/gufo` `85cbde1fdcd90023e59d1911cc0e949b1dfd71713094378e64f80f7933da8959`.

Status (PENDING is never a pass; commands are the exact ones that were run):

| # | Validation item | Status | Evidence |
| --- | --- | --- | --- |
| 1 | gpu-test configure, cache recorded | PASS | `configure-gpu-test.txt` |
| 2 | gpu-test full build + incremental rebuild after formatting | PASS (exit 0 each) | `build-gpu-test.txt` |
| 3 | Identity goldens (Card 02), exact bytes, zero skips | PASS 3/3 | `unit-identity-goldens.txt` |
| 4 | Affected unit selection (82 tests: qwen family/model, gates, GGUF, docs) | PASS 82/82, 8 asset-gated skips | `unit-affected-tests.txt` |
| 4b | Model-backed GPU units on real 27B + DFlash2: `qwen27b_target_test`, `qwen_dflash_gpu_test`, `qwen_dflash_upload_lifetime_test`, `qwen_vision_prepare_test` | PASS — orchestrator ran `qwen27b_target_test` with absolute assets (91.67 s, exit 0, F1 resolved); the others are 7.33 s / 3.17 s / exit 0 | `unit-affected-tests.txt`, `orchestrator-gates.txt` |
| 4c | CLI behaviour through the relinked libraries: `qwen27b.cli-options`, `cli_test.py --scope sampling` (48 generations, AR/fixed/adaptive) | PASS | `unit-affected-tests.txt` |
| 5 | CPU-only build (`ENGINE_ENABLE_HIP=OFF`) of `gufo_qwen_config`/`gufo_qwen_text` + 16 CPU tests; config/text compile lines carry no HIP library, include, offload flag or define | PASS | `cpu-tests.txt` |
| 6 | Production NDEBUG build + cache/`compile_commands.json` audit + per-source option survival + target-level context | PASS | `build-audit.md` |
| 7 | Host ISA audit vs the retained pre-card production objects (84 sources; 2438/2461 host functions byte-equal; 23 differences all label-relative `__FILE__`/`__hip_cuid` encodings; nothing unexplained) | PASS | `isa-audit.txt` |
| 7b | Kernel scratch parity: 2692 kernels, `0 worse, 0 better, 0 one side only` (release and gpu-test binaries) | PASS | `kernel-resources.txt` |
| 7c | Registered `kernel_resources_test` vs `tools/ci/kernel-resources.json` | FAIL — PRE-EXISTING: identical "30 over budget" list byte-for-byte on the parent revision's own binary | `kernel-resources.txt` |
| 8 | `check-format.py` | PASS after `--fix` (39 card-owned files; nothing outside the card diff) | `check-format.txt` |
| 9 | `check-extension-api.py --write` + check, SURFACE unchanged (6 paths) | PASS → no ext-api label or version bump needed | `check-extension-api.txt` |
| 10 | Isolation: `--fail-on-core-model-includes --fail-on-family-graph` + interim edge list | PASS | `isolation.txt` |
| 10b | Family gate + fixtures (`tests/tools/check_extension_isolation_test.py`, 14 tests) | PASS | `isolation.txt`, `handoff.md` |
| 10c | `check-docs.py` | PASS (260 files / 1467 links, zero errors; 2 self-inflicted broken links found and fixed) | `docs-check.txt` |
| 11 | `gufo diagnose` on production and gpu-test binaries | PASS (exit 0 both) | `diagnose.txt` |
| 12 | Windows native build/run of the affected targets | PENDING — no Windows host here | `pending-commands.md` |
| 13 | Shipped lock-mode HIP production artifact (`-DGUFO_EXTENSIONS_FROM_LOCK=ON`) | PENDING | `pending-commands.md` |
| 14 | Qwen 27B / Flash-Next functional profiles, `long-context`/`cache`, `prompt_lookup.py`, matched per-request/phase timings, standard speed benchmark | PENDING — deferred by the 2026-10-31 user decision to the single post-Card-14 qualification window | `pending-commands.md` |
| 15 | Hosted checks `nix build .#checks.x86_64-linux.pr` | see `orchestrator-gates.txt` (orchestrator runs it last, after every edit) | `pending-commands.md` |
| 16 | Orchestrator closure re-run of the whole pinned set (identity, 82-test selection, target unit, CPU fresh build + tests, kernel resources, flags/device/host parity, format, API, docs, isolation) | PASS, every exit 0; two closure fixes to the new Card 07 tools recorded there | `orchestrator-gates.txt` |

## Library layout and header assignment (definition of done, item 1)

Every one of the 61 headers the isolation scanner inventories across the two Qwen trees is
assigned to exactly one library, and each library's `library.json` manifest declares its
direct family dependencies (enforced, not prose):

| Target | Kind | Headers | Sources | Purpose |
| --- | --- | --- | --- | --- |
| `gufo_qwen_control_tokens` | INTERFACE | 1 | 0 | shared control-token spellings (Python mirror `tools/gufo/control_tokens.py`) |
| `gufo_qwen_hip_common` | INTERFACE | 7 | 0 | continuation hooks, execution/attention policy, recurrent state, GEMM routing, quant ops, RoPE layout |
| `gufo_qwen_config` | STATIC (CPU) | 1 | 1 | `ExtractQwenConfig` free function (was `GgufReader::ExtractModelConfig`) |
| `gufo_qwen_text` | STATIC (CPU) | 2 | 2 | tokenizer + chat template (+ pinned `reference/chat_template.jinja`) |
| `gufo_qwen_kernels` | STATIC (HIP) | 16 | 38 | shared HIP operators (`ops/*` launcher headers + kernel TUs) |
| `gufo_qwen_vision` | STATIC (HIP) | 4 | 4 | vision encoder, device input, image prompt, image cache |
| `gufo_qwen27b` (`src/models/qwen/`) | STATIC (HIP) | 30 | 40 | Qwen 27B target inference, MTP, DFlash2, continuation adapter |

`src/models/qwen/` stays the 27B model directory for this card; giving it its own
`extension.json` home is Card 20 work (recorded in `src/models/qwen/AGENTS.md`).
`ModelConfig` and `TextModelPreset` stayed in `src/core/` as required.

Cycle resolution actually applied (the RFC's provisional guesses became this):

- `vision/rope.hpp` → `hip_common` (RoPE layout is included by `kernels/ops/token.hpp`,
  `kernels/ops/attention.hpp` and the vision encoder, so it sits below both);
- `hip/detail/recurrent_state.hpp` → `hip_common` (three kernel TUs include it);
- `continuation_hooks.hpp`, `gemm_route.hpp`, `hip/quant_ops.hpp`,
  `hip/execution_policy.hpp`, `hip/detail/attention_policy.hpp` → `hip_common`;
- `vision/encoder.hip` → `kernels/ops/gemm.hpp` becomes a declared `vision → kernels` edge;
- family code may never include a model header — the gate rejects it (fixture covered).

Declared == real, acyclic, verified by `extension_family_graph_gate`
(`add_test` in `CMakeLists.txt`, listed in `gufo_pr_tests`):

```
qwen/control_tokens (no deps)      qwen/config -> (core only)
qwen/hip_common -> control_tokens  qwen/text -> control_tokens, config
qwen/kernels -> hip_common         qwen/vision -> control_tokens, hip_common, kernels, text
violations: []     libraries: 6     edges: 7
```

Interim edges (allowed by the RFC between steps 5 and 6, removed model by model later) and
the full cross-owner counts: `isolation.txt`. `gufo_core` no longer compiles any Qwen source.

## Remaining limitations

- Card 07 gates on unit/identity/CPU/hosted-shaped/static evidence only. All functional
  and performance qualification stays pending in the post-Card-14 window
  (`pending-commands.md`); no `run.py`, `prompt_lookup.py`, `compare.py`, `model-bench.py`
  or devlab invocation was made, and none is claimed.
- Windows native builds were not executed (no Windows host on this machine).
- Shipped lock-mode artifact not built/tested (`extensions.lock` is still empty, so lock
  mode currently selects nothing; the gate is listed rather than silently skipped).
- 27B MTP asset missing under `./models`: `qwen_mtp_gpu_test` fails with
  `Missing Qwen MTP tensor: token_embd.weight` and `qwen27b.cli` (full scope) skips 77.
  These are asset-pending, not passes.
- `kernel_resources_test` fails against the stale `tools/ci/kernel-resources.json` on this
  toolchain; reproduced byte-identically on the parent revision's binary, and Card 07
  changes no kernel's scratch (7b). Orchestrator disposition: pre-existing toolchain drift,
  out of Card 07 scope — the baseline is not widened here and the refresh/spill fix is
  tracked for the post-Card-14 qualification window (`pending-commands.md`), where the
  hosted check result is reported rather than hidden.
- 23 host functions differ by label-relative encodings only (longer `__FILE__` strings,
  per-TU `__hip_cuid_*` identifiers, including as call targets inside otherwise identical
  functions); `tools/prof/card07_host_parity.py` now enforces exactly that allowance and
  fails on anything else (`host-parity.txt`); device .text/.rodata/.note parity is now proven for all 39 device-bearing translation units; see `device-isa.txt` (whole-section equality includes every kernel and resource descriptor).

Integration corrections: G4b target is PENDING, never PASS. G7 full device section parity: 39 objects, 117 section comparisons identical (device-isa.txt). Manifest target/kind validation checks field shape only; CMake target/kind/link edges are NOT cross-checked by this lexical gate.

Integration F8: fresh CPU compilation proof remains PENDING (Nix daemon denied); historical 16-test result does not satisfy that missing build evidence. See integration-reruns.txt.

Orchestrator closure (2026-10-31): F1 and F8 are RESOLVED by the orchestrator's own pinned-
environment runs above; the host comparator is now a failing gate rather than a report, and
device-section parity reproduces without PATH manipulation. Every number in the status table
rows 1–11 and 16 comes from a command in `orchestrator-gates.txt` or a per-gate file; rows
12–15 remain PENDING and are not passes.
