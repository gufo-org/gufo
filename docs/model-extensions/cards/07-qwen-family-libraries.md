# Card 07: Extract acyclic Qwen family libraries

Status: **complete** (functional/performance criteria explicitly deferred — see validation
criteria and execution record; the deferral is not a pass). RFC step and dependencies: **5; 06**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Qwen headers form cycles and reach model-specific execution policy. Target extraction must resolve the real graph instead of hiding it in a large family library. ModelConfig and TextModelPreset remain in core until text migration.

### Original scope

Extract acyclic Qwen family libraries (tokens with independent Python mirror, config/text/vision/kernels and any justified common layer), make GGUF config extraction CPU-only, extract Qwen 27B and relink CLI/kernel tools. Resolve header-graph unassigned pieces explicitly. Keep ModelConfig/TextModelPreset until text migration. Linux/Windows, unit, Qwen/Flash-Next functional/performance and moved-HIP ISA/resource gates.

## Implementation plan

1. Use Card 01 graph and src/models/qwen/AGENTS.md to assign tokens, config, text, vision and kernels under src/families/qwen with explicit manifests/dependencies.
2. Resolve token-to-vision RoPE, vision-to-GEMM, prefill-to-QwenExecutionPolicy and continuation_hooks edges. Assign gemm_route.hpp and quant_ops.hpp explicitly; justify an acyclic common layer or merged kernel/vision layer.
3. Move GGUF config extraction out of GgufReader into a free CPU-only qwen_config function and preserve config tests. Move Qwen 27B code into its own library and relink gufo_llm_cli and gufo-kernel-bench.
4. Update independent tools/gufo/control_tokens.py mirror and AGENTS links as needed; preserve per-source HIP flags, target includes/definitions and assertion/sanitizer settings.

## Definition of done

- [x] Family graph is acyclic, every inventoried header is assigned, and manifests declare direct family dependencies. Six libraries / seven direct edges; all 61 inventoried Qwen headers in exactly one library; `extension_family_graph_gate` plus 14 fixtures enforce declared == real edges, cycles, header ownership and the family→model ban (`../evidence/card07/isolation.txt`, `README.md`).
- [x] CPU config users do not pull HIP, and token fixture expectations remain independent from production definitions. `gufo_qwen_config`/`gufo_qwen_text` compile and pass from source with `ENGINE_ENABLE_HIP=OFF` (orchestrator fresh build, exit 0) and their command lines carry no HIP library, include, offload flag or define; the moved config test and the ChatML/HF goldens assert independent expectations (`../evidence/card07/cpu-tests.txt`).
- [ ] Qwen 27B/Flash-Next functional and performance behavior is qualified; moved HIP resources and production ISA match or any intentional non-layout change is separately justified and tested. (The moved-code half is proven: 84/84 production compile flags identical, 39/39 device translation units byte-identical in `.text`/`.rodata`/`.note`, host code equal apart from the CU-id/local-label allowance the comparator now enforces, 2692/2692 kernels with unchanged scratch. The functional/performance half is deferred by the 2026-10-31 user decision to the single post-Card-14 qualification window and stays visibly pending with its exact commands; this item is therefore not checked.)

## Validation criteria

- Build affected family/model/CLI/kernel-tool targets on Linux and native Windows; run CPU config/token and affected model unit tests, retaining actual build logs.
- Run affected Qwen 27B and Flash-Next modes with their existing numerical, HTTP and CLI speed controls; include continuation suites where the move touches that code and exact identity goldens.
- Audit compile_commands.json and CMakeCache.txt; run tools/ci/check-kernel-resources.py using its current CLI and compare stripped NDEBUG gfx1151 ISA for moved sources. Merely matching source text or successful compilation is insufficient.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Implementation: pi worker runs `9e06dcd6-816e-483b-b8fc-a40594fd215e` and `75493672-16ad-4502-8e33-56d11ecb586c` (both died before final output). Independent read-only review: `d593f6ec-4fe6-4732-abe0-8f12962ed6a0` (no gates executed). Integration: pi `codex-exec-writer` run `d603b25c-a99a-4ed6-95fc-171d4bf3bd54`. Closure and this commit: the orchestrator.

| Finding | Disposition and evidence |
| --- | --- |
| F1 | PENDING: absolute assets exist; target rerun denied by Nix socket, exit 1. Removed unsupported 96.30-second claim. `integration-reruns.txt` retains exact command/denial. |
| F2 | RESOLVED: salvaged original scripts, removed implementation and integration scratch; orchestrator directory preserved. Cleanup record in integration-reruns.txt. |
| F3 | RESOLVED: repo CLI tools `tools/prof/card07_parity.py` and `card07_host_parity.py`; original rough sources retained as original-*.txt. Regenerated flag/device/host verdicts. |
| F4 | RESOLVED: correct hip target/type=o extracts 39 device-bearing objects; all 117 .text/.rodata/.note sections byte-identical. Whole-section equality covers every kernel and all register/spill/LDS metadata, stronger than selected-kernel sampling. SHA256 table in device-isa.txt; historical kernel count/resource table in kernel-resources.txt. |
| F5 | RESOLVED by striking overclaim: gate validates manifest field shape and header dependency edges, not CMake declarations/links. CPU Flash-Next consequence documented in handoff.md; engine is HIP-only, CPU config/oracle remains supported. |
| F6 | RESOLVED: family-root code now attributed and rejected without manifest; root CMakeLists exempt; fixture added. |
| F7 | RESOLVED: eight skips explicitly named; initial selection distinguished from later target/MTP FAILs; pending asset commands retained. |
| F8 | PENDING: fresh CPU build denied by Nix socket; historical no-work result explicitly insufficient; exact rebuild and 16-test commands in cpu-tests.txt. |
| F9 | RESOLVED: untracked hrx-system tree absent; unchanged whole-tree docs scanner passes 260 files/1467 links. |
| F10 | RESOLVED: DFS emits real cycle, fixture checks exact path. Legacy GUFO_MODELS_QWEN guards retained as harmless unique identifiers; no API behavior changed. |

Acceptance audit: family gate passes six libraries/seven direct edges and inventories both Qwen trees; 14 fixture tests pass, including root assignment and actual cycle path. CPU HIP-free compile/link and independent token/identity fixtures remain evidenced in cpu-tests.txt, unit-identity-goldens.txt and original tests; fresh CPU compilation remains unresolved F8. Production flags match 84/84 after excluding only unused core executable discovery macros. Device sections match 39/39 (`device-isa.txt`); host verdict retained separately. Integration commands, exit codes and tooling failures are in `../evidence/card07/integration-reruns.txt`. API surface check/write pass six paths unchanged; integration touched no C++.

Orchestrator closure (2026-10-31): the orchestrator re-ran the whole pinned gate set itself
after the last edit — formatting (730 files), API write+check (6 paths unchanged), docs
(260 files / 1467 links, zero errors), isolation (core boundary + family graph 6 libraries /
7 edges acyclic), identity goldens 3/3 exact bytes, the 82-test affected selection 100% in
201 s, `qwen27b_target_test` Passed 91.67 s with absolute asset paths (**F1 resolved**; the
0.06 s FAIL was the relative path), a fresh CPU-only build of the config/text libraries plus
their tests exit 0 (**F8 resolved**), kernel resources `0 worse, 0 better, 2692 same`, flags
`compared=84 failures=0`, device `compared=39 failures=0` (117 sections identical) and host
`sources=84 unexplained-code-differences=0`. Two closure fixes were made in the new Card 07
tools: `card07_parity.py` now resolves the HIP-toolchain offload bundler (the dev-shell one
cannot decompress ROCm bundles) instead of dying with a traceback, and
`card07_host_parity.py` became a real gate — identical instruction counts and mnemonics per
function, with only the `__hip_cuid_*` and `.L` string-pool allowances, non-zero otherwise.
The orchestrator also re-derived the pre-existing `kernel_resources_test` failure on the
parent revision's own binary: the 30 over-budget rows match by scratch bytes, VGPR, spill and
kernel name, so it is toolchain drift owned outside this card, not a Card 07 regression, and
the baseline was not widened. Details, commands and exits: `../evidence/card07/orchestrator-gates.txt`.
This revision is the closing commit; nothing was pushed.

Functional/performance half of definition-of-done remains PENDING: no forbidden harness, timings, speed benchmark or hosted check was run during the card. All literal commands remain in `../evidence/card07/pending-commands.md`: functional matrix (§1), long-context/cache and prompt_lookup, matched per-request/phase controls (5% and 10 ms), standard speed benchmark, shipped lock-mode HIP artifact, Windows native, hosted check (§9), asset-blocked MTP/vision rows (§7). Pre-existing kernel_resources_test failure (30 over budget, unchanged on parent) is accepted by the orchestrator as toolchain drift outside this card (§8); no baseline was widened. F1 and F8 were closed by the orchestrator's own pinned-environment reruns above. The RFC hash remains 2ce2becbc0bf0f844702b624de6827c2bae6eba921ab8a1ea883659fa2f96448. Evidence files remain below 1 MB. No staged files; no protected paths edited by integration. Reviewer closure is still required.
