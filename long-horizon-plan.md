# Long-Horizon Plan

## Objective

Complete GitHub issue #35 (`M004-C007: Implement AIE2P W4A8 SHQ4-T16 GEMM and
independent GEMV measurements`) on the Qwen3.8-27B model: a general,
self-contained W4A8 kernel family executing on the AIE2P (XDNA2) NPU with
native 4x16x16 microtiles, INT8 activations, INT4 weights, INT32
accumulation, and fused zero/activation/weight-scale epilogues — validated
against a wide CPU oracle across a reusable shape matrix, with GEMM and
batch-1 GEMV results explicitly separated and keyed by assigned AIE columns,
plus a measurement report covering activation packing, DMA (serialized vs
overlapped), program/configuration reuse, command and completion times, and
padding/lossless-backend-view costs.

## Context

- Relevant architecture:
  - Target: Linux x86-64, AMD Strix Halo (Ryzen AI MAX+ 395), gfx1151 GPU,
    XDNA2/AIE2P NPU (32 AIE tiles, 8 columns), ROCm 7.2.3, XRT 2.21.0,
    `npu.sbin 1.1.2.64/65`. Verified with `./result/bin/strix-server diagnose`
    (status [PASS]).
  - Q4_K weights map losslessly into native UINT4 tiles (code + scale
    packing); activations use dynamic group-32 INT8 quantization; compute is
    native AIE2P `4x16x16` W4A8 matrix instructions with INT32 accumulation
    and vector FP32 scale/zero correction.
  - Target weights are the 27B MTP module (`blk.64`):
    - `nextn.eh_proj` [5120, 10240] (identical shape already proven on the
      4B model by issue #83 / commit `b6a334e`)
    - `ffn_gate` and `ffn_up` [17408, 5120] (same shape, two tensors)
    - `ffn_down` [5120, 17408]
  - `token_embd` and `output` are Q3_K: not representable in UINT4 tiles, so
    out of W4A8 scope (document as a representation limit in the report, not
    a gap).
  - All real K/N are 16-aligned; tail/padding cases are synthetic.
  - Dependencies are all closed: #23 (CPU operator oracles), #26 (SHQ
    conformance vectors), #33 (minimal AIE2P program / XRT lifecycle).
- Relevant files:
  - `models/qwen35_08b` — not relevant; work targets `models/qwen38_27b`.
  - `src/core/xdna2/` — existing XDNA2 host code (`device.cpp`, `smoke.cpp`,
    `qwen_mtp_rmsnorm.*`, `qwen_mtp_eh_proj.*`). The eh_proj kernel belongs to
    issue #83 and must not be modified, refactored, or copied; existing
    AIE programs under `src/core/xdna2/programs/` may be read as a toolchain
    pattern only.
  - `models/qwen38_27b/npu/aie2p/` — currently empty (`.gitkeep` only);
    receives `w4a8/` (replaces the stale `qwen35_4b` paths in the issue).
  - `tests/models/qwen_cpu_oracles_test.cpp` — #23 CPU oracles (numeric
    reference).
  - `tools/strix/quality.py`, `tests/tools/test_shq_conformance.py` — #26
    byte-exact SHQ conformance vectors.
  - `tests/device/qwen_mtp_eh_proj_test.cpp` — #83 hardware test pattern.
  - `tools/strix-gguf.py` — GGUF inspector (weight shape/type discovery).
  - `models/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf` — 27B MTP weights.
  - `flake.nix`, `CMakeLists.txt`, `.devops/nix/*` — Nix build system.
- Known limitations:
  - At batch-1, the existing eh_proj NPU path (~1.20 ms warm) is slower than
    the GPU projection (~0.44 ms); this card produces evidence and a general
    kernel only — no decode/NPU promotion, no SHQ freeze.
  - NPU device nodes are created on demand by the XRT plugin; raw
    `/dev/xdna*` listing is not a valid availability check.
  - Build/tests must run through Nix only; `git add` before `nix build`.

## Deliverables

- [ ] Implementation: new W4A8 AIE2P program + host driver (new files only)
- [ ] `models/qwen38_27b/npu/aie2p/w4a8/metadata.json`: binds architecture,
      compiler, Qwen implementation, tensor contract, shape buckets, content
      hashes
- [ ] Automated tests: `tests/kernels/qwen38_27b/test_aie2p_w4a8.cpp`
      (CPU oracle gates + hardware tests)
- [ ] `tools/testing/qwen_aie2p_w4a8_shq4_matrix_bench.cpp` matrix bench
- [ ] `artifacts/m004/c007-xdna2.json` + microtile/epilogue conformance
      report (embedded W4A8 program/metadata hashes)
- [ ] Documentation or configuration updates: CMake/test wiring, Nix flake
      scope if new inputs are required
- [ ] Final completion report: update issue #35 via `gh` (status comment with
      files changed, exact commands + results, artifact paths/hashes, skipped
      checks, residual risks)

## Constraints

The worker must:

- Follow existing project conventions (Conventional Commits, single-line
  messages: `feat(scope): summary (#35)`).
- Keep secrets and runtime state out of source control.
- Avoid unrelated refactoring.
- Build and test with Nix only (`nix build`, `nix develop`); no direct host
  builds or Makefiles.
- `git add` (or `jj` equivalent) before `nix build`.
- Keep the implementation self-contained: zero changes to
  `src/core/xdna2/qwen_mtp_eh_proj.*` or any issue-#83 code.
- Measure honestly: never hide activation quantization, padding, repacking,
  configuration, or DMA cost; report serialized vs overlapped DMA/compute.

The worker must not:

- Push, publish, or deploy.
- Disable failing tests.
- Perform destructive version-control operations.
- Route decode to the NPU based on GEMM peak results.
- Freeze the SHQ format if the card is unavailable, failing, or not promoted.
- Expand beyond this card's scope (no NPU decode promotion, no arbitrary
  shape compilation, no concurrent GPU/NPU execution, no format freeze).

## Execution Plan

1. Inspect the relevant implementation and tests: XRT program toolchain
   pattern in `src/core/xdna2/programs/`, #23 oracles, #26 SHQ vectors,
   #83 hardware test gating (`STRIX_REQUIRE_XDNA2`).
2. Record assumptions and risks (see Stop Conditions and the open M-bucket
   question below).
3. P1 — CPU side: W4A8 reference kernel + packing/epilogue logic + CPU
   oracle tests (no NPU): dynamic group-32 INT8 activations, Q4_K lossless
   UINT4 weight view (U4Z vs S4 as declared), INT32 accumulation, zero
   correction, activation scale, BF16 weight scale, output conversion.
4. P2 — AIE2P program + host driver; hardware smoke on one shape
   (`M=1` eh_proj, real 27B weights) against the CPU oracle.
5. P3 — full shape matrix on hardware: real shapes (3) x M buckets
   {1, 2, 4, 8, 16, 32, 64, 128}, tail-K/tail-N (±15 on 16-multiple bases),
   every-nibble weight vectors (from #26 vectors), group boundaries,
   active-row masks.
6. P4 — GEMM/GEMV separation keyed by assigned AIE columns; DMA overlap
   evidence; bench report + artifacts + metadata + issue update.
7. Review the final diff for unrelated changes.

## Acceptance Criteria

- [ ] Dynamic INT8 A rows, SHQ4 U4Z/S4 weight interpretation as declared,
      INT32 accumulation, zero correction, activation scale, BF16 weight
      scale, and output conversion match a wide CPU oracle.
- [ ] Tests cover native 4x16x16 microtile ordering, every nibble, group
      boundaries, padded/tail K and N, active-row masks, and representative
      reusable M buckets.
- [ ] Large GEMM and batch-1 GEMV results are explicitly separated and keyed
      by assigned AIE columns.
- [ ] Report includes activation packing, DMA, configuration/program reuse,
      command, completion, padding, and lossless backend-view costs, plus
      serialized versus overlapped DMA/compute.
- [ ] Artifact metadata binds architecture, compiler, Qwen implementation,
      tensor contract, shape buckets, and content hashes.
- [ ] Existing behavior remains compatible (#83 kernel and all current tests
      untouched and passing).
- [ ] The final diff contains only relevant changes.

## Validation Commands

```bash
# Build (after git add)
git add -A
nix build
./result/bin/strix-server diagnose

# CPU oracle gates (XRT off)
nix develop -c cmake -S . -B build-m004-c007 -DENGINE_ENABLE_HIP=OFF -DENGINE_ENABLE_XRT=ON -DBUILD_TESTING=ON
nix develop -c cmake --build build-m004-c007
nix develop -c ctest --test-dir build-m004-c007 --output-on-failure -R qwen35_aie2p_w4a8

# Hardware gates (this board, XDNA2 required)
mkdir -p artifacts/m004
STRIX_ARTIFACT_DIR=artifacts/m004 STRIX_REQUIRE_XDNA2=1 nix develop -c ctest --preset npu-test --output-on-failure -R 'm004_c007_hardware'

# Matrix bench
STRIX_ARTIFACT_DIR=artifacts/m004 STRIX_REQUIRE_XDNA2=1 nix develop -c ./build-m004-c007/tools/qwen_aie2p_w4a8_shq4_matrix_bench

# Canonical PR gate
nix build .#checks.x86_64-linux.pr
```

Note: adapt test regex names (`qwen35_aie2p_w4a8`) to the final 27B naming
(`qwen38_27b` family) once test targets are created; the card's dev loop
predates the 27b rename.

## Autonomy Policy

The worker may inspect and edit project files, run validation, retry reversible
operations, and choose implementation details consistent with this plan.

The open M-bucket upper bound (128 assumed for speculative-verification
batches; 64 is the fallback) may be settled by the worker using existing MTP
batch configurations in the codebase as evidence; document the choice in the
completion report.

Request supervision when requirements conflict, credentials are needed, a
destructive operation appears necessary, or three materially different attempts
fail for the same reason.

## Stop Conditions

- Stop on the first unexplained CPU/AIE mismatch; retain the smallest
  failing tile and report it.
- Stop as blocked when required access is unavailable, acceptance criteria
  conflict with constraints, or further attempts would repeat a rejected
  approach.
- Do not hide activation quantization, padding, repacking, configuration, or
  DMA cost to make a result pass.

## Completion Report

Report the outcome, changed files, design decisions, validation results, remaining
risks, and every unsatisfied acceptance criterion. Update issue #35 via `gh`
with: files changed; exact commands run and their results; produced artifact
paths and hashes; any skipped hardware checks and why; residual risks or
follow-up cards without expanding this card's scope.

# Progress Report

*Written 2026-08-20 11:25 local, from the long-horizon worker's event log
(`~/.local/state/maki/long-horizon-worker/missions/7195ecc8756f4b5a92f07bc1c51b7b84/events.jsonl`),
its recorded design note (`maki memory design.md`), and the working tree. The
worker was stopped by the operator after ~30 minutes; the report below is the
record of its work. It must be read as an incomplete pass: P1 was not started.*

## 1. Steps done / pending (plan numbering)

| Plan step | State | Notes |
|---|---|---|
| 1. Inspect impl/tests | ✅ Done | See §3 for files read |
| 2. Record assumptions/risks | ✅ Done | Design note saved to maki memory; M-bucket settled (see §2) |
| 3. P1 CPU-side kernel + oracle tests | ❌ Not started | Worker jumped to P2 instead; no CPU oracle code written |
| 4. P2 AIE2P program + host driver | 🔶 In progress | AIE program written and compiles (small config); host driver `.h/.cpp` NOT written; hardware smoke NOT run |
| 5. P3 full shape matrix on hardware | ❌ Not started | |
| 6. P4 GEMM/GEMV separation, bench, artifacts, issue update | ❌ Not started | No artifacts, no metadata.json, no bench tool |
| 7. Review final diff | ❌ Not started | |

Deliverables present in the working tree (untracked in git, added to jj):
- `src/core/xdna2/programs/qwen_aie2p_w4a8/gemm.cc` (5355 B) — AIE2P W4A8 SHQ4-T16 GEMM microkernel
- `src/core/xdna2/programs/qwen_aie2p_w4a8/qwen_aie2p_w4a8.py` (10101 B) — IRON builder, one kernel ELF + per-config XCLBINs

Delivery directory `models/qwen38_27b/npu/aie2p/w4a8/` was **not** created.

## 2. Design decisions and discoveries

### 2.1 Kernel geometry (recorded in maki memory `design.md`)
- 8 columns × 4 rows = 32 AIE cores; T16 output lanes per core; native
  `mmul<4,16,16,int8,uint4,acc32>` with 4 distinct M rows (4x16x16 W4A8).
- Parameters baked per XCLBIN: `blocks = ceil(K/256)`,
  `tiles_per_core = ceil(N/(8*4*16)) = ceil(N/512)`, `rounds = ceil(M/4)` masked.
- Real-shape combos needed: `(blocks,tpc) ∈ {(20,10),(40,20),(20,34),(68,10)}`.
  Real shapes need **no N padding**: 5120/512=10, 10240/512=20, 17408/512=34.
- Program reuse: one XCLBIN per `(blocks,tpc,rounds)`; round variants `{1,2,4,8,16,32}`.

### 2.2 Record layouts (host packing contract, per gemm.cc header)
- **Weight record** per (n-tile = 16 lanes, k-block = 256):
  codes 2048 B (8 groups × 2 half-tiles × 16k × 16 lanes uint4,
  byte = k*8 + lane/2, low nibble = even lane) + scales 512 B
  (8g×16l FP32, BF16-width expanded) + zcorr 512 B (8g×16l FP32) = **3072 B**
  (192 B/lane — found by fixing numpy record type, see §5).
- **Input record** per (m-chunk = 4 rows, k-block): codes 1024 B (int8,
  broadcast over k) + scales 128 B + INT32 sums 128 B = **1280 B**.
  All columns share the same input per (chunk, block).
- **Output** per (tile, chunk): 4×16 FP32 = 256 B.

### 2.3 Epilogue (uniform across SHQ4 modes)
`acc[row][lane] += act_scale[row][g] * (wscale[g][lane] * dot_u[row][lane]
- zcorr[g][lane] * asum[row][g])`, where `dot_u` is INT32-exact over uint4 codes.
Mode mapping (host-side packing only, same kernel ELF):
- **u4z**: codes = uint4, zcorr = uint4 zero plane decoded to FP32, wscale = BF16→FP32.
- **s4**: codes stored unsigned as `u = s+8`, zcorr = 8.0, wscale = BF16
  (`dot_u - 8*asum == dot_s`).
- **q4k**: Q4_K lossless: codes = Q4_K nibbles, wscale = d*scale FP32 exact,
  zcorr = dmin*min FP32 exact.

### 2.4 Tiler / IRON semantics (empirically pinned)
- `IRON Worker` wraps its core function in a `while(true)` loop by default —
  this is how eh_proj re-processes multiple output tiles in one kernel.
- `TensorTiler2D` produces **per-column patterns** with tile-major / block-minor
  fill order (verified via scratch probes on the exact eh_proj dims
  `(5120,10240)`, tile `(64,256)`, groups `(10,40)`; 8 patterns).
- API exploration concluded: **one kernel ELF with per-config XCLBINs** is the
  right design (loop counts live in the Python graph, not the kernel).

### 2.5 Keys for columns assignment / GEMM-GEMV separation
`AieColumnKey`: 8 columns always; `"gemv"` when M == 1, `"gemm"` otherwise.

### 2.6 M-bucket decision
Codebase MTP batch evidence supports ≤ 128 → **M upper bound 128** (32 rounds
of 4); fallback 64 if compile volume is too large; to be documented in the
completion report.

### 2.7 Infrastructure plan
- Host driver `src/core/xdna2/qwen_aie2p_w4a8.{h,cpp}` (pattern: eh_proj).
- Nix: `.devops/nix/aie-qwen-aie2p-w4a8.nix`, `scope.nix`, `flake.nix` env
  `STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR/ROOT`.
- CMake: `strix_core` adds `qwen_aie2p_w4a8.cpp` under XRT; test
  `tests/kernels/qwen38_27b/test_aie2p_w4a8.cpp` (CPU gates label `cpu`;
  hardware gates label `npu`/`xdna2`/`xrt-hardware`);
  `tools/testing/qwen_aie2p_w4a8_shq4_matrix_bench.cpp`;
  `models/qwen38_27b/npu/aie2p/w4a8/metadata.json`;
  `artifacts/m004/c007-xdna2.json` (artifacts/ is gitignored).

## 3. Files read (with what was learned)

- `src/core/xdna2/programs/qwen_mtp_eh_proj/qwen_mtp_eh_proj.py` + `gemv.cc` —
  the #83 toolchain pattern: Python graph/compile driver + `.cc` kernel;
  record layouts, tiler calls, IRON usage.
- `src/core/xdna2/qwen_mtp_eh_proj.cpp` (624 L) — host driver pattern.
- `src/core/xdna2/device.h`, `device.cpp` — XRT device lifecycle / XDNA2 gating.
- `tests/device/qwen_mtp_eh_proj_test.cpp` — #83 hardware test pattern.
- `CMakeLists.txt`, `CMakePresets.json` — build wiring; `strix_core` sources;
  eh_proj artifact dir CACHE var; XRT gating.
- `.devops/nix/aie-qwen-mtp-eh-proj.nix`, `aie-smoke.nix`, `scope.nix`,
  `flake.nix` — Nix packaging of AIE programs (aiebu + llvm-aie + mlir-aie +
  xrt inputs, `nix develop .#aie` devshell).
- `tools/strix/shq.py`, `conformance.py`, `quality.py`,
  `tests/tools/test_shq_conformance.py`, `docs/QUANTIZATION.md` — SHQ4-T16 v1
  byte-exact packing contract (U4Z G64 layouts, nibble order, BF16 scale
  rounding, zero groups, padded K/N tails) that the W4A8 packing must match.
- `src/core/speculative/qwen_mtp_reference.hpp` — oracle/reference structure.
- `src/core/quant/ggml_dequant.hpp` — Q4_K dequant (lossless UINT4 mapping).
- `tests/models/qwen_cpu_oracles_test.cpp` — #23 CPU oracles.
- `src/main.cpp` (grep) — `STRIX_REQUIRE_XDNA2` env handling.
- mlir-aie 1.4.1 (Nix store, read-only): `aie/iron/worker.py`,
  `mlir_aie/python/aie/helpers/taplib/*` (`tap.py`, `tensortiler2d.py`),
  `include/aie_api/{vector.hpp,utils.hpp,aie.hpp,detail/mmul.hpp}`,
  `aie2p` intrinsics tree — tiler + mmul + extract/insert/select epilogue API.

## 4. Commands run and key results

- `gh auth status` → logged in (fedeizzo), active; `gh issue view 35` read.
- `jj 0.44.0` available; note "Git tree dirty" warnings on every `nix` run —
  jj working copy vs. git index; `git add -A` performed before builds.
- `git add -A && nix build` → **build OK** (default package).
- `./result/bin/strix-server diagnose` → **status [PASS]** (hardware present:
  Ryzen AI MAX+ 395, unified memory, XDNA2; matches plan expectations;
  GPU name prefix "A..." — AMD Radeon 8060S).
- `nix develop .#aie -c python src/core/xdna2/programs/qwen_aie2p_w4a8/qwen_aie2p_w4a8.py
  --output-dir /tmp/w4a8-test --blocks 2 --tpc 1 --rounds 1` → **compiled**
  (`qwen_aie2p_w4a8.xclbin`, `.pdi`, `.insts.elf`, `.insts.bin`; memory map
  printed: `weight_l2l1_*` 49152 B, `input_l2l1_*` 1280 B, etc.).
- `... --output-dir /tmp/w4a8-real --blocks 40 --tpc ...` (real scale) → build
  in flight (`/tmp/w4a8-real/qwen_aie2p_w4a8.prj` created 11:19) when stopped.
- Scratch tiler probes `/tmp/tap_probe.py` under `nix develop .#aie` — see §5.

## 5. Failed attempts and what they taught

1. `batch` tool with `tool_calls` param → rejected ("required, expected array");
   retried as two plain `Bash` calls. (Tool-shape issue, not design.)
2. Tiler probe crashes (3 iterations) — error: `TensorTiler2D` argument shape;
   worker had misread `WEIGHT_TILE_ROW_BYTES = INPUT_TILES *
   WEIGHT_ROW_CHUNK_BYTES = 10240`. Fixed dims → probe succeeded at 11:05:20
   ("weight patterns: 8", access_order (5120,10240), tile-major/block-minor).
3. First program compile → Traceback in the builder (record geometry):
   worker fixed `kGroupElements`/`kBlockElements` constants in `gemm.cc`
   (added `kGroupsPerBlock`), then re-ran.
4. Second compile → **success** after fixing `WEIGHT_RECORD_BYTES = 3072`
   (not 2048): discovered the fifo record is 3072 B total across 16 lanes
   (192 B/lane) — codes 2048 + scales 512 + zcorr 512; also fixed numpy types
   and `INPUT_RECORD_ROWS = 5`.
5. `memory write` failed once (`write error: No such file or directory`) — no
   state dir for the memory backend at that moment; retry wrote `design.md` OK.
6. Upstream LLM API: repeated 429 rate limits + StreamLake "SIGTERM received,
   graceful shutdown timeout" (400) at ~11:08–11:19; throttled throughput
   (1 response/~10 s) — this is why the worker was stopped: too slow to finish.

## 6. Assumptions and risks

- Real shapes need no N padding (5120/512=10, 10240/512=20, 17408/512=34);
  tail/padding cases are synthetic-only, per plan.
- Q4_K lossless mapping into UINT4 (code + scale packing) is assumed valid per
  docs/QUANTIZATION.md + #26 vectors; CPU oracle (P1) is the numerical proof.
- One kernel ELF + per-config XCLBINs keeps compile volume bounded; if
  32-round (M=128) combos blow up, drop to M=64 fallback.
- `while(true)` IRON Worker semantics must be verified against the actual
  hardware DMA behavior (P2 hardware smoke) — untested so far.
- **Risk:** the written `gemm.cc`/`.py` are untested beyond a small-config
  compile; no host driver, no oracle, no hardware execution. P1 (CPU oracle)
  was skipped — should be completed first on resume, per plan order.
- `models/qwen38_27b/npu/aie2p/w4a8/metadata.json`, bench, and issue #35
  update are entirely outstanding.

## 7. What to do next if resumed

1. **P1 first (plan order):** W4A8 CPU reference kernel (dynamic group-32 INT8
   activations, Q4_K→UINT4 lossless packing, U4Z/S4/Q4K modes, INT32 accum,
   zero/act/weight-scale epilogue) + oracle tests
   `tests/kernels/qwen38_27b/test_aie2p_w4a8.cpp` (CPU gates), using #26
   packing vectors and `qwen_mtp_reference`/`ggml_dequant` as references.
2. **P2:** write host driver `src/core/xdna2/qwen_aie2p_w4a8.{h,cpp}` (packing
   + XRT lifecycle, pattern: eh_proj, zero changes there), Nix package
   `aie-qwen-aie2p-w4a8.nix` + scope/flake wiring, CMake wiring; hardware
   smoke on M=1 eh_proj shape with real 27B weights vs. the CPU oracle
   (`STRIX_REQUIRE_XDNA2=1`).
3. **P3:** shape matrix (real shapes × M buckets {1..128}, tail-K/N ±15,
   every-nibble vectors, group boundaries, active-row masks).
4. **P4:** GEMM/GEMV separation keyed by assigned AIE columns, DMA overlap
   evidence, serialized vs overlapped DMA/compute timing, bench tool
   `tools/testing/qwen_aie2p_w4a8_shq4_matrix_bench.cpp`, artifacts
   `artifacts/m004/c007-xdna2.json`, `metadata.json`, completion report +
   issue #35 update via `gh`.
5. Re-run canonical gate `nix build .#checks.x86_64-linux.pr`; review diff for
   unrelated changes (none so far — only the two new program files).
