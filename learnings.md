# Learnings — M004-C007 (issue #35): AIE2P W4A8 SHQ4-T16 GEMM/GEMV

Live log of steps, discoveries, and gotchas while implementing
`long-horizon-plan.md`. Updated at each step.

## Operator notification convention

Notify operator via `curl -d "<message>" https://ntfy.fedeizzo.dev/alerts`
when: (1) stalled / blocked / long step without progress, (2) work finished,
(3) operator intervention needed, and (4) **every ~10 minutes** with a super
short recap of the current step. Implementation: check the epoch marker file
`/tmp/m004-notify-last`; if ≥600 s since last send, send one line and refresh
the marker. Keep messages terse (≤ 200 chars).

## 2026-08-20 — Resumed work (plan steps 1–3)

### Step 1: Inspection (files read on resume)

- Prior worker (stopped pass) left untracked
  `src/core/xdna2/programs/qwen_aie2p_w4a8/{gemm.cc,qwen_aie2p_w4a8.py}` +
  maki note `design.md`; P1 added `src/core/xdna2/qwen_aie2p_w4a8_pack.hpp` +
  `tests/kernels/qwen38_27b/test_aie2p_w4a8.cpp`. Pattern to copy, never
  modify: `qwen_mtp_eh_proj` (#83) — `qwen_mtp_eh_proj.*`, its `.devops/nix`
  package, CMake wiring, device test.
- Build: CMake gates XRT under `ENGINE_ENABLE_XRT`; program dirs from env
  `STRIX_AIE_*_PROGRAM_DIR` + `*_ROOT/include`; `scope.nix` + `flake.nix`
  devShells `default`/`aie`.
- 27B MTP tensors: eh_proj [5120,10240] Q4_K, ffn_gate/up [17408,5120],
  ffn_down [5120,17408]; hidden 5120, intermediate 17408.
- SHQ4-T16 normative packing in `tools/strix/shq.py` (verified by
  `tests/tools/test_shq_conformance.py`): codes `[nt][kg][k16][lane][kp8]`,
  BF16 scales, packed-zero u8/2 lanes, U4Z zeros, S4 no zero plane. Dynamic A8
  (`docs/QUANTIZATION.md` §318–367): amax/127, rne, clamp -127..127, no -128,
  INT32 sums exact, FP32 partials.
- Status: `gh` authed, `jj 0.44.0`, nix 2.34.8, server built; `models/qwen38_27b/npu/aie2p/` `.gitkeep` only.

### Step 2: Assumptions/risks

- M upper bound **128** (32 rounds) per design note; fallback 64 still OPEN.
- Q4_K → UINT4 nibble mapping lossless: `qs` byte = 2 lanes-in-group; groups
  paired (same byte, low/high nibble); 6-bit scales.
- P2 M-rounds accumulation bug: FIXED (Static-config XCLBIN below).

### Step 3 (P1): CPU-side reference + packing — DONE

Deliverable: `src/core/xdna2/qwen_aie2p_w4a8_pack.hpp` (header-only, no XRT):

- Record contract (matches `gemm.cc:11-23`): weight **3072 B** = codes 2048
  (8g×2 half-tiles × 16k × 16 lanes uint4, `byte = k*8 + lane/2`, low nibble
  even lane) + scales 512 (BF16-rounded FP32) + zcorr 512; input **1280 B** =
  codes 1024 + scales 128 (4×8 FP32) + INT32 sums 128 (asserts `pack.hpp:82-83`).
- Activation codes offset **`(group*2+half)*64 + row*16 + k`**
  (= `g*128 + h*64 + r*16 + k`, `pack.hpp:19-20`): group-1 half 0 starts at
  byte **128, not 64**; byte 64 = group-0 half 1, zero after a zero group 0
  (`test_aie2p_w4a8.cpp:717-719`).
- Output: padded row-major grid (`rounds·4` × `n_tiles·16` cols), padded/idle lanes stay 0 (`pack.hpp:567-569`).
- Mode packers: `PackQ4KBlockToRecord` (wscale = d·scale6, zcorr = dmin·min6), `PackShq4BlockToRecord` (U4Z/S4; S4 zcorr = 8.0 — `dot_u - 8·asum == dot_s`).
- Activation: group-32 dynamic INT8, scale stored BF16-RNE-rounded FP32
  (matches shq.py `_bf16r`), INT32 sums, tail-K zeros.
- `ReferenceBlockAccumulate` mirrors `gemm.cc` exactly; `ReferenceGemm` is the
  wide oracle (padded tail M rows quantize to zero via zero-guard rows).
- Gotcha: `Fp16ToFloat` in `ggml_dequant.hpp` is FP16, not BF16 (Q4_K d/dmin
  are FP16) → own BF16 RNE `RoundToBf16Rne` (ties-to-even, matches `shq.py`
  `f32_to_bf16_uint16`).

### CPU oracle gates — ALL PASS (10/10)

Build/run (no Nix wiring yet): `nix develop -c g++ -std=c++23 -O2 -I.
-I/tmp/stub_inc tests/kernels/qwen38_27b/test_aie2p_w4a8.cpp src/core/quant/ggml_dequant.cpp -o t && ./t`;
host-driver header stubbed by `/tmp/stub_inc/src/core/xdna2/qwen_aie2p_w4a8.h`
(`#pragma once`, `test_aie2p_w4a8.cpp:40`). Gates (`test_aie2p_w4a8.cpp:753-765`): BF16 RNE;
uint4 nibble decode + record round trip; SHQ4 U4Z/S4 → record; INT32 vs int64
oracle; Q4_K lossless vs `DotProductQ4_K` (EXACT, diff 0); dynamic A8 (amax
0/127, no -128); tail-K float-exact; GEMV row 0 == GEMM row 0; group boundaries (K=33).

### Gate bug history (fixed in this order: Q4K → dynamic-A8 → boundaries)

- **BIG ONE — tile-major vs row-major output bug** (`pack.hpp:582-631`):
  `ReferenceBlockAccumulate` ADDS into output; old code wrote tile-major
  within a round vs the padded-row-major contract, so (row0,tile1)/(row1,tile0)
  aliased the same 16 slots → out[16..31] summed two rows (209002 vs 153127);
  only n>16 gates failed. Fix: local 4×16 `tile_acc` per tile, scatter
  row-major, padded lanes stay 0. Lesson: keep the accumulate-into-output
  helper (AIE mirror `gemm.cc`) untouched — stage in a local buffer.
- DynamicA8 zero-group: amax==0 → 0/0 → NaN → `(int)NaN` UB → garbage
  codes/sums. Fix (`pack.hpp:483-499`): guard `active && lane < k` → scale 0,
  codes 0, sum 0.
- Test-data unsigned wrap: `(int)lane * (r + 1) % 254 - 127` — uint32
  `(r+1)` wraps to ~2^32 → scale ~3.4e7. Fix: `static_cast<int>(r + 1)`
  (`test_aie2p_w4a8.cpp:576-579`); keep the intended -126..-3 code domain.
- Harness trap: test aborts at FIRST failure — fix one gate, re-run to unmask the next (Q4K → dynamic-A8 → group boundaries).

### Static-config XCLBIN semantics (P2 program)

- (blocks, tpc, rounds) baked per XCLBIN (`qwen_aie2p_w4a8.py:275-298`);
  real shapes (20,10),(40,20),(20,34),(68,10) need no N padding; M-bucket 128
  (32 rounds) vs 64 still OPEN.
- rounds>1 needs per-round accumulators sized **ROUNDS·8·TPC·256** floats
  (`qwen_aie2p_w4a8.py:17-21, 82, 213`).
- P2 bug FIXED: one 256-float buffer per core → `acquire(ROUNDS)` + per-chunk
  `zero_fn`/`gemv_fn` (`qwen_aie2p_w4a8.py:164-183`).

### Behavioral gotchas

- IRON `.acquire(N)`: ONE memref for N=1, LIST of per-round memrefs for N>1
  (`qwen_aie2p_w4a8.py:166-173`) — per-round accumulators; guard `isinstance`.
- Q4_K group boundary: group-1 codes at byte 128, not 64 — gate asserts the
  127 code at `2·kMmulActivationElements` (`test_aie2p_w4a8.cpp:712-718`); an
  early version wrongly probed byte 64.
- BF16 RNE ties-to-even: see Step 3 gotcha above.

### Hardware / process

- Compile xclbin (aie devShell `flake.nix:91`), verified rounds=1 and 2:
  `nix develop .#aie -c python src/core/xdna2/programs/qwen_aie2p_w4a8/qwen_aie2p_w4a8.py --output-dir DIR --blocks N --tpc N --rounds N`
- Hardware gate = SKIP (`test_aie2p_w4a8.cpp:728-747`) until host driver `qwen_aie2p_w4a8.{h,cpp}` lands.

### Next steps

- P2: host driver `qwen_aie2p_w4a8.{h,cpp}` (pattern: eh_proj, zero changes
  there); Nix package + scope/flake; CMake wiring; hardware smoke M=1.