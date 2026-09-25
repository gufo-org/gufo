# ROCm 10 Spike — Build Works, Runtime Does Not

Date: 2026-09-04. Companion to `baseline.md` (ROCm 7.2.3 reference numbers).

Option 1 from the integration plan: build gufo against the ROCm 10 TheRock
`gfx1151` distribution outside Nix, to decide whether packaging ROCm 10 into
`.devops/nix` is worth doing.

Outcome: **the repository compiles and links cleanly against ROCm 10, but the
ROCm 10 HIP runtime cannot create a stream on this host, so nothing runs.**
No performance comparison was possible.

## What was tested

| Item | Value |
| --- | --- |
| Stable dist | `therock-dist-linux-gfx1151-10.0.0.tar.gz`, 1.79 GB, 8.4 GB extracted |
| Stable hash | `sha256-T+q9ny2nI1LfN/bXFKVIR9P+kTwDQfviplQsEWQCS68=` |
| Nightly dist | `therock-dist-linux-gfx1151-10.1.0a20260904.tar.gz` |
| Nightly hash | `sha256-Kpaha5XNzUl871SIP0nu6PVb9WenVR6uS4XG9YbaF2U=` |
| Device compiler | AMD clang 23.0.0git (dist `llvm/bin/amdclang++`) |
| Host compiler | gcc 15.3.0 from `nix develop` |
| aotriton | nixpkgs `aotriton-0.11.1b` (not shipped by TheRock) |

## What worked

- **Full build, zero source changes.** 199/199 targets, including all 51 `.hip`
  files: the rocWMMA-heavy DeepSeek V4 Flash kernels, the Qwen hipBLASLt GEMM
  route, Composable Kernel attention, and both aotriton call sites (v2 in
  `src/models/minimax_h3/dit.hip`, v3 in
  `src/models/qwen/hip/kernels/attention_split.hip`). Warnings only.
- **Every ROCm dependency resolved into the ROCm 10 tree**: `hip`, `hipblas`,
  `hipblaslt`, `rocblas`, `miopen`, Composable Kernel, rocWMMA, hipCUB,
  rocPRIM, roctx. Verified through `CMakeCache.txt`.
- **Clean runtime linkage**, no ROCm 7.2.3 objects in the closure:
  `libamdhip64.so.7`, `librocblas.so.5`, `libhipblaslt.so.1`, `libMIOpen.so.1`,
  `libroctx64.so.4`, `libhiprtc.so.7`, all from the dist.
- **aotriton soname compatibility.** TheRock does not ship aotriton, but ROCm 10
  keeps the `libamdhip64.so.7` soname, so the nixpkgs build (`NEEDED:
  libamdhip64.so.7`) loads against the ROCm 10 runtime without patching.

## The blocker

`hipStreamCreate` fails on the first call:

```
ld.lld: error: undefined hidden symbol: __amd_streamOpsIncrement
ld.lld: error: undefined hidden symbol: __amd_streamOpsDecrement
Error: Creating the executable from LLVM IRs failed.
:3:hip_stream.cpp :333 : hipStreamCreate: Returned hipErrorOutOfMemory
```

At first stream creation the HIP runtime JIT-compiles its internal
blit/stream-ops OpenCL program through COMGR 3.3. The final
`AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE` step fails to resolve two
symbols, and HIP surfaces that as `hipErrorOutOfMemory` — a misleading error,
not memory pressure.

Minimal reproducer (`scratchpad/hip_probe.cpp`), no gufo code involved:

```
hipGetDeviceCount:  no error (n=1)
hipMemGetInfo:      no error free=124542 MiB total=126976 MiB
hipStreamCreate:    out of memory
hipMalloc 256MiB:   no error
```

Memory is plainly not the problem: 124 GiB free, and a 256 MiB allocation
succeeds immediately after the failure.

### Ruled out

- **Not a stale release.** The 10.1.0a20260904 nightly fails identically.
- **Not hardware or driver enumeration.** ROCm 10 `rocminfo` reports gfx1151
  (queue max 131072) and the aie2p NPU; HSA runtime 1.21 loads.
- **Not environment.** Identical failure under `env -i`, and with
  `GPU_MAX_HW_QUEUES=1`, `HSA_MAX_QUEUES=1`, `HSA_ENABLE_SDMA=0`,
  `AMD_DIRECT_DISPATCH=0`, `HSA_SCRATCH_SINGLE_LIMIT`, `HIP_DEVICE_LIB_PATH`,
  `DEVICE_LIB_PATH`, `ROCM_KPACK_PATH`, `ROCM_KPACK_DISABLE`, and with
  `lib/rocm_sysdeps/lib` added to `LD_LIBRARY_PATH`.
- **Not missing or mismatched device libraries.** COMGR extracts its embedded
  device libs to a temp tree and links `opencl.bc`; that copy is byte-identical
  to the dist's `amdgcn/bitcode/opencl.bc`, and both define all four symbols
  (`llvm-nm`: `W __amd_streamOpsIncrement`, `Decrement`, `Wait`, `Write`).
  Only `Increment` and `Decrement` fail to resolve; `Wait` and `Write` link.

The two failing symbols back the stream-ops APIs ROCm 10 added for CUDA parity
(`hipStreamWriteValue32/64`, `hipStreamWaitValue32/64`, `hipStreamBatchMemOp`).
ROCm 7.2.3's `opencl.bc` defines none of them, which is why 7.2.3 never takes
this path. This looks like an upstream defect in the gfx1151 TheRock
distribution, worth filing against ROCm/TheRock with the reproducer above.

## Update: nixpkgs-style packaging works, and gufo runs on ROCm 10

The tarball verdict below was a packaging problem, as suspected. Rebuilding the
ROCm 10 sources through nixpkgs' packaging fixes `hipStreamCreate`:

```
=== RUN (nixpkgs-built ROCm 10) ===
hipGetDeviceCount: no error (n=1)
hipStreamCreate:   no error          <- "out of memory" with AMD's tarball
```

Same machine, same kernel, same `therock-10.0` sources. The vendored module set
lives in `.devops/nix/rocm-modules-10/`; `scratchpad/rocm10-scope.nix` and
`scratchpad/gufo-rocm10.nix` instantiate it and build gufo against it.

### Measured: ROCm 10 is slower than 7.2.3 with the current tuning

Same sweep as `baseline.md` (UD-Q4_K_XL, `--n-prompt 2048 --n-gen 128
--n-depth 4096,8192,12288,16384 --repetitions 1`).

| Test | ROCm 7.2.3 | ROCm 10.0.0 | Delta |
| --- | --- | --- | --- |
| pp2048 @ d4096 | 447.15 | 373.50 | -16.5% |
| pp2048 @ d8192 | 426.25 | 355.68 | -16.6% |
| pp2048 @ d12288 | 410.09 | 335.85 | -18.1% |
| pp2048 @ d16384 | 392.43 | 317.44 | -19.1% |
| tg128 @ d4096 | 11.46 | 10.87 | -5.1% |
| tg128 @ d8192 | 11.26 | 10.55 | -6.3% |
| tg128 @ d12288 | 11.06 | 10.55 | -4.6% |
| tg128 @ d16384 | 10.86 | 10.39 | -4.3% |

Both runs are single-repetition, but a 17-19% prefill drop repeated across four
depths is not thermal drift.

This is not yet a verdict on ROCm 10. The hipBLASLt plan database
(`src/core/hip/hipblaslt_plan_database.cpp`, tuned with `tune_hipblaslt`) and the
pinned rocBLAS solution indices (`src/models/minimax_h3/dit.hip:53-88`) were both
selected against 7.2.3, and hipBLASLt moved to 1.4.1 and rocBLAS to 5.6.0. The
prefill path is where those tuned plans are used, and prefill is what regressed
most. Re-tuning against the new libraries is the next measurement, not a
speculative optimization.

### Where the prefill regression is, and one rejected fix

The Q4_K_XL benchmark never touches hipBLASLt. `gemm_route.hpp:231` routes a
quantized shard to `kHipPrefillQuantDirect`, our own WMMA kernel in
`src/models/qwen/hip/kernels/prefill_quant_gemm.hip`; only dense BF16 shards take
`kHipPrefillBf16LtTryThenBlas`. Two consequences: `tune_hipblaslt` cannot help
this benchmark, and the plan database is irrelevant here (it is opt-in through
`GUFO_HIPBLASLT_PLAN_CACHE` anyway, and was unset in both runs).

hipBLASLt itself is not the problem. `gufo-kernel-bench --kernel gemm --type bf16`
is within ~2% between the toolchains at every size (4096^3: 6199 vs 6141 GFLOP/s).

clang 23 does allocate very differently for our WMMA kernels:

| Kernel | 7.2.3 (clang 22) | 10.0 (clang 23) |
| --- | --- | --- |
| `W8A8BlockedWmmaGEMMKernel<128,128,2,4>` | 138 vgpr / 0 spill | 256 / 156 |
| `W8A8BlockedWmmaGEMMKernel<128,16,4,8>` | 138 / 0 | 200 / 0 |
| `W8A8BlockedWmmaGEMMKernel<128,32,4,4>` | 148 / 0 | 256 / 64 |
| `W8A8BlockedWmmaGEMMKernel<128,64,4,8>` | -- | 256 / 148 |

**Removing the spills does not help.** `-mllvm -amdgpu-sched-strategy=iterative-minreg`
drops every spill to zero (234/143/182/244 VGPRs, occupancy 5->6, 7->10, 5->8) and
measured *worse*: pp2048@d4096 373.5 -> 288.7, tg128@d4096 10.87 -> 8.80. That
reproduces the finding already recorded above the kernel template: it is
latency-bound, and the scheduling work needed to hit a register target costs more
than the spills. Note the flag was applied to every HIP TU, which is why decode
regressed too; but prefill lost 23% on its own.

So register pressure is a symptom, not the cause. Remaining hypotheses, in order:
rocWMMA 1.7 fragment codegen versus 7.2.3; instruction-count/ISA diff of the hot
inner loop between clang 22 and 23; and scheduling changes that hurt latency
hiding without changing register counts.

### Root cause found: clang 23 SLP vectorization, and the fix

Instruction counts are nearly identical between the toolchains (+-6-9%); the
entire ISA delta is spill traffic and waits:

| Kernel | metric | clang 22 | clang 23 |
| --- | --- | --- | --- |
| `W8A8...<128,128>` | scratch ops | 0 | 250 |
| | `s_waitcnt` | 124 | 194 |
| `W8A8...<128,64>` | scratch ops | 0 | 174 |
| | `s_waitcnt` | 97 | 165 |

Across the whole translation unit, 67 of 214 instantiations spill under clang 23
(8570 VGPRs total) against zero under clang 22, and 76 lose occupancy. The cause
is the SLP vectorizer widening the inner loops past the 256-VGPR budget:

| variant | spilling kernels | spill VGPRs | occupancy below clang 22 |
| --- | --- | --- | --- |
| clang 23 baseline | 67 | 8570 | 76 |
| `-mllvm -amdgpu-schedule-relaxed-occupancy` | 66 | 8543 | 76 |
| `-O2` | 67 | 8570 | 76 |
| `-mllvm -slp-threshold=20` | 25 | 2183 | 14 |
| **`-fno-slp-vectorize`** | **25** | **2183** | **14** |

Measured with `-fno-slp-vectorize` (same sweep):

| Test | 7.2.3 | ROCm 10 | ROCm 10 + no-SLP | vs 7.2.3 |
| --- | --- | --- | --- | --- |
| pp2048 @ d4096 | 447.15 | 373.50 | 461.78 | +3.3% |
| pp2048 @ d8192 | 426.25 | 355.68 | 436.48 | +2.4% |
| pp2048 @ d12288 | 410.09 | 335.85 | 409.25 | -0.2% |
| pp2048 @ d16384 | 392.43 | 317.44 | 375.11 | -4.4% |
| tg128 @ d4096 | 11.46 | 10.87 | 10.85 | -5.3% |
| tg128 @ d16384 | 10.86 | 10.39 | 10.37 | -4.5% |

Decode is unmoved by the flag (10.87 -> 10.85), so the remaining decode gap has a
different cause and the global flag is not hurting other kernels here. Two open
items: decode at about -5%, and deep-context prefill at -4.4%.

Reproduce the register comparison with `scratchpad/vgpr-probe2.sh <build-dir>`
and score a flag with `scratchpad/score.py <label> <flags...>`.

### Landed

`CMakeLists.txt` now disables SLP vectorization for
`src/models/qwen/hip/kernels/prefill_quant_gemm.hip`, gated on
`CMAKE_HIP_COMPILER_VERSION VERSION_GREATER_EQUAL 23`. The pinned 7.2.3
toolchain (clang 22) is unaffected by construction. Scoped to the one file, it
reproduces the global-flag measurement: pp2048@d4096 456.20 +- 1.27 against
456.68 +- 0.67, three repetitions each.

### Remaining gaps on ROCm 10 (3 repetitions, +- shown)

| Test | 7.2.3 | ROCm 10 (fixed) | Delta |
| --- | --- | --- | --- |
| pp2048 @ d4096 | 443.26 +- 0.16 | 456.20 +- 1.27 | +2.9% |
| pp2048 @ d16384 | 389.33 +- 0.69 | 375.53 +- 3.06 | -3.5% |
| tg128 @ d4096 | 11.47 +- 0.01 | 10.86 +- 0.00 | -5.3% |
| tg128 @ d16384 | 10.95 +- 0.00 | 10.39 +- 0.00 | -5.1% |

Both remaining gaps are localized with `gufo-kernel-bench`:

| Kernel case | 7.2.3 | ROCm 10 |
| --- | --- | --- |
| `gemv 1x4096x4096` | 47.41 us / 708 GB/s | 49.75 us / 675 GB/s |
| `decode_attention 4096 tok` | 115.4 us | 159.7 us |
| `decode_attention 8192/16384/32768 tok` | 406 / 718 / 1440 us | 401 / 738 / 1438 us |
| `gemm bf16` (hipBLASLt, all sizes) | -- | within 2% |

The decode GEMV is 4.7% slower and accounts for essentially all of the decode
regression. Its kernels show *no* register or occupancy change between the
toolchains (0 of 2 in `gemv_quant.hip`, 0 of 8 in `attention_decode.hip`), so
that one is scheduling or memory-clause behaviour, not pressure -- a separate
investigation from the prefill fix. `decode_attention` at 4096 tokens is the
other outlier and is not explained yet; the deeper cases are at parity.

**Net position: 7.2.3 remains the faster toolchain overall.** ROCm 10 wins
shallow prefill by ~3% and loses deep prefill by ~3.5% and decode by ~5%. The
pin should not move until the GEMV gap is closed.

### Cross-checked against llama.cpp's gfx1151 work

ggml-org/llama.cpp#21284 ("Inefficient defaults for gfx1151 cost substantial
performance for prefill") reports ~20% prefill uplift on Strix Halo from three
areas. Item 1 is *the same root cause found here independently*: "the current
HEAD likely spills and exceeds the 256 VGPR". Their remedy is retuned MMQ tile
parameters (x=48, y=64, nwarps=4); ours is disabling SLP for the affected TU.

Their other items, checked against gufo:

| llama.cpp item | Status here |
| --- | --- |
| `dp4a` -> `__builtin_amdgcn_sudot4` for RDNA 3.5 | Already done (`prefill_quant_gemm.hip:1068`) |
| `roundf()` -> `__float2int_rn` in quantize | **Applied** -- see below |
| `expf()` -> `__expf()` in MoE routing / SiLU | Not applied: no MoE routing here; the `expf` sites are softmax in `attention_batched.hip` and `sample.hip`, where the accuracy trade needs its own evaluation |
| Loop-invariant hoisting in `concat.cu` | No equivalent slow path |

**`roundf` -> `__float2int_rn` (applied).** The activation-quantization sites in
`prefill_quant_gemm.hip` compiled to 18 instructions (`v_trunc`/`v_sub`/`v_cmp`/
`v_cndmask`/`v_bfi` -- the half-away-from-zero fixup) against 10 for the
intrinsic (`v_rndne` + `v_cvt`).

| Test | 7.2.3 before | 7.2.3 after | ROCm 10 before | ROCm 10 after |
| --- | --- | --- | --- | --- |
| pp2048 @ d4096 | 443.26 +- 0.16 | 447.05 +- 1.28 | 456.20 +- 1.27 | 461.16 +- 1.16 |
| pp2048 @ d16384 | 389.33 +- 0.69 | 392.05 +- 0.93 | 375.53 +- 3.06 | 376.88 +- 4.09 |
| tg128 @ d4096 | 11.47 +- 0.01 | 11.50 +- 0.00 | 10.86 | 10.86 |

About +0.9% prefill on the supported toolchain and +1.1% on ROCm 10.

**It changes numerics** -- `roundf` rounds halves away from zero, the intrinsic
rounds halves to even -- so it is not a free swap in a codebase that pins rocBLAS
solutions for byte-identical output. Evidence gathered:

* `--validate-prefill 1024` agreement *improves*: `max_abs_diff` 0.55557811 ->
  0.41680527, `rmse` 0.09512630 -> 0.07941676, cosine 0.99946874 -> 0.99962676.
  Round-half-to-even is unbiased; half-away-from-zero inflates magnitudes.
* The same numbers appear on both toolchains, so the shift is deterministic.
* 7.2.3 and ROCm 10 without the change produce byte-identical validation
  numbers, confirming the toolchain swap alone changes nothing numerically.
* Greedy output on a sample prompt is unchanged (only the load-time line
  differed).

That is not a substitute for the repository's own gates. Run
`nix build .#checks.x86_64-linux.pr` before landing, since speculation here is
documented as greedy-faithful and this alters quantization tie-breaking.

### Further experiments (all negative, recorded so they are not repeated)

**CU mode (`-mcumode`).** Screening looked promising -- spilled VGPRs in the
prefill GEMM TU fell from 2183 to 703 -- but it measured clearly worse:
pp2048@d4096 414.69 +- 1.90 against 461.16 +- 1.16, and decode fell too. CU mode
halves the LDS and scheduler resources a workgroup can reach, which costs more
than the spills. Rejected.

**Prefill tile sweep.** `GUFO_KQUANT_PREFILL_TILE` selects six tile variants, all
tuned under clang 22. Re-swept on ROCm 10 (pp2048, 2 repetitions):

| tile | pp2048 |
| --- | --- |
| default (128x128, 256 threads) | **476.46 +- 0.83** |
| wide-occ | 429.75 +- 1.45 |
| fuse | 423.94 +- 0.82 |
| bk1 | 421.51 +- 2.15 |
| wide | 418.19 +- 1.39 |
| narrow | 379.33 +- 0.32 |

The existing default stays best by a wide margin, so the tile tuning survives a
compiler generation. This is the opposite of llama.cpp#21284, where the gfx1151
defaults were poorly chosen.

**`expf` -> `__expf`.** Ruled out by profiling rather than measured: `expf` costs
21 instructions against 8 for `__expf`, but `GUFO_PROFILE` shows prefill is 94%
GEMM (FFN 65%, input projection 24%) with no classic attention softmax on this
model's path at all -- Qwen3.8 is an SSM hybrid. The `expf` sites in
`attention_batched.hip` are cold here.

### Decode regression: localized, mechanism identified, not fixed

`rocprofv3 --kernel-trace` over `tg128` on both runtimes (note: profiling a
ROCm 10 process needs the ROCm 10 `rocprofv3`; the 7.2.3 one cannot attach):

| Kernel | 7.2.3 | ROCm 10 | Delta |
| --- | --- | --- | --- |
| `Wave32FusedQuantSwiGLUGEMVKernel_2Rows` | 4481.6 ms | 4671.7 ms | +4.2% |
| `Q8KBlockGEMVKernel_2Rows` | 3797.0 ms | 4060.8 ms | +6.9% |
| `Wave32FusedSSMInputProjectionsKernel` | 1664.1 ms | 1807.5 ms | +8.6% |
| `Wave32FusedQKVProjectionsKernel_1Row` | 504.2 ms | 540.9 ms | +7.3% |
| `DeltaNetRecurrenceKernel` | 227.4 ms | 204.5 ms | -10% |
| `WKQuantA8BlockedWmmaGEMMKernel` | 183.1 ms | 177.4 ms | -3% |

The whole regression sits in the GEMV family; the WMMA prefill kernel is now
*faster*, which is the SLP fix showing up. These GEMVs are unchanged in
occupancy (16 waves/SIMD, no spills, 52->56 / 65->72 VGPRs).

The mechanism is the byte-load form. `Q8_0Block` is 34 bytes with `qs` at offset
2, so the code bytes are only ever 2-byte aligned:

| build | byte loads in `Q8KBlockGEMVKernel_2Rows` |
| --- | --- |
| clang 22 | 16x `global_load_d16_u8` |
| clang 23 | 8x `global_load_d16_u8` + 8x `global_load_i8`, plus 10 extra `v_cmp_lt_u32` |

clang 23 emits *fewer* instructions overall (836 against 867) but spends a full
VGPR per byte on half the loads and lengthens the dependency chain, which is the
wrong trade for a latency-bound GEMV.

Not reachable so far: `-fno-slp-vectorize`, `-mllvm -amdgpu-hard-clause-length-limit=16`,
and `-mllvm -amdgpu-set-wave-priority` all leave the load mix unchanged, and
writing the halfword pairing explicitly in source (loading through
`const uint16_t*` and extracting both bytes, accumulation order preserved) is
canonicalized straight back to the same mix. That attempt was reverted. This
looks like a clang 23 regression worth reporting upstream with the reproducer.

## Consequences for the integration plan

- Packaging ROCm 10 into `.devops/nix` is **premature**. The Nix work is
  mechanical and the compile side is already proven; there is no point landing
  it while no ROCm 10 binary can execute on the target.
- Kernel-level optimization for ROCm 10 (`src/models/qwen/`) is **blocked** for
  the same reason: no way to measure a change.
- The 7.2.3 pin stays. `baseline.md` remains the reference point for whenever a
  working ROCm 10 dist appears.

## Notes for the eventual migration

- **ROCm version no longer equals HIP version.** The 10.0.0 dist ships HIP
  headers at `7.15.26333` and `libamdhip64.so.7`; only `rocm-core` reports
  `10.0.0`. `src/cli/diagnose/diagnose.cpp:505` prints `HIP_VERSION_MAJOR/
  MINOR/PATCH` as "ROCm version" and would report `7.15.26333`. It must read
  `rocm-core` instead. Same for the `"7.2.3"` defaults in
  `src/core/diagnostics/fingerprint.h:31` and
  `src/core/diagnostics/system_inventory.h:64`.
- **Component versions are independent now**: hipBLASLt 1.4.1, rocBLAS 5.6.0,
  hipCUB/rocPRIM 4.6.0, MIOpen 3.6.0, Composable Kernel 1.2.0. Any Nix
  expression keyed on a single ROCm version string needs restructuring.
- **hipCUB 4.6.0 removals do not affect this repo**: only
  `hipcub::DeviceRadixSort` is used, which survives.
- **Still unvalidated** (blocked behind the runtime): the pinned rocBLAS
  solution indices in `src/models/minimax_h3/dit.hip:53-88` and the hipBLASLt
  plan database, both tuned against 7.2.3 and expected to need re-tuning.
- Reproduce the spike environment with
  `scratchpad/rocm10-env.sh`, which strips ROCm 7.2.3 out of
  `NIX_CFLAGS_COMPILE` and `NIX_LDFLAGS` (the dev shell injects both) and
  supplies the gcc toolchain, glibc headers, and crt paths that the unwrapped
  `amdclang++` lacks on NixOS.
