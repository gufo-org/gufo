# G6 production build + compile-graph audit (Card 07)

Audited artifacts: `build/card07-release/compile_commands.json`, `CMakeCache.txt`
(preset `release`: RelWithDebInfo, `ENGINE_ENABLE_HIP=ON`, `BUILD_TESTING=OFF`,
`GUFO_EXTENSIONS_FROM_LOCK=OFF` → discovery, `CMAKE_HIP_ARCHITECTURES=gfx1151`),
against the retained pre-card production build `build/card06-release`
(parent revision `11af44f9`, same preset and `nix develop` shell).

A build directory named `release` proves nothing by itself, so the cache was read:

| Cache key | card07-release | card07-gpu | card07-cpu |
| --- | --- | --- | --- |
| CMAKE_BUILD_TYPE | RelWithDebInfo | RelWithDebInfo | RelWithDebInfo |
| BUILD_TESTING | OFF | ON | ON |
| ENGINE_ENABLE_HIP | ON | ON | OFF |
| CMAKE_HIP_ARCHITECTURES | gfx1151 | gfx1151 | gfx1151 |
| GUFO_ENABLE_SANITIZERS | OFF | OFF | OFF |
| GUFO_EXTENSIONS_FROM_LOCK | OFF (discovery) | OFF | OFF |

`-DNDEBUG` is present on every production compile line; sanitizer flags are absent in all
three configurations. `-UNDEBUG` (the "tests keep assertions" option) does **not** appear on
these sources in any of the three builds: it is added by `add_compile_options()` after the
model/family subdirectories are already added, so it never reached them before the split
either. This is pre-existing behaviour, unchanged by Card 07 — the per-source flag
comparison below is the proof (it compares against the pre-move command lines).

## Target-level context replication

`gufo_core` used to supply the moved sources: `PUBLIC` definition `ENGINE_ENABLE_HIP=1`,
`SYSTEM PRIVATE` `HIPCUB_INCLUDE_DIR`/`ROCPRIM_INCLUDE_DIR`, `PUBLIC` links
`gufo_hip_host hip::host roc::hipblas roc::hipblaslt roc::rocblas`, `OpenMP::OpenMP_CXX`,
plus `OpenSSL::Crypto` and `gufo_cache_hip` from `src/models/qwen/CMakeLists.txt`.
The new targets reproduce what their sources actually use:

| Target | Kind | Links (family/core) | Device context |
| --- | --- | --- | --- |
| `gufo_qwen_control_tokens` | INTERFACE | none (header-only) | n/a |
| `gufo_qwen_hip_common` | INTERFACE | `gufo_core` (headers only) | n/a |
| `gufo_qwen_config` | STATIC | `gufo_core` | none; builds and tests under `ENGINE_ENABLE_HIP=OFF` |
| `gufo_qwen_text` | STATIC | `gufo_core`, control_tokens, config | none |
| `gufo_qwen_kernels` | STATIC | `gufo_core`, hip_common | `SYSTEM PUBLIC` HIPCUB+ROCPRIM, HIP language, inherits hip*/OpenMP through `gufo_core` |
| `gufo_qwen_vision` | STATIC | `gufo_core`, control_tokens, hip_common, kernels, text | same device context + `gufo_cache_hip` |
| `gufo_qwen27b` | STATIC | `gufo_core`, control_tokens, hip_common, config, text (+kernels/vision under HIP) | same device context + `OpenSSL::Crypto`, `gufo_cache_hip` |

`SYSTEM PUBLIC` replaces `gufo_core`'s `SYSTEM PRIVATE` on the two HIP family targets: the
device headers are now reached from family headers compiled by several targets, so the
include directory must propagate. It adds no flag to the moved sources themselves — the
`-isystem` entries in their command lines are identical to the pre-move lines (see below).

## Per-source options (the ones that change code generation)

Re-anchored from `TARGET_DIRECTORY gufo_core` to the new owning target, with the original
explanatory comments kept in `src/families/qwen/kernels/CMakeLists.txt` and
`src/models/qwen/CMakeLists.txt`:

- `-mwavefrontsize64`: `small_batch_wave64.hip`, `prefill_quant_wave64.hip`,
  `attention_decode_verify_wave64.hip`
- `-mwavefrontsize64;-gline-tables-only;-Xarch_device;-mllvm=-amdgpu-sched-strategy=iterative-ilp`:
  `small_batch_quant16_wave64.hip`
- `-fno-slp-vectorize`: `prefill_quant_gemm.hip`, `prefill_quant_wave64.hip`

Verified in the production `compile_commands.json` (not in CMake prose):

```
$ python3 compare <old small_batch_quant16_wave64.hip command> <new command>
old-only: ['-DGUFO_FFMPEG_EXECUTABLE="ffmpeg"', '-DGUFO_FFPROBE_EXECUTABLE="ffprobe"',
           'CMakeFiles/gufo_core.dir/src/models/qwen/hip/kernels/small_batch_quant16_wave64.hip.o',
           '/home/mixer/gufo/src/models/qwen/hip/kernels/small_batch_quant16_wave64.hip']
new-only: ['src/families/qwen/kernels/CMakeFiles/gufo_qwen_kernels.dir/small_batch_quant16_wave64.hip.o',
           '/home/mixer/gufo/src/families/qwen/kernels/small_batch_quant16_wave64.hip']
special preserved in new: ['-mwavefrontsize64', '-gline-tables-only', '-Xarch_device',
                           '-mllvm=-amdgpu-sched-strategy=iterative-ilp']
```

Across all 84 compiled Qwen sources the only difference is those two `gufo_core`-PRIVATE
minimax media defines (unused by Qwen code) plus the object/source path tokens; nothing was
added anywhere. Full per-source table: `isa-audit.txt`.

```
$ python3 tools/ci/check-extension-isolation.py --mode report | grep -E 'qwen|family'
  qwen_headers: 61        # covers src/models/qwen/** and src/families/qwen/** now
  family_libraries: 6
  family_edges: 7
```

Integration reproducibility: `python3 tools/prof/card07_parity.py --before build/card06-release/compile_commands.json --after build/card07-release/compile_commands.json --mode flags --scratch /tmp/card07-int-flags` -> exit 0, 84 comparisons, zero differences. Only removed core executable discovery defines GUFO_FFMPEG_EXECUTABLE and GUFO_FFPROBE_EXECUTABLE are excluded explicitly; all include/optimization/HIP flags stay compared. See flag-parity.txt. Original rough scripts retained as original-*.txt, not claimed as runnable gates.

Reproducible host comparator: python3 tools/prof/card07_host_parity.py --root /home/mixer/gufo --before build/card06-release/compile_commands.json --after build/card07-release/compile_commands.json -> EXIT=0; host-parity.txt retains symbol/disassembly comparison (not byte parity).
