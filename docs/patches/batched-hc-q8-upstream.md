# Batched HC Q8 reuse: focused upstream qualification

Base: gufo-org `773a16419282a3b9790d685ce6b4dd980b285943`.
The production change only enables an existing epilogue output and records it in
`ReadyQ8` for unpadded batches of at most eight rows. Existing wide/padded paths,
GPU arithmetic, cache identity checks and synchronization are retained.

## Current-canonical checks

An independent CMake RelWithDebInfo GPU-test build on Strix Halo gfx1151, private
ROCm 7.14.1/Clang 23, uses current canonical headers and libraries. It contains no
RDMA implementation and reuses no private fork ABI. Assertions are enabled in the
operator test. The shared format check passes all 866 C++ files.

The real GPU HC suite passes, including 48 added producer/separate-MMQ comparisons:
rows 1–8, injection on/off, input scales 0.0001/1/10000. Complete FP32/injection/Q8
bytes match, with guards at the actual output boundaries. The existing analytic
HC checks also pass. Matching two implementations is not an independent full-model
teacher evaluation.

A diagnostic full-model replay uses UD-Q4_K_XL and shared-Q8_0 MTP, four sessions of
65,536-token capacity and actual short/32,768-token contexts. All 104 records pass:
64 matched continuation records and 32 complete materialized post-replay state
comparisons, with full finite score hashes, tokens, positions, RNG and draft/work
counts equal. Borrowed same/cross-session, serialized cross-session and file
round-trip restores are covered. Observed verification widths are 2/4/7/8/10/16;
the producer guard is checked on eligible widths and the fallback on wider widths.
The test completed in 131.66 seconds; this is not an inference throughput result.
Minimum available memory was about 19.2 GiB and the boot did not change.

### Important replay boundary

The first attempt retained canonical production proposal planning and failed its
strict per-cycle comparison after restoring a session: the reference packed two
rows and returned one token for actor 0; the candidate packed three and returned
two tokens. Their histories reconverged on the following cycle. That run is kept,
including its 14 raw records; it does not establish a quantization defect or prove
production-controller replay.

Canonical `DecodeRequest` does not expose the downstream fork's per-request draft
ceiling or TP plan parameter. The successful numerical test therefore uses two
isolated diagnostic host objects: producer toggle/counters in `batch.cpp`, and an
explicit alternating one/zero-proposal override of the greedy batch plan in
`engine.cpp`, identical on both sides. Sampled planning remains unchanged. All
other archive members and GPU code match the canonical build. This fixes the
input history for the numerical comparison; **it is not a production greedy
controller replay, a shipped option, or a performance binary**. The actual PR
contains neither diagnostic hook. No numerical tolerance or golden was changed.

## Performance remains inconclusive

The [downstream fork draft](https://github.com/LostVector/gufo/pull/14) retains its
full source/raw evidence and limitations. Its component comparison saved
2.279–2.519 microseconds per mixer across the tested conditions. Its separate
20-start, paired end-to-end TP=2 study found:

| Completed client-wave throughput | Change | Marginal 95% interval |
| --- | ---: | ---: |
| Coding C2 | −0.089% | −1.341% to +1.353% |
| Short C2 | +0.468% | −0.025% to +0.987% |

Both remain statistically inconclusive. All 33 phase flags and 21 adaptive-work
comparison differences are retained. These results establish neither single-node
speedup, equivalence, uniform latency improvement nor adoption readiness. No new
speed claim is made by the upstream port.

## Reproduction and raw evidence

[Acquisition bundle](batched-hc-q8-upstream-data/acquisition.zip) contains the
sources, exact commands/environments, raw score/state hashes, logs and failed
attempts. It excludes weights, binaries, object files and multi-gigabyte snapshots.
[Manifest](batched-hc-q8-upstream-data/archive.json) lists every member's size/hash.
Run `python3 docs/patches/batched-hc-q8-upstream-data/check.py` for archive integrity
and recorded-coverage checks; this does not rerun the GPU test.

The bundle's `configure-v3.json` records the canonical CMake configuration. Build
`qwen38_flash_next_hc_mix_ops_test`, then run CTest with
`-R '^qwen38_flash_next.hc_mix_ops$' --output-on-failure`.
The external `build-tp1.py`, corrected `tp1-link-fix-commands.json`, `run-tp1.py` and
`analyze.py` retain the diagnostic replay procedure and fixed-plan limitation.
Local paths need adjustment on another machine.

Initial dependency-root/cache discovery failures and a diagnostic link failure
from placing `engine.cpp.o` in the wrong archive occurred before model execution
and are retained. The corrected link replaces exactly `batch.cpp.o` and
`engine.cpp.o` in a copied HIP archive, leaving the canonical model archive intact.
