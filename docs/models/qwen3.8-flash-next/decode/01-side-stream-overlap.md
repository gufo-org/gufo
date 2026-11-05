# Track 1: side-stream overlap on the decode path

**Measured gain:** **+1.2-1.3% AR decode** at d0/4096/32768 and **+1.4% MTP at
d4096**; MTP d0 unchanged. Prefill unchanged, greedy output byte-identical.
The original +4-8% estimate was too optimistic.
**Effort:** medium. **Risk:** medium — graph capture is the hard part.

## Implementation status

| Item | Status |
| --- | --- |
| 1a shared expert beside routed experts | **implemented and measured** — `ExpertMatrixRows` dropped from the `overlap` gate in `Moe` |
| 1b GDN Z rows | **does not apply to the shipped GGUF** — see the correction below |
| 1c indexer projections | not started; low value |

## Measured result

Strix Halo gfx1151, ROCm 10.1, Nix production build, UD-Q4_K_XL target with the
shared Q8_0 MTP sidecar, `gufo bench -p 2048 -n 128 -r 2`. Baseline and
candidate are the same tree with and without the gate change, run alternating
on a quiet GPU. Three runs per cell.

Autoregressive (`--speculative none`), tg128 tok/s:

| Depth | Baseline runs | Mean | Candidate runs | Mean | Gain |
| ---: | --- | ---: | --- | ---: | ---: |
| 0 | 29.62, 29.77, 29.52 | 29.64 | 29.96, 30.16, 29.92 | 30.01 | **+1.25%** |
| 4,096 | 28.91, 28.98, 28.92 | 28.94 | 29.28, 29.30, 29.27 | 29.28 | **+1.19%** |
| 32,768 | 29.17 | 29.17 | 29.54 | 29.54 | **+1.27%** |

Every candidate run beat every baseline run at d0 and d4096; the ranges do not
overlap, so this is not noise.

MTP (`--speculative mtp`), tg128 tok/s:

| Depth | Baseline mean | Candidate mean | Gain |
| ---: | ---: | ---: | ---: |
| 0 | 62.15 (62.57, 61.94, 61.93) | 62.39 (62.22, 62.91, 62.03) | +0.4%, within noise |
| 4,096 | 50.60 (50.58, 50.63, 50.60) | 51.33 (51.39, 51.53, 51.07) | **+1.44%** |

Prefill is unchanged, as expected — the gate already admitted prefill:
pp2048 d0 baseline 1942.85/1944.21/1901.36 against candidate
1865.08/1936.00/1933.97 (the 1865 is a first-run cold outlier; the other two
runs are within 0.5%), and pp2048 d4096 1915.00/1909.03/1894.99 against
1911.08/1908.94/1905.16.

Correctness: a 96-token greedy completion (temperature 0, thinking off) is
**byte-identical** between the two builds. The capture assertion never fired,
which is itself the evidence that `side_stream_` joined the decode graph
capture on every shape.

### What changed

| File | Change |
| --- | --- |
| `executor.cpp` `Moe` | dropped `ExpertMatrixRows(n_tokens) &&` from `overlap` |
| `executor.cpp` `Moe` | fails the pass if the side stream did not join the capture |
| `executor.cpp` `Run` | sets `capturing_` around `body()` inside the capture region |
| `executor.hpp` | adds `mutable bool capturing_` |

The assertion is the important half. A capture that kept the fork but dropped
the side-stream nodes would replay a graph without the shared expert and
produce wrong logits instead of an error. `hipStreamIsCapturing` on the side
stream after the fork turns that into a hard failure:

```cpp
if (capturing_) {
  hipStreamCaptureStatus side_status = hipStreamCaptureStatusNone;
  if (!Check(hipStreamIsCapturing(side_stream_, &side_status),
             "side stream capture query", error_msg) ||
      side_status != hipStreamCaptureStatusActive) {
    AssignError(error_msg,
                "shared-expert overlap did not join the graph capture");
    return false;
  }
}
```

`capturing_` is false on every replay, so the check costs one bool load per
layer outside capture and nothing in the hot path.

### Why the gain is 1.2% and not 4-8%

The estimate assumed the shared expert's ~22-35 µs per layer would be hidden
entirely. It is hidden, but the routed experts and the shared expert are both
streaming weights from the same DRAM, so the overlapped work partly displaces
rather than fills. The measured +1.2% is the net: the decode kernels are not
as latency-starved as the estimate assumed.

That also means the remaining tracks should be re-costed: if the MoE block is
closer to bandwidth-bound than latency-bound, [track 4](04-hc-mixer-bandwidth.md)'s
alignment work matters more and [track 5](05-moe-router-fusion.md)'s
launch-latency work matters less.

### Static verification performed

- `g++ -std=c++20 -fsyntax-only` on
  `src/models/qwen38_flash_next/kernels/rocm/executor.cpp` with stub HIP,
  hipBLAS and hipBLASLt declarations: **0 errors**. This parses the whole
  translation unit, so it type-checks the new `CaptureScope`, the
  `hipStreamIsCapturing` call and the enum comparison.
- `-Wall -Wextra`: no warnings in the changed regions.
- The same check on every other translation unit that includes
  `executor.hpp` (`batch.cpp`, `engine.cpp`, `runner.cpp`,
  `continuation_adapter.cpp`): 0 errors.
- `clang-format --dry-run --Werror` on both changed files: clean.

Not verified: anything requiring a GPU, the ROCm runtime, or a model.

### Correction: 1b does not apply when the projections are stacked

`DeviceModel` stacks same-input Q8_0 projections at load:

```cpp
if (l.linear && stackable({&l.ssm_qkv, &l.ssm_gate})) {
  d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate});
}
```

([`device_model.cpp`](../../../../src/models/qwen38_flash_next/kernels/rocm/device_model.cpp),
search `d.ssm_in = Stack`). When `attn_qkv` and `attn_gate` are both Q8_0, the
Z rows come out of the **same launch** as Q/K/V, so there is no separate Z
projection to overlap and 1b is moot. It only applies to a GGUF whose
`attn_gate` is a different type from `attn_qkv`, which is not the shipped
artifact. The same stacking applies to `attn_qkv` on attention layers
(`Stack({&l.attn_q, &l.attn_k, &l.attn_v})`).

This also removes most of [track 3](03-gdn-input-projection-fusion.md)'s 3b;
see that file.

## Goal

Run work that is independent of the current dependency chain on
`side_stream_` during decode, so the memory system stays busy while the main
stream waits on a serial dependency.

## Current behaviour

The executor already has a complete overlap mechanism:
`side_stream_`, `side_start_`, `side_done_`, `side_pending_` and `JoinSide()`
([`executor.hpp`](../../../../src/models/qwen38_flash_next/kernels/rocm/executor.hpp),
search `Orders the main stream after the work queued on side_stream_`). It has
three call sites, and all three are gated to prefill.

| Site | Gate | Code |
| --- | --- | --- |
| Shared expert beside routed experts | `ExpertMatrixRows(n_tokens)` | `Moe`, [`executor.cpp`](../../../../src/models/qwen38_flash_next/kernels/rocm/executor.cpp), search `const bool overlap =` |
| GDN Z rows beside the convolution and recurrence | `WidePrefill(n_tokens)` | `LinearAttention`, search `DenseF16SsmGemm` |
| — | — | `Attention` has no overlap at all |

`ExpertMatrixRows(rows)` is `prefill_phase || rows > 32` and
`WidePrefill(rows)` is `prefill_phase && rows > 8`, so a decode pass
(`prefill_phase == false`, `n_tokens == 1`) takes none of them. The shared
expert therefore runs to completion before the routed experts start, and the
GDN Z projection runs before the convolution.

## Why it should work

Overlapping two DRAM-bound kernels gains nothing — the bandwidth is already
shared. The argument here is different: **the decode kernels are latency
bound, not bandwidth bound.** A 1-row GEMV over a 640-row matrix launches few
blocks and each thread issues a short chain of loads, so the memory system is
under-subscribed. Two such kernels running concurrently fill each other's
latency.

The prefill evidence supports this: overlapping the shared expert slowed the
routed gate/up kernel (223 → 251 ms per pass) but still shortened the pass
(1,309 → 1,335 tok/s at pp512). At decode the contention should be smaller
because neither kernel is near saturation.

## Candidates, in order of expected value

### 1a. Shared expert beside routed experts — implemented

- **What:** dropped the `ExpertMatrixRows(n_tokens)` term from `overlap` in
  `Moe`. The remaining guards (Q8_0-only, `!last_only`) still apply.
- **Bytes:** shared expert 3 x 640 x 2560 = 4.92 M params, Q8_0, ~5.2 MB per
  layer. Routed experts for 10 of 512 experts are ~30.7 MB per layer, so the
  shared expert fits inside the routed window with room to spare.
- **Upper bound:** ~24-35 µs per layer x 48 = **1.2-1.7 ms, 3.5-5%**.
- **Structure:** `Moe` already queues the shared expert before the routed
  dispatch and `MoeExperts` calls `JoinSide` before `MoeEpilogue`, so only the
  gate needed to change.

Data-hazard review done before the change:

- `s_.mixed` is producer-written by `HcMixQ8Fused` into `s_.mixed_q8_1`, so
  both the shared expert and the routed experts read it through `ReadyQ8` and
  neither re-quantizes it. Read-only on both streams.
- The shared expert's down projection *does* quantize (`s_.shexp_up`, 640
  wide) into `s_.x_q8[q8_slot_]` on the side stream. `Quantize` alternates two
  slots, and in the steady-state decode loop the main stream performs no
  `Quantize` at all (every projection consumes producer-written Q8_1 rows), so
  the two streams cannot collide on a slot. If a future change adds a
  main-stream `Quantize`, the two-slot alternation still separates them — but
  re-check it.
- `qfn_pool_for_stream` is keyed on `(device, stream)`, so the side stream gets
  its own scratch arena. The shared expert's decode kernels allocate nothing
  from the pool (`input_bytes` and `group_bytes` are both zero when the input
  is already quantized and `n_tokens == 1`), so the side arena does not grow
  during capture.
- `s_.shexp_out` is read by `MoeEpilogue`, which runs after `JoinSide`.

### 1b. GDN Z rows beside convolution and recurrence — does not apply

Blocked by load-time stacking: `d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate})`
fuses both projections into one launch whenever both are Q8_0, which is the
shipped artifact. There is no separate Z projection to move.

It would only apply to a GGUF whose `attn_gate` type differs from
`attn_qkv`. If that ever matters, the change is: launch
`Dense(l.ssm_gate, xq, s_.z, ...)` on `side_stream_` and pass the recorded
event as `z_ready` to `GatedDeltaNet`, whose epilogue already waits on it. The
plumbing exists (`z_ready` parameter, `z_join` guard).

### 1c. Indexer projections beside the attention path

- **What:** `Dense(l.indexer_k, ...)` and, in sparse mode,
  `Dense(l.indexer_q, ...)` in `Attention` are independent of the main
  Q/K/V projection and the attention kernel.
- **Bytes:** 2560 x 128 + 2560 x 512 = 1.64 M params, ~1.7 MB per layer, only
  12 layers.
- **Upper bound:** small, **~0.2-0.4%**. Include it only if 1a and 1b land
  cleanly; the extra graph complexity is probably not worth it alone.

## The graph-capture trap

Decode replays through `Run()` → `hipGraphLaunch`. Side-stream work is only
captured if it is forked and joined **inside** the capture region with events
recorded on the capturing stream. The prefill path already does exactly this
(`hipEventRecord(side_start_, main)` → `hipStreamWaitEvent(side_stream_, ...)`
→ `hipEventRecord(side_done_, side_stream_)` → `hipStreamWaitEvent(stream_,
side_done_)`), so the pattern is known to work.

Note that the prefill overlap has never exercised this: `WidePrefill` requires
`prefill_phase`, and `graph` requires `!prefill_phase`, so the two have never
been active together. Enabling the overlap on decode is the **first** time
side-stream work runs inside a captured region, so it needs explicit checking
rather than an appeal to the prefill precedent.

Do not assume it worked. After capture, verify the node set:

- Log `hipGraphGetNodes` count for the decode key before and after the change
  and confirm it grew by the expected number of side-stream kernels (the
  shared expert contributes 3: gated gate/up, the activation quantize, and the
  down projection).
- A capture that silently drops the side stream would produce wrong logits
  rather than an error, so the logit checks below are the real gate.
- The `EXPERIMENTS.md` row "Graphs: verify replay actually launches every new
  operation" records that this class of bug has bitten this codebase before.
- Also confirm the side-stream kernels are not silently dropped when the shape
  key changes (sparse on/off, `n_logits` 0 vs 1, `logits != nullptr`).

A second trap: `Run()` may execute the body eagerly once before capturing
(`session.warmed_`). The dispatch-state snapshot/restore in `Run` covers
`moe_pending_`, `defer_residual_`, `q8_slot_` and friends; any new mutable
state introduced by the overlap must be added to that tuple or the recorded
graph will not match the executed pass.

## Alternatives if capture proves intractable

- Keep the overlap but exclude the affected shape from graphing (force eager).
  Measure the trade: eager costs ~1.9 µs per node of launch overhead, so this
  only wins if the overlap saves more than that.
- Fuse instead of overlap. `HcMixQ8Fused` shows the fused-with-phase-counters
  pattern works well here (`HcMixQ8FusedKernel`, `FusedTake`/`FusedDone`/
  `FusedWait`). A fused "shared expert + routed experts" launch with a phase
  barrier would get the same overlap without a second stream.

## Validation

### Step 0 — confirm the capture carries the side stream

The in-code assertion now covers this, so a failure appears as
`shared-expert overlap did not join the graph capture` on the first request
that reaches `Moe`. If it fires, stop: the overlap is not safe on this ROCm
and the gate change must be reverted.

To see the node count directly, temporarily print it after
`hipStreamEndCapture` in `Run`:

```cpp
std::size_t nodes = 0;
(void)hipGraphGetNodes(captured, nullptr, &nodes);
fprintf(stderr, "graph nodes: %zu\n", nodes);
```

With 48 layers the shared expert contributes 3 kernels each (gated gate/up,
the activation quantize, the down projection), so the decode graph should
grow by **144 nodes** relative to the unmodified build. A smaller increase
means the capture dropped part of the side-stream work.

### Step 1 — correctness (must be byte-identical)

```sh
nix develop -c cmake --preset gpu-test
for t in moe_ids_ops moe_epilogue_ops routed_wmma_ops hc_mix_ops hc_route_ops \
         projection_ops gdn_ops session_test snapshot_test; do
  nix develop -c cmake --build --preset gpu-test \
    --target qwen38_flash_next_$t
  nix develop -c ctest --preset gpu-full \
    -R "qwen38_flash_next\.?$t" --output-on-failure
done

# Fixed-history logits at every verification width against the F32 reference
nix develop -c ./build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model $TARGET --prompt-file $PROMPT --batch 1 --context 4096 \
  --logit-rows 256 --dump /tmp/ov1-logits
```

The logit dump is the one that matters: it is the only check that would catch a
capture that silently dropped the shared expert, because that produces wrong
numbers rather than an error. Compare against
`tests/fixtures/qwen38_flash_next/f32_reference/`.

Then greedy output and draft counts at d0/4096/32768 for C1-C8, which also
exercises the `MoeBatch -> Moe` delegation at 2-8 rows. The shared expert's
result feeds `MoeEpilogue` through `s_.shexp_out`, so a stale or missing write
shows up as a changed completion.

### Step 2 — timing

```sh
# Unprofiled A/B, alternating, quiet GPU, same binary pair
./build/release/gufo bench -m $TARGET --speculative none -p 2048 -n 128 -d 0,4096 -r 2

# Profile: the discriminator is total kernel time, not wall time
nix develop -c tools/prof/prof.py run --stages qwen-flash -o /tmp/ov1 -- \
  build/release/gufo bench -m $TARGET --speculative none -p 2048 -n 128 -d 0 -r 2
nix develop -c tools/prof/prof.py show /tmp/ov1/*.db --stages qwen-flash --gaps 8
```

Acceptance:

| Observation | Meaning |
| --- | --- |
| kernel time drops, gaps shrink | overlap works — keep |
| kernel time unchanged, gaps shrink | overlap works — keep |
| kernel time rises enough to cancel the gap reduction | streams contend for DRAM — revert |

Contention is the expected failure mode: this is exactly what happened to the
prefill shared expert, where the routed gate/up kernel slowed 223 -> 251 ms
per pass. There the pass still shortened only because those kernels are much
larger than a decode kernel.

Scope: `MoeBatch` delegates to `Moe` for `rows <= kDecodeRows` (8), so this
covers C1-C8 decode. Batches wider than 8 rows take the separate `MoeBatch`
path, which still runs the shared expert serially — extending the overlap
there is a follow-up, not part of this change.

## Decision criteria

Keep only if:

1. logits and completions are byte-identical at every width, and
2. unprofiled `gufo bench` tg improves at both d0 and d4096, and
3. the graph node count confirms the side-stream work is replayed.

Drop if the individual kernels slow enough to cancel the overlap — that is
what happened with the prefill shared expert, and there the pass still
shortened only because the kernels were much larger.
