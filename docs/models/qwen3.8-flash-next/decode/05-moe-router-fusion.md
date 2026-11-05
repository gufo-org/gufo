# Track 5: fuse the router chain and the shared expert

**Est. gain:** +1.5-2% AR decode.
**Effort:** low. **Risk:** low.

## Goal

Remove two launches per layer from the MoE block, both of which are pure
latency at width 1.

## Current behaviour

`Moe`
([`executor.cpp`](../../../../src/models/qwen38_flash_next/kernels/rocm/executor.cpp),
search `bool Executor::Moe(`) issues, per layer:

| # | Launch | Shape at `n_tokens == 1` |
| --- | --- | --- |
| 1 | `Dense(l.router, ...)` -> `SmallGemmKernel` | 128 blocks, 4 rows each, 17.7 µs measured |
| 2 | `RouterTopK` -> `RouterTopKKernel<512>` | **grid = 1 block**, 256 threads |
| 3 | `GatedDense(shexp_up, shexp_gate, ...)` | 640 rows, Q8_0 gated GEMV |
| 4 | `Dense(shexp_down, ...)` | 2560 rows, 640 wide |
| 5 | `GatedExperts(ffn_gate_exps, ffn_up_exps, ...)` | 3200 blocks |
| 6 | `Experts(ffn_down_exps, ...)` | 3840 blocks |
| 7 | `MoeEpilogue` | 51 KB of slot rows |

Two of these are pathological at width 1:

- **`RouterTopKKernel` runs one block.** `dim3(n_tokens)` with `n_tokens == 1`
  puts a single 256-thread block on a 40-CU part. It computes a 512-way
  softmax and ten sequential argmax passes. That is a few microseconds of work
  and a full launch plus drain of latency.
- **The shared expert is two launches** with a 2.5 KB intermediate between
  them. The gate/up GEMV already fuses SiLU
  (`qfn_mmq_q8_0_dense_vec_preq`), so the only thing keeping this at two
  launches is that the down projection needs the whole gate/up result.

Note: the expert-count download is **not** on the decode path. `Moe` guards
`ExpertCounts` + `hipMemcpyAsync(counts_host_, ...)` with
`ExpertMatrixRows(n_tokens)`, which is false at width 1. Nothing to remove
there.

## Change

### 5a. Fold top-k into the router GEMV epilogue

The router GEMV finishes with one block per four output rows; the top-k needs
all 512 logits. Use the phase-counter idiom already proven in
`HcMixQ8FusedKernel` (`FusedTake` / `FusedDone` / `FusedWait` /
`FusedClear`, [`kernels.hip.cpp`](../../../../src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp)):
each block publishes its rows and increments a done counter; the block that
brings the counter to the grid size performs the top-k over `s_.router`.

- Saves launch 2 entirely, plus the full-grid drain between the two kernels.
- `RouterTopKKernel`'s arithmetic must be reused verbatim: the block max, the
  `__expf(src - max)` probabilities, the block sum, the normalisation, then
  the ten sequential argmax passes with the existing tie ordering. The
  repository treats expert-selection tie ordering as part of the contract, so
  do not "simplify" it into a sort.
- The counter must be cleared for the next launch or graph replay; follow the
  existing `FusedClear` pattern and make sure the clearing block is the last
  one to exit.

**~7 µs per layer x 48 = 0.34 ms, 1%.**

### 5b. Fuse the shared expert into one launch

gate/up (640 rows, 2560 wide, Q8_0) -> SiLU -> down (2560 rows, 640 wide,
Q8_0). Same structure as the HC mixer: a small gated projection feeding a
transposed projection. The `HcMixQ8FusedKernel` phase design applies directly
— a combine-like phase, a gate/up phase, a down phase, an epilogue.

- Saves one launch and one full-grid drain.
- The intermediate is only 640 floats per token, so the traffic saved is
  irrelevant; the gain is entirely latency.

**~5 µs per layer x 48 = 0.24 ms, 0.7%.**

If 5b is awkward, an alternative that gets most of the benefit: run the shared
expert on `side_stream_` beside the routed experts
([track 1](01-side-stream-overlap.md)). Doing both is redundant — pick one.

## Traps

- **`s_.router` is shared with the shared-expert gate.** `Moe` reads
  `s_.router + c.num_experts` (index 512) as the shared expert's sigmoid gate,
  and `RouterTopK` is called with `stride = c.num_experts + 1` so the router
  GEMV produces 513 values. A fused router kernel must produce and publish all
  513 before the shared expert reads index 512.
- **`ReadyQ8` reuse.** `s_.mixed` is quantized once per layer and reused by
  the router, the shared expert and the routed experts. A fused kernel must
  take the already-quantized rows rather than calling `Quantize` again, or it
  will add a quantization pass per launch.
- **Graph keys.** The MoE block's node set changes for every decode graph key.
  Re-warm all of them.

## Validation

Correctness:

- `moe_ids_ops` (routing and tie ordering), `moe_epilogue_ops`,
  `hc_route_ops`.
- `routed_wmma_ops` for the expert path that consumes the ids.
- Fixed-history logit dumps at widths 1-8.
- Greedy output hashes and draft/acceptance counts at d0/4096/32768, C1-C8.
- Sampled requests: the shared-expert gate feeds a sigmoid that multiplies the
  shared output, so verify sampled replay as well as greedy.

Timing:

- `prof.py` before/after on the `moe: routing/layout` and `moe: grouped
  gate/up` stages.
- Unprofiled `gufo bench` at d0 and d4096.

## Decision criteria

Keep if byte-identical and unprofiled tg at d0 improves. This track is cheap
enough to try before the larger ones, and it de-risks the phase-counter idiom
for [track 4](04-hc-mixer-bandwidth.md).
