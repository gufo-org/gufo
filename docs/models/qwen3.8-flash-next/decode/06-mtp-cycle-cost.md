# Track 6: the MTP cycle cost curve

**Est. gain:** +5-10% MTP decode (and it compounds with tracks 1-5).
**Effort:** medium. **Risk:** medium — the controller and the verifier width
are both part of the contract.

## Goal

Raise MTP throughput by moving the cycle cost curve, not by retuning the
draft-length controller.

## The measured curve

[`mtp_costs.hpp`](../../../../src/models/qwen38_flash_next/mtp_costs.hpp)
carries calibrated C1 cycle costs in milliseconds for draft widths 1-8. At
depth 0:

| Width `w` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 d0 (ms) | 38.5 | 47.3 | 56.1 | 65.7 | 74.1 | 82.2 | 89.7 | 96.8 |
| delta | — | 8.8 | 8.8 | 9.6 | 8.4 | 8.1 | 7.5 | 7.1 |

The curve is almost exactly linear at depth 0:

```
T(w) = 38.5 + 8.33 * (w - 1)   ms
```

and it separates cleanly into two parts:

- **Fixed ~38.5 ms** — the ordinary token (33.3 ms at AR d0) plus MTP
  catch-up. Paid once per cycle and shared by every verification row.
- **Marginal ~8.33 ms per row** — the extra routed-expert weight reads a wider
  verification adds. Ten experts per row, 48 layers, and with 512 experts the
  overlap between rows is small (8 rows touch ~75 distinct experts, only 7%
  fewer than 80 slots), so the cost is close to linear in rows.

Throughput is `(1 + A(w)) / T(w)` where `A(w)` is accepted drafts. Because a
row costs a quarter of a token, adding one pays only if it raises acceptance
by more than ~25% of a token. Solving for the acceptance implied by the
measured 62.70 tok/s at d0 gives 1.97/2, 2.52/3, 3.12/4, 3.65/5, 4.15/6,
4.62/7, 5.07/8 — a plausible decaying acceptance curve, which means **the
adaptive controller in
[`mtp_policy.hpp`](../../../../src/models/qwen38_flash_next/mtp_policy.hpp) is
already near the optimum width.**

That is the key conclusion: retuning the controller is not the lever. The
lever is moving the curve.

## Lever 6a: lower the fixed part (largest, and shared)

The 38.5 ms fixed part is 40% of a width-8 cycle and 52% of a width-5 cycle.
Every millisecond removed from the ordinary token is removed from every cycle
at every width, so tracks [1](01-side-stream-overlap.md) through
[5](05-moe-router-fusion.md) compound here.

A 10% cut to the 33.3 ms ordinary token is 3.3 ms, which is **4.5% at width 5
and 3.4% at width 8**. This is why the tracks are ordered the way they are.

## Lever 6b: lower the marginal row

The marginal row is ~1.47 GB of routed-expert weights (10 experts x 48 layers
x ~3.07 MB) in 8.33 ms, which is **177 GB/s** against the ~220 GB/s the
machine sustains on streaming reads. Closing that gap saves ~1.6 ms per row:

```
width 5: 6.4 ms of 74.1  = 8.6%
width 8: 11.2 ms of 96.8 = 11.6%
```

The kernels are `mul_mat_vec_moe_gated` and `mul_mat_vec_moe_dispatch`
([`mmvq.hip.cpp`](../../../../src/models/qwen38_flash_next/kernels/rocm/mmq/mmvq.hip.cpp)).
At width 1 the gated path launches `grid((rows+1)/2, experts_used)` with a
64-thread block, so each block reads only 2 rows of one expert — a very small
unit of work. At widths 2-8 it switches to `launch_moe_grouped`, which groups
slots by expert.

Before optimizing, check the achieved bandwidth of these two kernels in
isolation. If they are already at 210+ GB/s, the 177 GB/s figure is wrong and
the marginal cost is coming from somewhere else (the head's per-row output
writes, or the epilogue). `qwen38_flash_next_routed_gemm_bench` is the right
instrument.

Note that this is the one place where the usual advice is inverted: the
repository's retained decode work on these kernels
([`../EXPERIMENTS.md`](../EXPERIMENTS.md) "Decode producer-written Q8_1 rows",
"+5.8-6.0% at d0") already covers the single-token path. What is *not* covered
is the 2-8-row grouped path's bandwidth, which is what the marginal cost
measures.

## Lever 6c: the catch-up

The width-1 cycle is 38.5 ms against a 33.3 ms AR token, so catch-up costs
about **5.2 ms — 16% of the AR token, 7% of a width-5 cycle**. The MTP block is
one full attention layer plus a 512-expert MoE at Q8_0 (the sidecar is 2.79 GB),
so its own weight traffic is only about 160 MB (~1 ms at 150 GB/s). The other
~4 ms is latency or dispatch.

Measure it before optimizing:

```sh
nix develop -c ./build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model $TARGET --mtp-model $MTP --cost-audit 0 --depth 0
```

If the catch-up really is 5 ms of latency rather than traffic, it is another
instance of the same small-kernel problem the other tracks address, and
[track 1](01-side-stream-overlap.md)'s overlap machinery may apply to it
directly.

## Lever 6d: widen past 8 (low value, high effort)

The deltas decline (8.8, 8.8, 9.6, 8.4, 8.1, 7.5, 7.1), so the curve is convex
and wider cycles are cheaper per row. Extrapolating to width 12 with a 6.8 ms
marginal and `A(12) ~ 6.9` gives about 63.7 tok/s against 62.70 — **~2%**, for
a change that requires raising `kMaxMtpDraftTokens`, extending
`kMtpCycleMilliseconds` to widths 9-16, widening `verify_logits_`, and
teaching the routed expert path to take 9-32 rows (currently
`ExpertMatrixRows` is false there, so 9-32 rows take the non-tiled route while
only >32 rows get the tiled one).

Not worth it on its own. Revisit only if 6b makes the marginal row much
cheaper, which would steepen the payoff for width.

## What not to do

- **Retune the controller.** The implied acceptance curve says the current
  width choice is at or near the optimum for this cost curve. The repository
  has already rejected several controller variants
  ([`../EXPERIMENTS.md`](../EXPERIMENTS.md): "Fixed three-proposal default",
  "Context cost without handling saturated acceptance", "Recalibrate
  deterministic sampled draft lengths").
- **Second-filter the proposal lattice.** Rejected: "one short sampled request
  rose from three to four verification rounds, about 307 -> 403 ms".
- **Predict all eight neural rows before returning the selected prefix.**
  Rejected: "Default-sampled prose falls from about 24.7 to 23.45 tok/s".

## Validation

Any change to the cycle cost changes which width the controller prefers, so
the output can change even when the arithmetic does not. Required:

- `mtp_ops_test`, `mtp_sampling_test`, `verification_sampling_test`,
  `wide_verification_test`, `session_test`, `snapshot_test`.
- Greedy output hashes and draft/acceptance counts at d0/4096/32768, C1-C8.
- Sampled: seeded replay, p/q rejection, residual correction, EOS.
- Multi-turn continuation and snapshot restore.
- Re-run `--cost-audit` after the change and update
  `kMtpCycleMilliseconds`; a stale table silently mis-prices every future
  cycle. Record the new table in the experiment row.

Timing:

- `gufo bench --speculative mtp` at d0/4096, unprofiled.
- HTTP serving with mixed and repetitive prompts, since the repetitive case
  hits the prompt-lookup path and does not exercise the same cycle shape.

## Decision criteria

Keep a change if the unprofiled MTP tg improves at both d0 and d4096 with
identical greedy output and identical or better acceptance. Because the
controller adapts, a change that only helps one width may show up as a
different width choice rather than a throughput change — inspect the actual
draft counts, not just the rate.
