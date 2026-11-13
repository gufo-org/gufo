# GLM-5.3-Flash benchmarks (gfx1151)

Model: `GLM-5.3-Flash-UD-IQ1_M` (90.87 GiB, `glm5-next` 320B.A18B, 1.75 bpw),
`--n-gpu-layers 99`, flash-attn on, single UMA device (AMD Radeon 8060S,
gfx1151), 126976 MiB. All runs go through `tools/bench/gpu_exclusive.sh`
(bench_bg/bench_wait discipline). Gate and matched-A/B timings use
`gufo serve` over HTTP with server-reported `timings` to replicate the
captures below; `gufo bench` (registered for `glm5-next` since 2026-11-13)
is the fast exact-depth scan tool and reads ~5 % faster on pp because it
excludes the serve scheduler and HTTP (see
[staging/models/glm5_next/AGENTS.md](../../../staging/models/glm5_next/AGENTS.md)).

## Reference (llama.cpp)

llama.cpp c49edf096 (11879), ROCm build `build-glm5` (HIP arch gfx1151,
`-DLLAMA_CURL=OFF`). Captured 2026-11-07, repetitions=2 ± stddev, single
capture, no thermal control, no ABBA alternation (gufo-vs-reference matched
comparisons require the full contract in `docs/COMMON_OPTIMIZATION_STRIX_HALO.md`).

| test | llama.cpp t/s | gufo t/s |
| --- | --- | --- |
| pp2048 | 189.44 ± 5.11 | 79.9 ± 0.1 † |
| tg32 | 16.80 ± 0.02 | 12.4 ± 0.1 † |

† Captured 2026-11-13 (P7 row-warp GatherScore) on the rebuilt gate
harness (`/tmp/opencode/glm5_ab_p7.sh`, `serving_bench --concurrency 1`,
prompt_n ≈ 2232), so these cells are not comparable to the archived
P6-column protocol (67.89 pp / 11.45 tg for P6 on the original session);
same-session control was 66.7 ± 0.3 pp / 12.4 tg, making P7 **+19.7 %**
matched. The P6 protocol text below is retained for history.

Offline `gufo bench` reference point (same P7 code, GPU-exclusive,
repetitions=2): pp2048 **85.11 ± 0.06**, tg32 **11.13 ± 0.07**
(`/tmp/opencode/glm5_bench2k.log`). With the 256-token prefill chunk
(+6.0 %, bit-identical — greedy sha and `rocm_forward` max |dlogit|
0.0000 vs the golden): pp2048 **90.3 ± 0.1** (tg unchanged). With the
P9 dense Q8_0 WMMA tier on top (+9.7 %; same sha, forward gate ALL PASS
0.0000): pp2048 **99.2 ± 0.2**, pp16384 90.0/90.7 vs 83.1 control.
With the 2048-token prefill chunk (+9.1 % at pp2048, +8.3 % at pp4096,
n-gen-1 sha identical across chunk widths, full `glm5_next` suite 9/9):
pp2048 **109.2 ± 0.5**, pp4096 **104.6**.
With single-launch chunked KDA on top (+21.7 %; chunk conv, fused
fold/update with the state resident in LDS, flattened E-Gate/Scale/
Out-Gate grids; n-gen-1 sha identical, full `glm5_next` suite 9/9
including new bit-exact parity checks): pp2048 **132.9 ± 1.4**.
With the HC/MoE elementwise loops row-batched into single launches
(route, coeffs, inputs, write-back, weighted sum; bit-identical —
n-gen-1 sha and 9/9 with new row-batch parity checks): pp2048
**164.7 ± 1.7**.
Serve-protocol gate not yet re-captured for these steps. Use these
only as offline scans, not interchangeably with the serve-protocol gate
cells.

Gufo column captured 2026-11-12 on `build/gpu-test/gufo` (P6 dense Q5_K
WMMA tier over the P5 fixed + widened WMMA routed tier) through
`gufo serve` over HTTP with server-reported `timings`, greedy (seed 1),
`--think off`, `--prefill-chunk 2048`, fresh request per sample, one warmup
then two timed samples, GPU-exclusive (bench_bg/bench_wait). These captures
ran with the default (native, 1M-token) `--context`; since 2026-11-11 all
gates, A/Bs and lever runs pin `--context 131072` (see
[staging/models/glm5_next/AGENTS.md](../../../staging/models/glm5_next/AGENTS.md)),
so re-measure controls at 131072 before comparing against the rows above.
The matched
three-point session (`/tmp/opencode/glm5_bench_ab3.sh`, same session):
baseline (`kChunk = 4`, slot-order grouped MoE) 20.69 ± 0.18 pp2048 /
12.14 ± 0.08 tg32 → bucket-compacted routed tier 38.56 ± 0.53 → WMMA
IQ1_M gate/up tier 42.21 ± 0.01 / 12.24 ± 0.00 — later found numerically
invalid (row-offset bug; see the P4 correction in
[EXPERIMENTS.md](EXPERIMENTS.md)). The fixed kernel with widened tier
coverage (IQ1_M + IQ2_XXS gate/up, IQ3_XXS `down`, P5) measures in a
fresh alternating session (control = `gufo_rows16`,
`/tmp/opencode/glm5_ab.sh`): pp2048 **51.65 ± 0.17 / 51.65 ± 0.31** vs
  control 42.69 / 42.35 (**+21.4 %**), tg32 flat (11.30/11.48 vs
  11.51/11.12); at ~8.4K prompt the tier keeps **+17.2 %** (43.6 avg vs
  37.2 avg). Phase-3 gate
  (pp2048 ≥ 0.9×, tg32 ≥ 1.0×) is still **not met**: 0.27× on pp2048,
  0.68× on tg32.
  P6 (same day, same protocol, `/tmp/opencode/glm5_ab_p6.sh` with
  control = `gufo_p5`): the dense Q5_K WMMA tier adds **+32.6 %** —
  candidate pp2048 68.03 / 67.89 / 67.83 vs control 51.12 / 51.32,
  cumulative **+59.7 %** over the `gufo_rows16` control (42.52) — tg32
  flat (11.22 / 11.45 vs 11.26 / 11.31). At ~8.4K prompt the tier keeps
  **+28 %** (55.8 avg vs 43.6 avg control,
  `/tmp/opencode/glm5_ab8k_p6.sh`). Gate attainment is now
  **0.40×** on pp2048 (67.9 vs ≥ 170.5) and 0.68× on tg32.

The bucket-reuse saturation probe (`kChunk = 512`, flat within noise)
shows the remaining pp2048 gap is expert-decode efficiency at ~42 GB/s
effective, not weight re-reads. Both GEMV families are now on the
tensor-core tier (routed experts +21.4 % under P5, dense Q5_K +32.6 %
under P6, dense Q8_0 +9.7 % under P9); the remaining tensor-core-scale
candidates are the MLA GatherScore pass (13.1 % of pp) and the KDA
chunked scan (9.3 %).

Gufo reproduce:

```sh
setsid tools/bench/bench_bg.sh p6ab 4500 /tmp/opencode/glm5_ab_p6.sh & disown
sleep 2
tools/bench/bench_wait.sh p6ab 5400 90
```

Reproduce:

```sh
setsid tools/bench/bench_bg.sh glm5_base 900 \
  /home/praburaja/projects/llm/llama.cpp/llama.cpp/build-glm5/bin/llama-bench \
  -m /home/praburaja/projects/llm/models/gguf/GLM-5.3-Flash-IQ1_M/GLM-5.3-Flash-UD-IQ1_M-00001-of-00003.gguf \
  -p 2048 -n 32 -fa on -ngl 99 -r 2 -v & disown
sleep 2
tools/bench/bench_wait.sh glm5_base 1200 90
```

## Fixtures

`tests/fixtures/glm_next/llama_reference.json` is a gufo **self-regression
golden** (schema `gufo.glm_next.llama_reference.v2`): greedy chain, per-case
top-32 logits after prefill and after gen, recorded from the rocm executor
itself (capture mode `GUFO_GLM5_NEXT_STEPS=<steps.json>` on the forward
test, same GPU-exclusive discipline; the steps document is converted into
the fixture). The strict replay in `glm5_next_rocm_forward_test` demands
top-1/greedy exactness inside a 0.5-nat budget and matched the capture
bit-for-bit on this host. Refresh it only by re-capturing on the
candidate. Refreshed 2026-11-13 (captured at revision 457151058) for
the P7 GatherScore kernel: the reassociation moved one chaotic-boundary
`code` snapshot beyond the 0.5 budget while the greedy chain, every
top-1 and all prefill snapshots stayed identical; the re-capture
replayed bit-identical on this host (`rocm_forward` ALL PASS).

`tests/fixtures/glm_next/wt_reference_seq_5c8854fd8.json` is the
**semantic anchor**: the same cases captured from llama.cpp @ 5c8854fd8
with `tools/glm5_parity/llm_probe` (one token per evaluation, top-32 logits
at every position). Compared manually, never gated: gufo typically differs
from wt by 1-2 nats on shared tokens with rare top-1 flips at coin-toss
margins — wt's own GPU lm_head alone deviates 4.65e-3 relative from
float64 while gufo's stays at 1.2e-7, so a tighter cross-engine budget is
not attainable. Independent numpy float64 simulation of the mHC pipeline
places gufo closer to the exact result on divergent positions.
