# GLM-5.3-Flash benchmarks (gfx1151)

Model: `GLM-5.3-Flash-UD-IQ1_M` (90.87 GiB, `glm5-next` 320B.A18B, 1.75 bpw),
`--n-gpu-layers 99`, flash-attn on, single UMA device (AMD Radeon 8060S,
gfx1151), 126976 MiB. All runs go through `tools/bench/gpu_exclusive.sh`
(bench_bg/bench_wait discipline).

## Reference (llama.cpp)

llama.cpp c49edf096 (11879), ROCm build `build-glm5` (HIP arch gfx1151,
`-DLLAMA_CURL=OFF`). Captured 2026-11-07, repetitions=2 ± stddev, single
capture, no thermal control, no ABBA alternation (gufo-vs-reference matched
comparisons require the full contract in `docs/COMMON_OPTIMIZATION_STRIX_HALO.md`).

| test | llama.cpp t/s | gufo t/s |
| --- | --- | --- |
| pp2048 | 189.44 ± 5.11 | 21.07 ± 0.26 |
| tg32 | 16.80 ± 0.02 | 12.12 ± 0.06 |

Gufo column captured 2026-11-10 on `build/gpu-test/gufo` through
`gufo serve` over HTTP with server-reported `timings`, greedy (seed 1),
`--think off`, `--prefill-chunk 2048`, fresh request per sample, one warmup
then two timed samples, GPU-exclusive (bench_bg/bench_wait). Phase-3 gate
(pp2048 ≥ 0.9×, tg32 ≥ 1.0×) is **not met**: 0.11× on pp2048, 0.72× on tg32.

Root cause of the pp2048 gap is structural, not kernel tuning: the executor
prefills with the same `kChunk = 4` GEMV path as decode, so all weights are
re-read every 4 tokens (~185 ms/chunk ≈ 527 GB/s effective, ~21× below the
llama.cpp batched-GEMM prefill). tg32 implies ~59.5 ms/token ≈ 42 GB/s
effective expert-weight traffic against a ~2.53 GB/token routed read, i.e.
heavily below the measured memory ceiling — the fused IQ-expert MoE decode
kernel and a wide-batch prefill kernel are the Phase-3 headroom.

Gufo reproduce:

```sh
setsid tools/bench/bench_bg.sh glm5gate 600 /tmp/opencode/glm5_bench_gates.sh & disown
sleep 2
tools/bench/bench_wait.sh glm5gate 660 90
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
bit-for-bit on this host. Refresh it only by re-capturing on the candidate.

`tests/fixtures/glm_next/wt_reference_seq_5c8854fd8.json` is the
**semantic anchor**: the same cases captured from llama.cpp @ 5c8854fd8
with `tools/glm5_parity/llm_probe` (one token per evaluation, top-32 logits
at every position). Compared manually, never gated: gufo typically differs
from wt by 1-2 nats on shared tokens with rare top-1 flips at coin-toss
margins — wt's own GPU lm_head alone deviates 4.65e-3 relative from
float64 while gufo's stays at 1.2e-7, so a tighter cross-engine budget is
not attainable. Independent numpy float64 simulation of the mHC pipeline
places gufo closer to the exact result on divergent positions.
