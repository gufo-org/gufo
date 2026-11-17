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
With the Q6_K WMMA dense tier (`ffn_down_shexp` and the other Q6_K
projections moved from wide GEMV to the tensor-core tier; bit-identical —
n-gen-1 sha and 9/9, `rocm_dense_wmma` extended with Q6_K shapes):
pp2048 **179.0 ± 0.3** (`/tmp/opencode/gufo_bench_q6kab.log`).
With the token-batched DSA indexer/MLA launches (four kernels gain the
token as a grid dimension over per-token scratch; bit-identical —
n-gen-1 sha and 9/9): pp2048 **184.5 ± 0.4**, pp16384 **147.0 ± 0.6**
(`/tmp/opencode/gufo_bench_mla2ab.log`,
`/tmp/opencode/gufo_bench_mla2gate.log`).
With the single-list top-k run at its `k2`-sized bitonic network
(bit-exact — n-gen-1 sha, 9/9, existing single-list contract):
pp2048 **193.2 ± 2.0**, pp16384 **149.4** parity
(`/tmp/opencode/gufo_bench_topk1ab.log`).
With the KDA fold/update chunk keeping each state column in registers
(bit-exact contract zero margin): pp2048 **215.96 ± 4.6**, pp16384
**163.5** (alternating controls 153.5), sha and 9/9 unchanged
 (`/tmp/opencode/gufo_bench_kda2.log`, `_kda3gate.log`).
With the Q5_K wmma fragment decoded through dword-aligned LDS windows
(bit-exact — wmma contracts untouched): pp2048 **216.5 / 217.6**,
pp16384 **169.5 / 162.2** (alternating controls 162.0/160.4), sha and
9/9 unchanged (`/tmp/opencode/gufo_bench_shipgate.log`).
Serve-protocol gate not yet re-captured for these steps. Use these
only as offline scans, not interchangeably with the serve-protocol gate
cells.

### B10 — MTP nextn speculative decoding (2026-11-15)

Opt-in `--speculative mtp` (serve and bench): the GGUF `nextn` block drafts
up to k tokens, verified with the correction row in one batched trunk pass.
`kMaxVerifyRows = 5` bounds k ≤ 4; the engine adapts k in
[`--min/max-draft-tokens`], growing one after two full accepts and dropping
one on any mismatch (greedy accept is exact: the longest verified prefix +
correction token preserves the plain decoding chain).

Offline scan (greedy, `-p 2048` depth, n-gen 64, repetitions 1): tg64
9.31 → **14.90** (**+60 %**) with `output_sha256` **identical**
(`1d79e532…`) between plain and MTP — lossless at tg64; pp2048 260.79 →
241.88 (**−7.3 %**, draft-block prefill of the prompt rows)
(`/tmp/opencode/gufo_bench_glm5mtp.log`).

Serve-protocol A/B (`gufo serve --context 131072`, `serving_bench` default
case greedy max-tokens 32, concurrency 1, repetitions 2,
`/tmp/opencode/glm5_mtp_gate2.sh`): decode median 12.21 → **14.12**
(**+15.6 %**, max sample 16.21; acceptance **70.5 %**, 44 drafted /
31 accepted over 32-token completions), prefill median 213.1 → **199.4**
(−6.4 %) — pp side of the Phase-3 gate still held (199.4 ≥ 0.9×189.44)
(`/tmp/opencode/gate_cplain.json`, `/tmp/opencode/gate_mtp1.json`).
Policy scans: fixed k=4 regresses (decode median 11.01, acceptance 42.9 %)
and grow-after-one regresses (13.09, acceptance 68.6 %); the two-streak
policy above remains the default (`gate_mtp4/gate_mtp1b` artifacts).

The **absolute tg32 cell (llama.cpp 16.80) is not attained at the 32-token
decode window**: best matched median 14.12 = **0.84×**. Ramp-up over ~20
verify steps dominates short generations (max sample 16.21; the offline
tg64 point 14.90 shows the asymptotic direction). Suite `glm5_next` 9/9 on
the candidate including the strict `rocm_forward` golden replay and the new
`Add` op parity case (`/tmp/opencode/gufo_bench_glm5suite2.log`).

### B11 — Opt-in W8A8 dense prefill tier (2026-11-16)

`GUFO_GLM5_FAST_PROFILE=w8a8` routes the dense projections of the
wide-chunk dispatch through the shared Qwen prefill kernels
(`WKQuantA8BlockedWmmaGEMM`): FP32 activation rows are quantized once per
chunk into tiled Q8_1 and consumed against the native Q5_K/Q6_K/Q8_0
weight encodings with per-K32 INT8 scales. Token-1 decode rows keep the
F16 DenseWmma/scalar tiers; scratch is sized once at load and any
stride/fit mismatch falls through to the default tier. Staging-only, no
default-path change, no new CLI flag.

Offline bench A/B (GPU-exclusive, alternating, repetitions 2): pp2048
(n-gen 32) `260.10 ± 3.12` → **`270.78 ± 1.57`** (**+4.1 %**, disjoint
error bars), tg32 flat by design (11.87 → 11.90); pp16384 (n-gen 1)
`209.83 ± 0.23` → `212.94 ± 4.05` (+1.5 %, inside noise)
(`/tmp/opencode/gufo_bench_w8a8pp2k.log`,
`/tmp/opencode/gufo_bench_w8a8pp16k.log`). Well below the PLAN estimate:
the dense operands are only ~2.2 s/pass of a 14 s pp chain, so the
routed-expert half of B7 remains the real prize.

Quality: greedy tg64 `output_sha256` **bit-identical** (B10 baseline
`1d79e532…`) between tier-off and tier-on at prompt depths 2048 and 4096
(`/tmp/opencode/gufo_bench_w8a8sha.log`); new contract
`glm5_next.rocm_w8a8_dense` — 13 shapes across Q5_K/Q6_K/Q8_0 over both
launch branches (BN 64/128), relative NMSE vs the scalar `GemvRows`
reference ≤ 1e-4 (observed worst 1.5e-5), deterministic replay and
non-finite checks (`/tmp/opencode/gufo_bench_w8a8unit2.log`). Suite
`glm5_next` **10/10** on the candidate (`/tmp/opencode/gufo_bench_glm5tests2.log`).

### B12 — W8A8 dense tier promoted to default (2026-11-16)

`GUFO_GLM5_FAST_PROFILE` removed: `Executor::Create` now always sizes the
Q8_1 activation scratch and enables the B11 tier; stride/fit mismatch and
scratch allocation failure still fall through to the exact
DenseWmma/GemvRows paths. Staging-only, no new CLI flag or env.

Promotion gate, same pre-flip binary, env OFF vs ON over
`gufo serve --context 131072 --prefill-chunk 2048 --think off`,
`serving_bench` canonical prompt at `--prompt-repeat 43`
(server-reported `prompt_tokens` 2063 both sides), greedy max-tokens 32,
concurrency 1, one warmup + two timed samples, GPU-exclusive:
prefill `179.83 / 181.55` → **`185.93 / 189.50`** (**+3.9 %**, sample
ranges disjoint), decode flat (server `decode_tps` 12.0/11.7 →
12.0/11.8) (`/tmp/opencode/w8a8_gate_off.json`,
`/tmp/opencode/w8a8_gate_on.json`).

Post-flip on the default build: greedy tg64 `output_sha256`
**bit-identical** to the B10/B11 baseline (`1d79e532…`) at prompt depths
2048 and 4096 (offline pp2048 271.42 / pp4096 253.83,
`/tmp/opencode/gufo_bench_w8a8verify.log`); suite `glm5_next` **10/10**
on the gpu-test tree including the strict `rocm_forward` golden and the
`rocm_w8a8_dense` contract (`/tmp/opencode/gufo_bench_w8a8suite.log`).
B11's longer-depth evidence (pp16384 +1.5 %, noise) carries over.

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

### B13 — W8A8 dense tier wave64 192×128 tile (2026-11-17)

The B11/B12 dense tier's `LaunchTuned` batch ≥ 96 branch swaps to a
64-wavefront 192×128 `WKQuantA8BlockedWmmaGEMMKernel<192, 128, 2, 4, 1>`
slab (new TU `staging/models/glm5_next/kernels/rocm/w8a8_wave64.hip.cpp`,
`-mwavefrontsize64`, from the Flint `v64 192x128 bk2 4x1` sweep lead);
batch < 96 keeps the shipped wave32 128×64 tile and all fall-through
tiers are unchanged. No flag, promoted to default.

Offline bench A/B (GPU-exclusive, alternating, repetitions 2): pp2048
(n-gen 32) `266.66 / 272.00` → **`278.00 / 284.37`** (**+4.4 %**, sample
ranges disjoint), tg32 flat (`/tmp/opencode/gufo_bench_t192ab.log`).
Serve-protocol gate (`gufo serve --context 131072 --prefill-chunk 2048`,
`serving_bench` canonical case greedy max-tokens 32, concurrency 1,
two alternating rounds): stage prefill `212.1 / 211.2` →
**`215.8 / 215.1`** (+1.8 %, candidate ≥ base in all four pairs), decode
flat 11.6–12.4 (`/tmp/opencode/t192/gate_*.json`,
`/tmp/opencode/gufo_bench_t192g4.log`).

Quality: greedy tg64 `output_sha256` **bit-identical** to the B10–B12
archived value (`1d79e532…`) on both arms; suite `glm5_next` **10/10** on
the candidate including the strict `rocm_forward` golden replay,
`rocm_device_model` and the `rocm_w8a8_dense` contract (NMSE ≈ 1.4e-5
across types, non-multiple m covered).

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
