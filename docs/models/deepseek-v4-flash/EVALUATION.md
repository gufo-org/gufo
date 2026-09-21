# DeepSeek V4 Flash evaluation

Serving qualification (2026-09-21): AR/DSpark prefix and disk restoration,
mixed sampling, EOS isolation, cancellation and late-arriving prefill passed.
Snapshots retain the live attention/compression windows and committed DSpark
frontier. The loader accepts the checkpoint's integer expert-mapping tensor
without weakening tensor extent validation.

**Target parity is unresolved. No quality improvement is established by the
current arithmetic audits.** Antirez's implementation is a differential control,
not official ground truth. DSpark/AR agreement cannot detect shared target errors.

## Maintained checks

Tests live in `tests/models/deepseek_v4_flash`; commands in
[tools/ds4](../../../tools/ds4/README.md). Run affected checks while iterating.

| Check | Required coverage |
| --- | --- |
| `ds4.template`, `ds4.cli`, `ds4.dataset`, `ds4.eval` | Framing, wiring, pinned fixtures and probability/grading invariants |
| `ds4.sampling` | Exact p/q and residual distributions, conditional proposals, penalties, seed replay and confidence stopping |
| `ds4.projections` | Independent weight decoding, FP64 formulas, all IQ2 signs, MoE ownership, exact batch/scalar outputs, official HC/Sinkhorn equations |
| `ds4.attention` | Independent attention, DSpark Markov/confidence and window formulas; exact prefill scores, masks, poisoned rows, ties, scratch bounds and top-k ordering |
| `ds4.target` | Official tokens, pinned trajectory, full logits, replay, capacity equality and state isolation |
| `ds4.dspark` | Exact scalar tokens/frontier logits through C8/16K, acceptance, policy, snapshots and forks |
| `ds4.serving` | Sampling, physical batching, EOS checkpoint isolation, bounded prefill, prefix/disk caches, cancellation and exhaustion |
| `ds4.chat` | Real sampled prompt and three-turn chat after length/EOS stops, AR/DSpark token identity and active drafting |

```sh
nix develop -c tools/ds4/check.py fast
nix develop -c tools/ds4/check.py kernels
nix develop -c tools/ds4/check.py model --model "$MODEL" --dspark-model "$DSPARK"
nix develop -c tools/ds4/check.py reference --model "$MODEL" \
  --upstream /path/to/pinned-antirez-checkout --output /tmp/ds4-reference
```

The trajectory guard requires **116/128 top-1**, rank sum ≤142, worst rank ≤3.
State controls require finite logits, **RMSE ≤1.12, cosine ≥0.979, max error ≤5**.
Never relax limits to admit an optimization. Recent indexer changes preserve ten
full-logit prefill vectors across 2K/4K calls and five depths, plus five vectors
across a 64K disk snapshot, continued prefill and decode. Independent FP64
operators, 156 exact top-k cases and 736 DSpark replay choices pass.

## Target arithmetic

The optimized 2026-09-19 build scores **115/128, rank sum 145, worst rank 4**;
legacy Debug scores 116/128, 142, 3. The unchanged optimized baseline has
identical logits. Mixed-library controls isolate build sensitivity to
`backend.hip.cpp`; both use `-O3`, while `NDEBUG` changes assertion control flow
in HIP shuffle wrappers. The precise cause remains TODO.

Hosted-checkpoint continuation likelihood covers 100 prompts / 2313 tokens,
plus five smoke prompts / 14 tokens. API probabilities are saturated, so this
is not full-distribution parity:

| 100-case run | Average NLL ↓ |

Optimized/Debug disagree on **33/2327 greedy choices**. Their paired 100-case
NLL delta is +0.001169 (95% case-bootstrap interval −0.003416 to +0.005655).
Both retain 14/14 smoke tokens, but Gufo smoke NLL is 0.034343 versus 0.010908
for the same-machine upstream control. This finite sample establishes neither
equivalence nor an improvement.

[Matched prefill comparison](artifacts/antirez-ds4-ar-comparison.json): four of
20 full-logit checks fail immediately after prefill; post-decode vectors pass.
Gufo repeats exactly across 1280 choices and 20 vectors. The
[formula audit](artifacts/prefill-formula-audit.json) retains all four alerts:

| Context / call size | RMSE | Cosine | Max logit error | Max probability difference |
| --- | ---: | ---: | ---: | ---: |
| 4K / 4K | 0.9691 | 0.98006 | 5.3386 | 5.90 percentage points |
| 12K / 4K | 1.1819 | 0.96935 | 5.4680 | 39.81 percentage points |
| 16K / 2K | 0.8128 | 0.98681 | 5.8521 | 11.63 percentage points |
| 16K / 4K | 1.3476 | 0.96214 | 5.8032 | 9.80 percentage points |

The official Flash 0731 formula projects raw FP32 HC activations before RMS
scaling. Gufo follows that order with F16 GGUF weights; the Antirez wide path
normalizes first and rounds inputs to F16. Its own 12K probabilities change by
up to 40.58 percentage points across call sizes, versus 6.00 for Gufo.
Imitating that rounding fails the independent formula oracle and worsens two
probability comparisons. It is rejected.

[Captured-input audit](artifacts/captured-operator-audit.json), layers 0/2/3/42:
HC projection, Sinkhorn, reduction and normalization match independent equations;
raw FP8 KV bytes and sampled selectors match. F16 router/compressor rounding can
change nearly tied expert choices. FP32 compressor/router prototypes nevertheless
fail end-to-end qualification and were removed. The 16K trace shows accumulated
post-attention error (RMSE 0.000404 at layer 0, 0.994 at layer 42), without proving
one root cause. Local precision alone is insufficient evidence of better output.

Next correction: controlled operator/layer ablation, captured-input regression,
all four frontiers, reproducibility and official continuation checks. Full
original-checkpoint distribution and capability qualification remain **TODO**.

Source pins: official `deepseek-ai/DeepSeek-V4-Flash-0731`
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`; differential `antirez/ds4`
`6289c516273979173abbc062209a81dd3706b804`. Artifact hashes and methods remain
in the retained audit JSONs.

## DSpark

Exact p/q acceptance and normalized residual correction preserve the target
distribution. Filtered sampled C>1 uses hybrid top-8 proposals; C1/unfiltered
sampling retains point-mass proposals. Offline costs, acceptance history and
checkpoint confidence choose work; live timings never change token decisions.
RNG, policy and acceptance state are request-private and reset on prefix reuse.
Seeded replay requires the same execution configuration/schedule.

[Sampling evidence](artifacts/dspark-sampling.json) retains the 2026-09-19 C2
controls at T=1, top-p 0.95, seed 7: 17.40/21.01/15.27 per-user tok/s for
explanation/repetition/naming. [Controller costs](artifacts/cost-calibration.json)
and [qualification summary](artifacts/quality-qualification.json) retain their
own release identities; they are not a new whole-model qualification.

## Capability

The 2026-09-10 greedy C1 baseline passes **53/75** questions, with 22 failures,
zero execution errors and nine length finishes (context 32768, output limit
16000). A complete current comparison is **TODO**. The first 16 candidate cases
repeat exactly: 13/16 versus baseline 14/16; the changed answer also changes in
AR and matches DSpark. Request confirmation before the full 75-question rerun.
