# DeepSeek V4 Flash evaluation

**Target parity remains unresolved.** Antirez/ds4 is an independent comparison,
not official ground truth. DSpark/AR agreement cannot detect shared target errors.
This documentation/build update does not establish a quality improvement.

## Quality status

| Check | Retained result |
| --- | --- |
| Optimized trajectory, September 19 | **115/128** top-1, rank sum 145, worst rank 4; required ≥116, ≤142, ≤3 |
| Optimized versus Debug | **33/2327** greedy choices differ; cause isolated to build-sensitive `backend.hip.cpp`, not resolved |
| Hosted continuation likelihood | 100 prompts / 2313 tokens; optimized-minus-Debug NLL +0.001169, 95% interval −0.003416 to +0.005655 |
| Post-prefill full logits | **4/20** checks fail against the historical antirez control; post-decode vectors pass |
| Deterministic replay | **1280/1280** choices and 20 logit vectors repeat exactly |
| Operator/state controls | Independent FP64 formulas, 156 exact top-k cases and 736 DSpark replay choices pass |
| Serving, September 21 | AR/DSpark prefix/disk restore, sampled replay, EOS isolation, cancellation and late prefill pass |
| Capability, September 10 | **53/75**, 22 failures, zero execution errors; nine length finishes (context 32768, output limit 16000) |
| Current original-checkpoint / capability qualification | **TODO**; ask before rerunning the full 75-question comparison |

The [prefill comparison](artifacts/antirez-ds4-ar-comparison.json) and
[formula audit](artifacts/prefill-formula-audit.json) retain the four alerts:

| Context / call size | RMSE | Cosine | Max logit error | Max probability difference |
| --- | ---: | ---: | ---: | ---: |
| 4K / 4K | 0.9691 | 0.98006 | 5.3386 | 5.90 percentage points |
| 12K / 4K | 1.1819 | 0.96935 | 5.4680 | 39.81 percentage points |
| 16K / 2K | 0.8128 | 0.98681 | 5.8521 | 11.63 percentage points |
| 16K / 4K | 1.3476 | 0.96214 | 5.8032 | 9.80 percentage points |

Limits remain RMSE ≤1.12, cosine ≥0.979 and maximum error ≤5. Never relax
quality gates to admit an optimization. The official HC formula projects raw
activations before RMS scaling; Gufo follows that order. Copying the historical
antirez FP16 rounding fails the independent formula oracle. The
[captured-input audit](artifacts/captured-operator-audit.json) finds accumulated
attention error, without proving a single root cause. Resolution still requires
operator/layer ablation and original-checkpoint continuation checks.

Historical quality sources: official Flash 0731
[`7872f01b`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/7872f01b1d1fe23eabc4c98b48bffcef5a386062),
antirez/ds4 [`6289c516`](https://github.com/antirez/ds4/tree/6289c516273979173abbc062209a81dd3706b804).
These results do not qualify the newer performance-reference pin below.

## DSpark

p/q acceptance and normalized residual correction preserve the target sampling
distribution. Request-private RNG/controller state supports seeded replay for
the same execution configuration and schedule; live timings do not affect
sampled token decisions. [Sampling evidence](artifacts/dspark-sampling.json),
[controller costs](artifacts/cost-calibration.json) and
[qualification summary](artifacts/quality-qualification.json) retain their own
build identities and acceptance metrics.

Interrupted-chat checks cover reasoning/visible output, retained/omitted
reasoning, a third turn and disk restart at 262144-token capacity. Replay uses
the same prefill/decode history; it does not close the target arithmetic gaps.

## Maintained checks

Tests: `tests/models/deepseek_v4_flash`; commands: [tools/ds4](../../../tools/ds4/README.md).
Run affected checks while iterating; retain full model gates for arithmetic changes.

| Suite | Coverage |
| --- | --- |
| `fast` | Template, CLI, dataset, evaluation and sampling contracts |
| `kernels` | Independent projection/HC formulas, attention and exact top-k |
| `model` | Target trajectories/full logits, DSpark C1–C8/state restore, HTTP sampling, caching, cancellation and three-turn chat |
| `reference` | Matched antirez differential comparison; not official-model parity |

```sh
nix develop -c tools/ds4/check.py fast
nix develop -c tools/ds4/check.py kernels
nix develop -c tools/ds4/check.py model --model "$MODEL" --dspark-model "$DSPARK"
nix develop -c tools/ds4/check.py reference --model "$MODEL" \
  --upstream /path/to/pinned-antirez-checkout --output /tmp/ds4-reference
```

## Benchmark method

The proposed grid is pp2048/tg128 at cached depths 0, 4K, 8K, 12K, 16K, 32K,
64K and 128K; C1/2/4/6/8 use the same d0 mixed/repetitive prompts. Prefill all
sessions before timed decoding, then sum individual request decode rates.
One warmed sample per point; repeat only to investigate a discrepancy. Preserve
AR/speculative completion hashes and acceptance details in artifacts.
Loading uses C1/DSpark at capacity 262144; memory uses C1/AR at the same capacity.

The unmodified antirez server returns token/cache counts but **no per-request
prefill/decode durations**. On ROCm, `--batched-session` disables DSpark even
at C1; omit it for single-user speculation. C>1 DSpark reference cells remain
**TODO** pending upstream support. Stage timing and prepared-cohort reuse must be
qualified before its throughput sweep; the driver rejects those unqualified
runs before loading a model. Whole-request wall time cannot fill a tg cell.
`ds4-bench` supports single-user AR and DSpark (`--dspark --mtp-model`), with
per-frontier pp/tg CSV rates; it has no multi-session loop. It is a candidate
for the single-user comparison: match Gufo's exact prompt/frontier and decode
token accounting first. `ctx_tokens` includes the newest prefill; `prefill_tokens`
is only the increment. Retain full `gen_tps`, not its steady-only alternative.
Existing CLI rates are not reused. The full sweep waits for template review.

Reference: [`antirez/ds4 0aaea5a2`](https://github.com/antirez/ds4/tree/0aaea5a238fb41a35106a551e73c8409dfb751ac),
ROCm 7.2.3 / gfx1151, official
[`strix-halo` build](https://github.com/antirez/ds4/blob/0aaea5a238fb41a35106a551e73c8409dfb751ac/Makefile).
The optional [Nix recipe](../../../.devops/nix/ds4-reference.nix) follows the
[upstream prerequisites](https://github.com/antirez/ds4/blob/0aaea5a238fb41a35106a551e73c8409dfb751ac/docs/STRIX_HALO.md);
[`fedeizzo/ds4`](https://github.com/fedeizzo/ds4/blob/f386629c5217ced3003d2b0d67b461c9265d3f6c/flake.nix)
provided an earlier packaging example. It is excluded from Gufo, the normal
dev shell and hosted checks.

[Reference smoke, September 23 UTC](artifacts/reference-smoke.json): Nix build
and installed commands pass; four C1/C2 AR HTTP replies match (32 tokens each).
Native DSpark completes two frontiers × eight tokens with snapshot restoration
and valid CSV rates. These bounded checks do not establish model quality or
benchmark performance.

```sh
nix build .#ds4-reference --out-link result-ds4-reference
./result-ds4-reference/bin/ds4-server --rocm --model "$MODEL" \
  --ctx 4096 --host 127.0.0.1 --port 8081
# DSpark: append --dspark --mtp-model "$DSPARK"
# HTTP requests: set thinking.type="disabled" and temperature=0.
# Native single-user AR; append the same DSpark options for speculation:
./result-ds4-reference/bin/ds4-bench --rocm --model "$MODEL" \
  --chat-prompt-file /path/to/prompt.txt --ctx-start 2048 --ctx-max 2048 \
  --gen-tokens 128 --csv /tmp/ds4.csv

nix develop -c python3 tools/bench/model-bench.py --model deepseek-v4-flash tables
nix develop -c python3 tools/bench/model-bench.py --model deepseek-v4-flash render
```
