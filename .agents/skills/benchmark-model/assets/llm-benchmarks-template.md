# <Model> benchmarks

<Hardware, model/quantization, Gufo/reference versions and sampling setup.>
Positive gain favors Gufo. **TODO** means unmeasured.
Quality and measurement details belong in `EVALUATION.md`.

## Single user, autoregressive

pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar -->
<!-- /bench -->

## Single user, speculative

One table per quantization: shared pp, separate mixed/repetitive tg and gains.
pp is the highest measured rate per engine and depth across both text types.

<!-- bench:single-<spec> -->
<!-- /bench -->

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
Prefill every session before timed decoding; sum individual request decode rates.

<!-- bench:multi-ar -->
<!-- /bench -->

## Multiple users, speculative

Same pp2048 mixed/repetitive prompts as single-user d0, tg128. C1 directly
cross-checks that row. Prefill every session before timed decoding.
One table and figure per quantization.

<!-- bench:multi-<spec> -->
<!-- /bench -->

## Loading time

C1, capacity <tokens>, <speculative mode>. Cold model files to HTTP readiness.

<!-- bench:loading -->
<!-- /bench -->

## Memory occupation

<Context capacity and any artifact mismatch needed to interpret these numbers.>

<!-- bench:memory -->
<!-- /bench -->
