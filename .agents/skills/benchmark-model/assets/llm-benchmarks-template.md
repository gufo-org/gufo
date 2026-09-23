# <Model> benchmarks

<Hardware, model/quantization, Gufo/reference versions and sampling setup.>
Positive gain favors Gufo. **TODO** means unmeasured.
Quality and measurement details belong in `EVALUATION.md`.

## Single user, autoregressive

pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar -->
<!-- /bench -->

## Single user, speculative

<!-- bench:single-<spec> -->
<!-- /bench -->

## Single user, speculative, repetitive

<!-- bench:single-<spec>-repetition -->
<!-- /bench -->

## Single user, speculative, thinking

<!-- bench:single-<spec>-thinking -->
<!-- /bench -->

## Multiple users, autoregressive

Context 4096 per user, tg128. Sum individual request decode rates; exclude
prefill and scheduling. Measure AR once per concurrency.

<!-- bench:multi-ar -->
<!-- /bench -->

## Multiple users, speculative

Mixed prompts and repetitive output share a table and figure per quantization.
Retain separate workload measurements and C1 AR quality references.

<!-- bench:multi-<spec> -->
<!-- /bench -->

## Loading time

<!-- bench:loading -->
<!-- /bench -->

## Memory occupation

<Context capacity and any artifact mismatch needed to interpret these numbers.>

<!-- bench:memory -->
<!-- /bench -->

## Image encoder

<State whether time includes image preprocessing or text prefill.>

<!-- bench:image-encoder -->
<!-- /bench -->
