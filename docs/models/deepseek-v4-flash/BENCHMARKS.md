# DeepSeek V4 Flash benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Antirez Flash 0731 mixed
`IQ2XXS` target, DSpark support sidecar. Greedy, thinking off.
Reference: antirez/ds4 `0aaea5a2` (ROCm).

Template for review; measurements are pending. Positive gain favors Gufo.
[Quality and measurement details](EVALUATION.md#benchmark-method).

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar -->
| DS4 Flash IQ2XXS AR<br>Depth (tokens) | Gufo pp (tok/s) | antirez/ds4 pp (tok/s) | Gain | Gufo tg (tok/s) | antirez/ds4 tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | TODO | TODO | TODO | TODO | TODO | TODO |
| 4,096 | TODO | TODO | TODO | TODO | TODO | TODO |
| 8,192 | TODO | TODO | TODO | TODO | TODO | TODO |
| 12,288 | TODO | TODO | TODO | TODO | TODO | TODO |
| 16,384 | TODO | TODO | TODO | TODO | TODO | TODO |
| 32,768 | TODO | TODO | TODO | TODO | TODO | TODO |
| 65,536 | TODO | TODO | TODO | TODO | TODO | TODO |
| 131,072 | TODO | TODO | TODO | TODO | TODO | TODO |
<!-- /bench -->

## Single user, DSpark

pp is the highest measured rate per engine and depth across mixed/repetitive
text.

<!-- bench:single-dspark -->
| DS4 Flash IQ2XXS DSpark<br>Depth (tokens) | Gufo pp (tok/s) | antirez/ds4 pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | antirez/ds4 tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | antirez/ds4 tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 4,096 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 8,192 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 12,288 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 16,384 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 32,768 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 65,536 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 131,072 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
<!-- /bench -->

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.

<!-- bench:multi-ar -->
| DS4 Flash IQ2XXS AR<br>Users | Gufo AR (tok/s) | antirez/ds4 AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | TODO | TODO | TODO |
| 2 | TODO | TODO | TODO |
| 4 | TODO | TODO | TODO |
| 6 | TODO | TODO | TODO |
| 8 | TODO | TODO | TODO |
<!-- /bench -->

## Multiple users, DSpark

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. C1 cross-checks the single-user table.
The pinned antirez ROCm server disables DSpark when batching; its C>1 cells
remain TODO.

<!-- bench:multi-dspark -->
| DS4 Flash IQ2XXS DSpark<br>Users | Gufo mixed (tok/s) | antirez/ds4 mixed (tok/s) | Gain | Gufo repetitive (tok/s) | antirez/ds4 repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | TODO | TODO | TODO | TODO | TODO | TODO |
| 2 | TODO | TODO | TODO | TODO | TODO | TODO |
| 4 | TODO | TODO | TODO | TODO | TODO | TODO |
| 6 | TODO | TODO | TODO | TODO | TODO | TODO |
| 8 | TODO | TODO | TODO | TODO | TODO | TODO |
<!-- /bench -->

## Loading time

C1, context capacity 262144, DSpark. Cold target/sidecar files to HTTP readiness.

<!-- bench:loading -->
| DS4 Flash IQ2XXS<br>Target | Gufo ready (s) | antirez/ds4 ready (s) | Gain |
| --- | ---: | ---: | ---: |
| IQ2XXS | TODO | TODO | TODO |
<!-- /bench -->

## Memory occupation

C1, context capacity 262144, AR. Peak memory reported by HIP.

<!-- bench:memory -->
| DS4 Flash IQ2XXS AR<br>Workload | Gufo GiB | antirez/ds4 GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | TODO | TODO | TODO |
| 16K prefix, pp4096 + tg128 | TODO | TODO | TODO |
<!-- /bench -->
