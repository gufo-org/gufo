# GLM-5.3-Flash: model rules

## Benchmark context size

All GLM-5.3-Flash benchmarking and optimization work (gates, A/Bs, lever
runs, profiles) starts `gufo serve` with `--context 131072`. Never leave
`--context` unset: 0 resolves to the native 1,048,576-token context, whose
F16 MLA latent KV is preallocated at load (~28 GiB on top of the 90.9 GiB
IQ1_M weights) and leaves Strix Halo UMA at a few hundred MB free, which
OOM-kills the host when anything else touches memory. At 131072 the KV costs
~3.5 GiB and runs hold ~30 GiB headroom. Numbers recorded with a different
server context are a different protocol and must say so.

Results and protocol: [docs/models/glm-5.3-flash/BENCHMARKS.md](../../../docs/models/glm-5.3-flash/BENCHMARKS.md).

## Measurement command

Offline timing scan: `gufo bench --model <GGUF> --n-prompt <N> --n-gen <N>
--repetitions 2` (registered through `staging/integration/staging_cli.cpp`;
greedy, C1 only, optionally `--speculative mtp --draft-tokens N` for the
nextn chain). It sizes the caches to the requested
depth, loads in ~30 s and prints the standard table.

Gate and matched-A/B numbers must still come from `gufo serve` over HTTP
with server-reported timings (`tools/gufo/serving_bench.py`) to replicate
the archived capture protocol; the offline bench reads ~5 % faster on pp
(~6 t/s at 2048) because it excludes the serve scheduler and HTTP.
`gufo serve` has no `/tokenize` endpoint; prompt depth is set with
`serving_bench --prompt-repeat` and the exact depth is read back from
the server-reported `prompt_n` in the result JSON.
