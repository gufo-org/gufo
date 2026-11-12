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
