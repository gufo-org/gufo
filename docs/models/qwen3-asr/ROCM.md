# ROCm 10.1 migration

October 24, 2026, Strix Halo `gfx1151`. The existing ASR implementation passes
the focused migration checks without a kernel, precision or BLAS-policy change.

ROCm 7.2.3 uses pinned Nix dependencies, Clang 22 and GCC 15.3. ROCm 10.1.0
uses Clang 24/GCC 15.2 in `rocm/dev-ubuntu-26.04:10.1.0-full`, with the matching
repository ROCr queue-memory patch. These container checks do not qualify the
optional Nix ROCm10 package. Builds and inference run separately.

| Migration check | Result |
| --- | --- |
| Production builds | Pass on both toolchains |
| Independent convolution/attention, argmax ties/nonfinite values | Pass, original limits |
| Official 15.05-second English fixture | 49/49 token IDs and transcript on both |
| Complete audio encoder versus official | Both: cosine 0.999852, relative L2 0.0172329 |
| Full prefill logits versus official | Both: cosine 0.999019, relative L2 0.0473274 |
| Long recording, small context, streaming, cancellation and reuse | Native 10.1 transcription checks pass |

The checkpoint and independent reference revisions are pinned in
[QUALITY.md](QUALITY.md). The current matched control has the same numerical
errors on both compilers; older dated quality figures are not substituted for
that control. The existing hipBLASLt selection remains unchanged.

One warmup followed by three resident CLI transcriptions, with the exact
official float32 waveform, context and 128-token cap:

| Phase | ROCm 7.2.3 | ROCm 10.1 |
| --- | ---: | ---: |
| Audio encoder | 34.27–34.46 ms | 34.32–34.42 ms |
| Text decoder | 947.13–947.47 ms | 943.47–944.46 ms |
| Complete resident transcription | 988.56–989.93 ms | 985.74–986.55 ms |

Each individual phase passes the 5% and 3 ms timing gate. Loading is excluded.
Raw results are retained under `gufo-models-rocm101-20261023/port-models` on
the test host. [BENCHMARKS.md](BENCHMARKS.md) keeps its published figures.
Multilingual and long-form corpus quality remain outside this bounded check.
