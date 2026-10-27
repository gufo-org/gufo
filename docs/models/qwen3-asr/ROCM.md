# ROCm 10.1 migration

October 24, 2026, Strix Halo `gfx1151`. The existing ASR implementation passes
the focused migration checks without a kernel, precision or BLAS-policy change.

ROCm 7.2.3 uses pinned Nix dependencies, Clang 22 and GCC 15.3. ROCm 10.1.0
was first checked with Clang 24/GCC 15.2 in
`rocm/dev-ubuntu-26.04:10.1.0-full`, then built through `nix build`
with cached Clang 24/GCC 16.2 and the matching repository ROCr patch.
Source `65bf6798` includes main `67d1f6ac`; this model's kernels are unchanged
from the independently checked `6d43e786`. Builds and inference run separately.

| Migration check | Result |
| --- | --- |
| Production builds | Pass on 7.2.3, container 10.1 and Nix 10.1 |
| Independent convolution/attention, argmax ties/nonfinite values | Pass, original limits |
| Official 15.05-second English fixture | 49/49 token IDs and transcript on 7.2.3 and both 10.1 builds |
| Complete audio encoder versus official | Both: cosine 0.999852, relative L2 0.0172329 |
| Full prefill logits versus official | Both: cosine 0.999019, relative L2 0.0473274 |
| Long recording, small context, streaming, cancellation and reuse | Native container and Nix 10.1 transcription checks pass |

The checkpoint and independent reference revisions are pinned in
[QUALITY.md](QUALITY.md). The current matched control has the same numerical
errors on both compilers; older dated quality figures are not substituted for
that control. The existing hipBLASLt selection remains unchanged.

Final Nix production control: one warmup followed by three resident CLI
transcriptions, with the exact official float32 waveform, context and
128-token cap:

| Phase | ROCm 7.2.3 | ROCm 10.1 |
| --- | ---: | ---: |
| Audio encoder | 33.95–34.13 ms | 33.97–34.52 ms |
| Text decoder | 948.19–948.99 ms | 944.63–949.51 ms |
| Complete resident transcription | 989.97–990.38 ms | 986.20–992.04 ms |

Each individual phase passes the 5% and 3 ms timing gate. Loading is excluded.
Raw results are retained under `gufo-models-rocm101-20261023/port-models` on
the test host. [BENCHMARKS.md](BENCHMARKS.md) keeps its published figures.
Multilingual and long-form corpus quality remain outside this bounded check.
