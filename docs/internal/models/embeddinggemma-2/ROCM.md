# ROCm 10.1 migration

October 26, 2026, Strix Halo `gfx1151`, performance power profile. No large
resident-workload degradation was detected between the flake's ROCm 7.2.3
and 10.1 packages. Text, image and audio reference checks pass without kernel,
precision or tolerance changes. Individual concurrent-request timing remains
qualified below.

Both production builds use PR #768 after merging main `c811b451`, source
`0fe9c9c3` (tree `293d8b2d`). The control uses HIP Clang 22/GCC 15.3;
10.1 uses HIP Clang 24/GCC 16.2. This is a whole pinned toolchain comparison,
including hipBLASLt tuning, rather than an isolated runtime-library swap.
Binary, model, fixture and harness hashes, toolchains, temperatures, every
measured request latency, all initial results and focused reverse-order repeats
remain in the ignored `artifacts/rocm10-embedding-768/` directory on the
measurement host, alongside raw responses, logs and orchestration scripts.

## Numerical qualification

Each build passes 43 text SDK requests, 26 image requests and 39 audio
requests. These cover reference agreement, input order, batching, concurrency,
dimensions/base64, context limits, malformed media, cancellation and recovery.
The independent float32 fixtures and retained bounds are unchanged; their
provenance is in [QUALITY.md](QUALITY.md).

| ROCm 10.1 reference probes | Worst cosine | Required |
| --- | ---: | ---: |
| Ten text vectors | 0.9999848 | ≥0.9999 |
| Seven image/mixed-input vectors | 0.9999570 | ≥0.9999 |
| Ten audio/interleaved vectors, two transports | 0.9999687 | >0.999 |

Seven image cases and ten audio cases additionally pass ten warmed requests
each, with exact expected token counts and finite normalized vectors. This
checks upstream numerical agreement, not retrieval accuracy or MTEB scores.

## Resident HTTP performance

Loopback `POST /v1/embeddings`, same BF16 checkpoint, prompts, work counts and
server flags. Each standard text row has one warmup followed by 60 requests.
Latency is complete client HTTP time, including response parsing. Loading is
excluded. The server does not expose separate embedding phase timers.

| Concurrency × inputs × words | Tokens/request | 7.2.3 median ms | 10.1 median ms | Change | 7.2.3 p95 ms | 10.1 p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 × 1 × 16 | 40 | 6.45 | 6.43 | -0.3% | 6.60 | 6.59 |
| 1 × 1 × 128 | 404 | 12.28 | 13.18 | +7.3% | 12.57 | 13.32 |
| 1 × 1 × 512 | 1,940 | 57.79 | 55.82 | -3.4% | 58.72 | 56.92 |
| 1 × 1 × 2100 | 8,072 | 323.59 | 332.50 | +2.8% | 326.04 | 333.27 |
| 1 × 8 × 128 | 3,495 | 90.46 | 88.10 | -2.6% | 91.03 | 88.60 |
| 1 × 32 × 128 | 15,670 | 376.67 | 374.83 | -0.5% | 377.73 | 376.34 |
| 4 × 1 × 128 | 404 | 45.09 | 45.39 | +0.7% | 45.90 | 46.02 |
| 16 × 1 × 128 | 404 | 152.92 | 152.21 | -0.5% | 156.69 | 156.99 |
| 16 × 1 × 16 | 40 | 21.34 | 21.77 | +2.0% | 24.58 | 24.94 |
| 4 × 8 × 128 | 3,495 | 326.22 | 325.76 | -0.1% | 326.90 | 328.12 |
| 16 × 8 × 128 | 3,495 | 1304.69 | 1302.83 | -0.1% | 1307.01 | 1304.70 |

An unrelated Nix/llama.cpp CPU build overlapped the initial candidate's early
text rows. Those timings remain visibly inconclusive in the record. Idle
reverse-order repeats cover the five affected text rows, then only the
single-batch/concurrency rows with individual timing flags; the unflagged
16 × 1 × 128 row retains the initial pair. No runs were discarded or averaged
together. The 404-token single-input row is 0.90 ms (7.3%) slower; the
8,072-token row is 8.91 ms (2.8%) slower. All sequential text requests pass
the combined >5% and >10 ms regression gate after the focused repeats.

Five individual concurrent-request flags persist: one at 4 × 1 × 128 and four
at 16 × 8 × 128. Matching submission indices does not match device admission
or batch history. Repeating unchanged toolchains also produces such flags
(seven on the unchanged 7.2.3 16 × 8 × 128 control). The repeated cross-toolchain
median, p95 and sorted latency distributions show no material slowdown, but
the individual admission-sensitive timings remain **inconclusive**. This is
not an unconditional per-request latency-parity claim.

For media, one warmup and ten sequential requests per independent fixture
case yield image median changes from −4.2% to +2.4%, and audio/interleaved
median changes from −29.1% to +0.04%. Every individual media request passes
the same combined gate. The short-audio improvement is not claimed as a
speedup: the control was affected by host activity and no gain-focused repeat
was made. These probes include the shared text backbone, the vision/audio
towers and the interleaved path. No cold-load or full retrieval-corpus claim
is made. Published [BENCHMARKS.md](BENCHMARKS.md) figures are unchanged.

## Reproduce

Build sequentially before measuring on an idle GPU, retaining both packages:

```sh
nix build --out-link result-rocm7
nix build --out-link result-rocm10
./result-rocm10/bin/gufo serve embedding --model models/embeddinggemma-2 \
  --port 18768 --log-level debug > artifacts/embedding-server.log 2>&1
```

With the repository's Python environment (OpenAI SDK, NumPy and Pillow), run
these against each server separately. Wait for `/ready` first:

```sh
python3 tests/functional/embeddings.py --base-url http://127.0.0.1:18768/v1 \
  --output artifacts/embedding-text.json
python3 tests/functional/embedding_images.py --base-url http://127.0.0.1:18768/v1 \
  --output artifacts/embedding-images.json
python3 tests/functional/embedding_audio.py --base-url http://127.0.0.1:18768/v1 \
  --output artifacts/embedding-audio.json --server-log artifacts/embedding-server.log
python3 tools/models/embedding_gemma2/http_bench.py \
  --base-url http://127.0.0.1:18768/v1 --requests 60
```

The media timing probes use the same fixture inputs as the functional scripts,
warm each case once, then retain ten complete HTTP latencies and vectors.
Repeat only flagged workloads, reversing toolchain order and retaining earlier
results. Keep the checkpoint, power profile, tuning warmup and server options
fixed.

## Hosted HIP compiler cache

The hosted build exposed a separate stale-object problem in ccache 4.13.6:
Clang's bundled HIP preprocessor output produces a cache manifest with zero
included headers. Adding a host function to an included HIP header still gives
a direct cache hit; the reused object lacks the new symbol and fails to link.
The focused reproduction under the ignored
`artifacts/rocm10-embedding-768/ccache-repro/` directory retains the source,
flags and failing cached/successful uncached link controls.

The Nix package now leaves HIP compilation uncached, matching the existing
Linux CMake policy, while retaining C/C++ caching. This prevents reuse of stale
HIP objects from the main timing baseline or an earlier CI cache. No model
code, compiler flags or numerical bounds change; the migration measurements
above already used uncached HIP compilation.
