# MiniMax H3

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

Gufo runs the text-only FL2VA path on Linux gfx1151: prompt encoding, the
50-block BF16 transformer, F32 Euler integration, VisualVAE, AudioVAE, and
MP4 composition. Model code lives in
[`src/models/minimax_h3`](../../../src/models/minimax_h3).
First/last-frame conditioning, image references, Ref2VA, and 2K regeneration
are unsupported. Model tensor computation runs on HIP; host work handles
loading, planning, progress, and media composition.

## Model acquisition

| Artifact | Pinned identity |
| --- | --- |
| MiniMaxAI/MiniMax-H3, FL2VA | `42ed227ee7df40d41602854ae760620d6eb651fe` |
| Source manifest SHA-256 | `00a83367b87017f1f3d5547a963b20567a46d7aea44f0aac495c816147720d89` |
| MiniMax H3 Community License Agreement | August 2, 2026 |
| License SHA-256 | `59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44` |

The operator obtains the checkpoint and accepts its terms independently.
Gufo distributes engine code and binaries, without H3 weights, tokenizer
payloads, converted weights, or private authorization documents. Build,
installation, and server startup do not download this model.
See [third-party notices](../../../THIRD_PARTY_NOTICES.md) for the model and
implementation licensing records.

After obtaining access, download the selected family and verify it:

```sh
nix develop -c hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include LICENSE README.md model_index.json "FL2VA/**" \
  --local-dir /var/llms/huggingface/MiniMax-H3

nix develop -c python3 tools/h3/gufo-h3-manifest.py \
  --model-root /var/llms/huggingface/MiniMax-H3 \
  --verify src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json
```

The inventory tool checks revision metadata, payload hashes, safetensors
headers, tensor shapes/dtypes, and shard coverage without executing
model-provided Python. Runtime inspection validates the pinned inventory
before device allocation or job admission. Missing, partial, mismatched,
unsupported, or unsafe shard paths fail loading.

## CLI and server

Build with `nix build` and use `result/bin/gufo`. The presets are frozen in
[`generation.cpp`](../../../src/models/minimax_h3/generation.cpp):

| Preset | Internal canvas | Output canvas | Sigma points / transitions | Active blocks | Velocity reuse interval |
| --- | --- | --- | ---: | ---: | ---: |
| `exact` | 512x512 | 512x512 | 50 / 49 | 50 | 1 |
| `exact-1344x768` | 1344x768 | 1344x768 | 50 / 49 | 50 | 1 |
| `fast` | 384x384 | 512x512 | 20 / 19 | 45 | 2 |
| `aggressive` | 320x320 | 512x512 | 20 / 19 | 40 | 3 |
| `dev` | 256x256 | selected 256x256 frames | 5 / 4 | 50 | 1 |

`exact` follows the released Diffusers schedule. Fast and aggressive are
approximate native presets with gate-ranked block thinning and extrapolated
velocity reuse; their schedule transitions do not all execute the transformer.
They protect blocks 0, 1, and 49 and execute retained blocks in model order.
Token reduction is disabled in every preset.

The CLI's `dev` preset uses the real 22-frame latent window, decodes frames
0, 11, and 21, and skips audio/muxing:

```sh
./result/bin/gufo video \
  --model /var/llms/huggingface/MiniMax-H3 \
  --preset dev --seed 42 --frames-dir /tmp/h3-preview --profile \
  "A red fox walking through snow"
```

For a complete user-requested MP4, select `exact` with
`--output /tmp/h3.mp4`, or `exact-1344x768` for the full-resolution preset.
`--latents-dir` optionally retains final F32 latents and provenance for an
offline comparison. `--attention-kernel scalar` is an explicit diagnostic;
the default is `row-parallel`.

Start a video-only server:

```sh
./result/bin/gufo serve \
  --host 127.0.0.1 --port 8080 \
  --video-model /var/llms/huggingface/MiniMax-H3 \
  --video-root /var/llms/huggingface/gufo-h3-jobs --video-ttl 3600
```

`POST /v1/videos` accepts JSON or multipart form data. For example:

```json
{
  "model": "minimax-h3",
  "prompt": "A red fox walking through snow",
  "size": "512x512",
  "seconds": 5,
  "gufo": {"seed": 42}
}
```

Requests use the released 24-fps, `17n + 5` frame alignment and 345-frame
ceiling. Durations in the 5–15-second range must align within that ceiling
(14.375 seconds of output); five seconds aligns to 124 frames. The
one-second/22-frame diagnostic extension is also supported. The prompt
encoder enforces **4,096 tokens after tokenization and NFC normalization**.
The HTTP byte budget is a separate request-resource limit.

The API's `dev` preset requires `size: "256x256"` and
`gufo.output_format: "ppm"` and returns one selected frame. An explicit
`gufo.frames` must match the aligned duration. Unsupported conditioning and
conflicting preset/model selections are rejected.

One worker executes H3 jobs, with one additional queued job; excess admissions
receive HTTP 429. Status, progress, cancellation, content ranges, expiry, and
restart recovery are described in [the server guide](../../SERVER.md).
`DELETE /v1/videos/{id}` cancels work through every phase and reclaims its
artifacts. Parameter reports retain prompt hashes and generation controls;
persisted status omits prompt text and private paths.
