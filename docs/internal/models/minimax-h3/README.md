# MiniMax H3 implementation notes

Developer detail behind the [public model page](../../../models/minimax-h3.md).
[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md) · [ROCm](ROCM.md)

Gufo runs the FL2VA text-to-video and keyframe-conditioned paths on Linux
gfx1151: prompt encoding with a native Qwen3-VL vision tower, the 50-block
BF16 transformer, F32 Euler integration, VisualVAE encode/decode, AudioVAE,
and MP4 composition. Model code lives in
[`src/models/minimax_h3`](../../../../src/models/minimax_h3).
Image references, Ref2VA, and 2K regeneration are unsupported. Model tensor
computation runs on HIP; host work handles loading, planning, progress, and
media composition.

Gufo distributes engine code and binaries, without H3 weights, tokenizer
payloads, converted weights, or private authorization documents. Build,
installation, and server startup do not download this model.
See [third-party notices](../../../../THIRD_PARTY_NOTICES.md) for the model and
implementation licensing records.

The inventory tool checks revision metadata, payload hashes, safetensors
headers, tensor shapes/dtypes, and shard coverage without executing
model-provided Python. Runtime inspection validates the pinned inventory
before device allocation or job admission. Missing, partial, mismatched,
unsupported, or unsafe shard paths fail loading.

## CLI and server

Build with `nix build` and use `result/bin/gufo`. The presets are frozen in
[`generation.cpp`](../../../../src/models/minimax_h3/generation.cpp):

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

For a complete user-requested MP4, select `exact` with
`--output /tmp/h3.mp4`, or `exact-1344x768` for the full-resolution preset.
`--latents-dir` optionally retains final F32 latents and provenance for an
offline comparison. `--attention-kernel scalar` is an explicit diagnostic;
the default is `row-parallel`.

Keyframe conditioning (issue #177) anchors the first frame, the last frame,
or both. `first_frame` and `last_frame` carry a base64 image data URL
(JSON) or a file part (multipart); remote URLs are rejected rather than
fetched. PNG, JPEG, and WebP decode through the
shared image reader in display orientation. The CLI mirrors this with
`--first-frame FILE` and `--last-frame FILE`.

As in the released pipeline, the geometry anchor (the first keyframe, or a
lone last one) sets the canvas aspect ratio. The preset's internal canvas
sets the size budget: its short edge and pixel count, so an anchored request
costs about the same as the preset. Upstream's rule then rounds both axes to
multiples of 32, which can exceed the budget by up to 3.6%, and keyframes
must lie between 1:4 and 4:1. The resolved canvas must also hold the vision
processor's 65,536-pixel minimum, below which the released processor would
upscale the keyframe; only `dev` at extreme ratios falls short and is
rejected. A 1664x928 keyframe generates at 1376x768 with
`exact-1344x768` (exactly as upstream) and at 672x384 with `exact`; presets
that upscale their output keep that scale. The geometry anchor is then
stretched onto the canvas, which only absorbs the rounding, and a last frame
beside a first is cover-cropped. The generation telemetry reports the canvas
actually used (`internal_width`/`internal_height`,
`output_width`/`output_height`); `parameters.json` keeps the requested preset.

Each keyframe enters the transformer as a clean VisualVAE latent at fixed
temporal slots with per-step condition timesteps of 0.999/0.001 against the
diffusing window, and its Qwen3-VL embeddings and deepstack taps replace the
corresponding prompt tokens. Anchored requests otherwise follow the selected
preset unchanged; presets that cannot condition report an error instead of
silently ignoring the anchor.

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
restart recovery are described in [the server guide](../../../server.md).
`DELETE /v1/videos/{id}` cancels work through every phase and reclaims its
artifacts. Parameter reports retain prompt hashes and generation controls;
persisted status omits prompt text and private paths.
