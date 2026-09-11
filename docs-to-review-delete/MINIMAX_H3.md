# MiniMax H3 Integration Boundary

Status: active engineering policy, 2026-08-22

## Purpose

This document defines how gufo may integrate an operator-supplied
MiniMax H3 checkpoint without distributing model weights or treating the
engine's MIT license as permission to use the model.

It is an engineering record, not legal advice.

## Pinned Inputs

The initial supported source is:

```text
repository: MiniMaxAI/MiniMax-H3
revision:   42ed227ee7df40d41602854ae760620d6eb651fe
task:       FL2VA
license:    MiniMax H3 Community License Agreement, August 2, 2026
license sha256:
  59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44
```

The upstream port used as an implementation reference is pinned separately by
the H3 source-manifest work. No model weights are copied from that project.

## Reproducible Acquisition and Verification

After independently obtaining access from MiniMax, a clean machine downloads
only the selected family:

```sh
hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include LICENSE README.md model_index.json "FL2VA/**" \
  --local-dir /var/llms/huggingface/MiniMax-H3
```

The native inventory tool parses JSON and safetensors metadata directly. It
does not import or execute Python supplied by the model repository:

```sh
nix develop -c python3 tools/h3/gufo-h3-manifest.py \
  --model-root /var/llms/huggingface/MiniMax-H3 \
  --verify src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json
```

Verification checks the Hugging Face revision metadata for every file, hashes
every payload, validates every safetensors header/index/tensor, and rejects
missing or unreferenced shards, unsupported dtypes, Ref2VA, 2K regeneration,
pickle checkpoints, and wrong revisions before runtime device allocation.

Numerical and audiovisual work follows the frozen
[MiniMax H3 quality-oracle contract](MINIMAX_H3_QUALITY.md). Large teacher
outputs remain outside Git in schema-validated, content-addressed directories;
CI never regenerates them.

The model-private loader's phase boundaries and first gfx1151 residency
measurement are recorded in
[the MiniMax H3 residency baseline](../src/models/minimax_h3/RESIDENCY.md).

## Text-Only Prompt Encoder

The first executable H3 boundary is the FL2VA tokenizer plus the first 50
Qwen3-VL-32B decoder layers. It intentionally stops after layer index 49,
before the checkpoint's final text norm, because that BF16 hidden state is the
conditioning tensor consumed by H3.

The tokenizer is model-private. It implements the pinned NFC, GPT-2 byte
encoding, BPE merge order, added-token policies, longest/earliest special-token
matching, and empty-prompt pad behavior. ICU supplies complete Unicode NFC and
category handling. No BOS, EOS, chat template, or vision framing is inserted
for the initial text-only path.

The HIP encoder:

- validates the fixed 151936x5120 embedding and all 11 tensors in each layer;
- keeps activations and operation boundaries in BF16 with FP32 reductions and
  matrix accumulation;
- uses hipBLASLt BF16 projections plus model-private RMSNorm, Q/K norm, RoPE,
  causal GQA, residual, and SwiGLU kernels;
- loads only the embedding and layers 0 through 49;
- overlaps the next layer's read-only registered mapping/device copy with
  current-layer execution, then unregisters every host page;
- admits one prompt-encoder phase at a time;
- rejects vision spans and deepstack inputs before allocation;
- does not call the repository's Qwen language-model runtime and has no CPU
  tensor-compute fallback.

The offline teacher capture is reproducible but not part of the production
package:

```sh
nix develop -c python tools/gufo/h3_prompt_golden.py \
  --model-root /var/llms/huggingface/MiniMax-H3 \
  --layers 50 \
  --output /var/llms/huggingface/gufo-h3-oracles/\
fox-layer50-transformers.bf16
```

The raw BF16 oracle and metadata remain outside Git.

## Text-Only Layout and Sampler

The initial FL2VA sampler boundary is model-private and text-only. It accepts
checked 32-pixel canvas multiples within the released 768x1344 pixel-area
limit, aligns requests to `5 + 17*n` frames through the 345-frame ceiling, and
derives the separate video and audio latent timelines at 24 fps.

Before any large DiT allocation, the host planner freezes:

- text, target-audio, then target-video row order;
- three-dimensional MM-RoPE coordinates and per-step AdaLN row maps;
- 2x2 visual patch order and stereo audio row order;
- independent video/audio shifted sigma grids with terminal zero;
- one request PCG/Box-Muller stream, consumed by video first and then packed
  channel-major audio before the audio layout is unpacked;
- denoiser evaluation selection and bounded linear velocity extrapolation.

This is host planning and input construction, not a CPU tensor-inference
fallback. Euler sample state remains F32 in gfx1151 device memory. The HIP
kernel reads the most recent and previous F32 DiT velocities, applies the
frozen extrapolation ratio, and updates only the requested device range. Its
velocity coefficient reconstructs sigma from the float32 model timestep while
the Euler blend ratio uses the original sigma grid, matching the two-source
Diffusers arithmetic instead of collapsing it to `sigma - sigma_next`.
Keeping the final-head velocity in F32 matches the Diffusers Euler boundary
and avoids a quality-changing BF16 round trip.

The public API counts denoiser evaluations. Diffusers' nominal
`num_inference_steps=50` schedule contains 50 sigma points including terminal
zero and therefore executes 49 model forwards. The `exact` preset consequently
uses 49 evaluations; the 20-point fast and aggressive schedules use 19.

First/last-frame conditioning and ordered Ref2VA inputs are deliberately not
accepted by this text-only layout API; their row kinds remain future work.

## BF16 DiT Block Baseline

The first transformer-core boundary is one complete dense H3 block at the
released width: hidden size 5376, 56 grouped QKV heads, head dimension 128,
attention width 7168, and SwiGLU width 14336.

`DitBlockSession` loads only the selected block's eight BF16 tensors directly
from the validated safetensors inventory. Session creation allocates every
input, retained boundary, activation, comparison buffer, and rocBLAS resource.
It pins and byte-validates the gfx1151 rocBLAS solution before admitting a
run. `Run` performs no device allocation and keeps a single explicit stream.

The exact path is:

1. RMS AdaLN slots 0/1;
2. BF16 grouped QKV projection;
3. per-head Q/K RMSNorm and 3D partial MM-RoPE;
4. deterministic full scaled dot-product attention;
5. BF16 output projection and gated residual slot 2;
6. RMS AdaLN slots 3/4;
7. BF16 FC1, SwiGLU, FC2, and gated residual slot 5.

Model-private utility kernels also cover BF16/F32 casts, add/subtract, SiLU,
layer norm, F32 patch projection to BF16, and BF16 final projection to F32.
The patch/output helpers retain straightforward fixed-order accumulation for
the initial quality route; later pipeline work may replace them only against
the same retained boundaries.

The independent oracle is generated outside production:

```sh
nix develop -c python3 tools/gufo/h3_dit_golden.py \
  --model-root /var/llms/huggingface/MiniMax-H3 \
  --output /var/llms/huggingface/gufo-h3-oracles/dit-block0-528-v2
```

Run the analytic and external-model gates:

```sh
./build-h3-173/minimax_h3_dit_hip_test --analytic

GUFO_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
GUFO_H3_DIT_GOLDEN=/var/llms/huggingface/gufo-h3-oracles/dit-block0-528-v2 \
  ./build/hardware-test/minimax_h3_dit_hip_test --real
```

Sparse attention, token reduction, cross-block fusion, reuse, layer thinning,
and quantized DiT weights are not part of this baseline.

## Complete BF16 Transformer and Denoiser

`DenoiserSession` is the quality-first full FL2VA transformer route. Creation
consumes the six-row layer-50 prompt tensor, refines it through
`condition_proj` and both token-refiner blocks, precomputes the independent
video/audio timestep schedule, and then admits all 50 dense core blocks.

The memory lifecycle is explicit:

1. prompt-encoder ownership ends before denoiser creation;
2. timestep embedding weights are loaded, executed, and released;
3. each block's 96,768-wide AdaLN projection is loaded, executed for all
   unique timestep rows, copied into that block's persistent constants, and
   released before the next projection;
4. the corresponding core block and its pre-sized activation arena become
   resident;
5. the final AdaLN projection is released before patch and output heads are
   admitted.

Every core block uses the byte-validated BF16 projection policy from the
one-block baseline. The current quality route keeps a separate fixed arena per
block so setup can fail atomically and timed execution performs no device
allocation. Between block streams, BF16 residual rows cross a preallocated
host staging pair. This is synchronization and byte transport on the unified
memory machine, not CPU tensor compute; all projection, normalization,
attention, MLP, head, and Euler arithmetic remains on gfx1151.

A fresh forward:

- projects packed F32 audio/video latents to BF16 and packs refined text,
  audio, then video rows;
- executes all 50 blocks with full attention and the released modality row
  maps;
- applies final audio/video AdaLN and F32 output heads;
- retains F32 velocity boundaries for diagnostics, reuse, extrapolation, and
  the device Euler update.

The quality baseline uses all 50 blocks and one fresh evaluation per schedule
step. When fewer blocks are requested, the runtime scores AdaLN gate slots 2
and 5 over every timestep row and modality, protects blocks 0, 1, and 49, and
skips the lowest-scoring remaining blocks exactly as h3.c does. Selected
blocks still execute in original residual-stream order. The serving API also
accepts a whole-denoiser reuse interval. Skipped evaluations reuse the latest
two F32 velocity boundaries with the frozen bounded extrapolation plan; the
first and last schedule steps are always evaluated. Sparse attention, token
reduction, core-residual reuse, and quantization remain disabled.

Preset authority is explicit: `exact` is the released Diffusers 50-point,
49-evaluation BF16 route. `fast` and `aggressive` retain Diffusers' 20-point,
19-evaluation grid and borrow only h3.c's gate ranking and velocity-reuse
ideas. They are native serving modes, not exact reproductions of h3.c's
20-transition schedules.

Generate the independent single-forward 256x256 teacher outside the repository:

```sh
nix develop -c python3 tools/gufo/h3_denoiser_golden.py \
  --model-root /var/llms/huggingface/MiniMax-H3 \
  --conditioning /var/llms/huggingface/gufo-h3-oracles/\
fox-layer50-transformers.bf16 \
  --output /var/llms/huggingface/gufo-h3-oracles/\
denoiser-256x256x22-forward-v2 \
  --noise-mode seeded --forward-only
```

Run one complete 256x256 transformer forward, without advancing the sampler or
decoding either VAE:

```sh
GUFO_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
GUFO_H3_DENOISER_GOLDEN=/var/llms/huggingface/gufo-h3-oracles/\
denoiser-256x256x22-forward-v2 \
  ./build/hardware-test/minimax_h3_denoiser_hip_test --oracle-forward
```

Production-shape development coverage uses the real 512x512, 22-frame,
1,872-row layout but executes only the first complete 50-block forward. This
catches shape-dependent attention, patch-layout, and output-head errors without
advancing the sampler, invoking a VAE, or producing a video:

```sh
nix develop -c python3 tools/gufo/h3_denoiser_golden.py \
  --model-root /var/llms/huggingface/MiniMax-H3 \
  --conditioning /var/llms/huggingface/gufo-h3-oracles/\
fox-layer50-transformers.bf16 \
  --output /var/llms/huggingface/gufo-h3-oracles/\
denoiser-512x512x22-forward-v2 \
  --width 512 --height 512 --steps 2 --noise-mode seeded --forward-only

GUFO_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
GUFO_H3_DENOISER_GOLDEN_512=/var/llms/huggingface/gufo-h3-oracles/\
denoiser-512x512x22-forward-v2 \
  ./build/hardware-test/minimax_h3_denoiser_hip_test --oracle-512-forward
```

There is deliberately no full-schedule development test. The `exact` preset remains
available to end users and for rare, explicitly requested release validation,
but ordinary implementation, profiling, and CI use operator or single-block
oracles. A complete one-forward oracle is reserved for changes that cross block
boundaries. One deliberately short selected-frame decode may be used once as a
visual smoke test after such a change passes parity; it is not a routine
development gate.

## VisualVAE and AudioVAE Output Phases

The two released FL2VA output decoders are independent model-private phases.
They consume only the final normalized latents, never overlap their complete
weight inventories, and expose cancellation and memory telemetry separately
from the denoiser.

`VideoVaeDecoder` keeps the released F32 transformer arithmetic on gfx1151.
It uses the released fixed 256-pixel spatial tiles with at least 64 pixels of
overlap, decodes legal seven-latent temporal chunks into 22-frame windows, and
blends the released five-latent/17-frame stride. The selected-frame API still
runs every complete model chunk needed by the requested global frame indexes;
it omits unrelated chunks and RGB reconstruction/readback only. Spatial and
temporal stitching are output composition, not CPU model inference.

Generate and run the operator-owned VisualVAE oracle:

```sh
nix develop -c python3 tools/gufo/h3_video_vae_golden.py \
  --model-root /var/llms/huggingface/MiniMax-H3 \
  --latent /var/llms/huggingface/gufo-h3-oracles/\
denoiser-256x256x22-4step-v1/video_final.f32 \
  --output /var/llms/huggingface/gufo-h3-oracles/\
video-vae-256x256x22-v1

GUFO_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
GUFO_H3_VIDEO_VAE_GOLDEN=/var/llms/huggingface/gufo-h3-oracles/\
video-vae-256x256x22-v1 \
  ./build/hardware-test/minimax_h3_video_vae_hip_test
```

The native development gate invokes `DecodeSelected` once for frames 0, 5,
11, 16, and 21. It does not run an additional complete-frame decode or a
repeatability pass. The selected path still executes the complete released
model chunk required by those frames, so it validates the same decoder
arithmetic and fixed 256-pixel tile context.

`AudioVaeDecoder` accepts channel-major normalized F32 `[32,2,T]` audio
latents and returns channel-major F32 `[2,T*800]` PCM at 32 kHz. The native
route covers the released input projection, exact weight normalization,
Conv1d, ConvTranspose1d, seven BigVGAN upsampling stages, 21 residual blocks,
127 alias-free SnakeBeta activations, final clipping, and channel
recombination. Left and right channels remain independent batches throughout.
The unused audio encoder and its attention projection are not loaded into the
text-to-video serving path.

All 127 released alias-free upsample/downsample filters were checked
byte-for-byte and are identical, so the runtime shares one resident pair
without changing arithmetic. Model convolution and activation compute stays
on gfx1151; host work is limited to checkpoint I/O, progress, final PCM
readback, evaluation, and media output.

Generate and run the operator-owned AudioVAE oracle:

```sh
nix develop -c python3 tools/gufo/h3_audio_vae_golden.py \
  --model-root /var/llms/huggingface/MiniMax-H3 \
  --latent /var/llms/huggingface/gufo-h3-oracles/\
denoiser-256x256x22-4step-v1/audio_final.f32 \
  --output /var/llms/huggingface/gufo-h3-oracles/\
audio-vae-256x256x22-v1

GUFO_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
GUFO_H3_AUDIO_VAE_GOLDEN=/var/llms/huggingface/gufo-h3-oracles/\
audio-vae-256x256x22-v1 \
  ./build/hardware-test/minimax_h3_audio_vae_hip_test --real
```

The real AudioVAE gate performs one decode and compares both the waveform and
deterministic spectrogram. It does not repeat the waveform for statistical or
byte-repeat evidence.

The MP4 writer sends RGB24 and channel-major PCM to FFmpeg through two
concurrent nonblocking pipes. Audio interleaving uses a bounded 4,096-sample
chunk rather than a second full waveform. The child, both descriptors, and
partial output are reclaimed on success, write failure, child failure, or
cancellation. The pinned Nix package uses headless FFmpeg with H.264 and AAC;
FFprobe validates codec, dimensions, 24 fps, stereo 32 kHz audio, duration,
and start-time synchronization. When internal and output canvases differ,
FFmpeg performs an explicit Lanczos scale before H.264 encoding. Media tools
are runtime infrastructure only, not a CPU inference fallback.
F32 RGB is quantized with explicit round-to-nearest, ties-to-even semantics,
matching Diffusers' NumPy conversion and h3.c's `lrintf` behavior.

## Text-to-Video CLI and Frozen Presets

`gufo video` is the first complete text-only serving path. It requires
an explicit operator-supplied model directory, validates the pinned manifest,
tokenizes and encodes layer 50, denoises on gfx1151, decodes the released
VisualVAE and AudioVAE, and atomically publishes either an audiovisual MP4 or
selected diagnostic PPM frames. It never downloads weights and has no CPU
model-inference fallback.

The named preset contract is versioned as
`gufo.minimax-h3-text-generation.v1`:

| Preset | Internal canvas | Output canvas | Sigma points / evaluations | Active blocks | Denoiser reuse |
| --- | ---: | ---: | ---: | ---: | ---: |
| `exact` | 512x512 | 512x512 | 50 / 49 | 50 | 1 |
| `exact-1344x768` | 1344x768 | 1344x768 | 50 / 49 | 50 | 1 |
| `fast` | 384x384 | 512x512 | 20 / 19 | 45 | 2 |
| `aggressive` | 320x320 | 512x512 | 20 / 19 | 40 | 3 |
| `dev` | 256x256 | selected 256x256 frames | 5 / 4 | 50 | 1 |

Token reduction is false in every preset. The development preset decodes only
frames 0, 11, and 21 and skips AudioVAE and muxing, so numerical work can
iterate without waiting for a complete video.
The 45- and 40-block presets use schedule-dependent AdaLN gate ranking, not a
prefix of the first 45 or 40 transformer blocks.

Rapid development example:

```sh
nix develop -c ./build-h3-173/gufo video \
  --model /var/llms/huggingface/MiniMax-H3 \
  --preset dev \
  --frames-dir /tmp/h3-fox-dev \
  --profile \
  "A red fox walking through snow"
```

Manual end-user or release exact output:

```sh
nix develop -c ./build-h3-173/gufo video \
  --model /var/llms/huggingface/MiniMax-H3 \
  --preset exact \
  --seed 42 \
  --output /tmp/h3-fox-exact.mp4 \
  --latents-dir /tmp/h3-fox-exact-latents \
  --profile \
  "A red fox walking through snow"
```

Released five-second full-resolution output uses 124 aligned frames at 24 fps:

```sh
nix develop -c ./build/hardware-test/gufo video \
  --model /var/llms/huggingface/MiniMax-H3 \
  --preset exact-1344x768 \
  --seed 42 \
  --output /tmp/h3-fullres.mp4 \
  --profile \
  "An anime soccer goalkeeper summons a giant cyan energy hand to catch an incoming shot."
```

Every admitted request writes a parameter document containing the preset,
prompt hash, seed, geometry, step/block/reuse values, and selected frames.
Successful runs add a timing, memory, denoiser-evaluation-count, and swap
report. Neither document contains the prompt text or model path. `SIGINT`
propagates through the phase cancellation token; temporary frames and media
are never published as final output.

Advanced geometry, step, block, reuse, output-canvas, and selected-frame
overrides exist for controlled experiments and are fully represented in the
parameter document. `--latents-dir` atomically retains final F32 video/audio
latents plus their shapes and hashes for an explicit teacher comparison. Its
manifest also binds the model/reference revisions, seed, prompt hash, exact
layer-50 conditioning hash, geometry, evaluation/block/reuse policy, and
attention kernel. The comparison rejects provenance drift before applying a
numeric tolerance. This diagnostic output is never enabled by the API.
`--attention-kernel scalar` retains the original deterministic SDPA as an
explicit profiling/rollback route; `row-parallel` is the named-preset default,
and the selected policy is included in parameters and telemetry. First-frame,
last-frame, and ordered-reference inputs are rejected until their future
conditioning milestones land. No implicit conditioning/model cache is enabled
in this CLI.

Sparse attention is explicitly future work. It must not change these preset
names or silently enter the exact route.

## Asynchronous Video API

`gufo serve --video-model <DIR>` enables the versioned
`gufo.video-api.v1` text-to-video API. When `--model` is not also supplied,
the process is video-only and does not load an unrelated text model:

```sh
nix develop -c ./build-h3-173/gufo serve \
  --host 127.0.0.1 \
  --port 8080 \
  --video-model /var/llms/huggingface/MiniMax-H3 \
  --video-root /var/llms/huggingface/gufo-h3-jobs \
  --video-ttl 3600
```

Startup inspects the operator-supplied directory against the pinned source
manifest before binding the listening socket. A missing, partial, or
incompatible checkpoint therefore fails configuration rather than creating a
job that is known to be unusable. There is no CPU inference fallback.

The supported routes are:

| Method | Path | Result |
| --- | --- | --- |
| `POST` | `/v1/videos` | Admit one asynchronous text-to-video job |
| `GET` | `/v1/videos/{id}` | Read stable status, progress, and error fields |
| `GET` | `/v1/videos/{id}/content` | Read completed MP4 or diagnostic PPM content |
| `DELETE` | `/v1/videos/{id}` | Cancel or remove a job and reclaim its artifacts |

`GET /v1/models` advertises `minimax-h3` and the
`minimax-h3-{exact,fast,aggressive,dev,fullres}` aliases. An alias selects its
matching frozen preset; if `gufo.preset` is also present, the two must agree.

Create accepts `application/json` and the `multipart/form-data` shape used by
OpenAI video clients. JSON remains convenient for the nested Gufo extension:

```json
{
  "model": "minimax-h3-fast",
  "prompt": "A red fox walking through snow",
  "size": "512x512",
  "seconds": "1",
  "gufo": {
    "seed": 42,
    "output_format": "mp4"
  }
}
```

The released full-resolution JSON request is:

```json
{
  "model": "minimax-h3",
  "prompt": "An anime soccer goalkeeper summons a giant cyan energy hand.",
  "size": "1344x768",
  "seconds": 5,
  "gufo": {
    "seed": 42,
    "output_format": "mp4"
  }
}
```

The equivalent standard text-only form selects the mode through the model
alias:

```sh
curl http://127.0.0.1:8080/v1/videos \
  -F model=minimax-h3-fast \
  -F 'prompt=A red fox walking through snow' \
  -F size=512x512 \
  -F seconds=1
```

Multipart duplicate fields and malformed boundaries fail closed. A multipart
`input_reference` file is recognized but rejected explicitly until the
first/last-frame conditioning milestone lands.

Exact, fast, and aggressive support the rapid one-second, 22-frame, 512x512
MP4 route. `exact-1344x768` additionally supports the released five-second,
124-frame, 1344x768 audiovisual MP4 route. The development alias requires
`size: "256x256"` and `output_format: "ppm"`; it accepts a released
VisualVAE-decodable frame count through `gufo.frames` (`22, 39, ... 345`) and
exactly one diagnostic frame through `gufo.selected_frame`. Although the
denoiser geometry can represent five frames, the released VisualVAE's minimum
valid temporal chunk is seven latents and 22 output frames. The development
path therefore preserves the real 22-frame latent window while decoding and
reading back only the requested preview frame. First-frame, last-frame,
image-reference, and ordered-reference fields fail closed until their
conditioning milestones land.

The 22-frame route is a Gufo development/serving extension for rapid
iteration. The released Diffusers production pipeline admits aligned
durations from 5 through 15 seconds; matching that production-duration
envelope is tracked separately and does not remove the short native diagnostic
route.

One worker owns the H3 generation path and one additional request may wait in
the bounded queue. Further admissions return HTTP 429 with `Retry-After`.
Deleting the active job propagates cancellation through prompt encoding,
denoising, both VAEs, and media publication. Completed content supports a
single standard HTTP byte range and reports `Accept-Ranges: bytes`.

Job status is persisted atomically. Completed artifacts survive process
restart until their configured TTL; queued, interrupted, expired, deleted,
and partial artifacts are reclaimed. Status documents never store the prompt
or private filesystem paths. Parameter reports store only a SHA-256 prompt
digest plus the reproducibility controls, and telemetry contains bounded
numeric execution data.

## Full-Route Profiling

`tools/gufo/h3_profile.py` drives the same public CLI route used by serving.
It is a manual release tool, not a development or CI gate. Development
optimization uses one independently frozen 528-row DiT block oracle for
correctness and one 1,872-row DiT block for performance and profiling.
Decoder-local work uses the independently frozen 256x256 selected-frame
VisualVAE oracle. Neither iteration gate runs all 50 blocks or advances a
denoising schedule. A complete exact generation is run only when explicitly
requested after those gates pass.

The issue #175 performance workload is a manual single-block profile, not a
CTest target:

```sh
GUFO_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
  ./build/hardware-test/minimax_h3_dit_hip_test --profile-1872
```

It executes block 0 once with deterministic production-shape inputs. The
retained progression measured:

| Attention implementation | Block GPU time |
| --- | ---: |
| custom row-parallel baseline | 38.518 s |
| batched rocBLAS QK/PV + parallel BF16 softmax | 1.076 s |
| CK BF16 WMMA fused attention + optimized elementwise path | 88.279 ms |

The final implementation is `436.32x` faster than the original block and
`12.19x` faster than the first batched path, a 99.77% and 91.80% latency
reduction respectively. The supported production shape fuses QK, scaled
softmax, and PV without materializing the former 1.097 GiB F32-score and
BF16-probability matrices. A checked fallback allocates those matrices only
for unsupported shapes. SwiGLU and residual gates process eight BF16 values
per thread.

The final trace's eleven timed dispatches sum to 88.225 ms. Dense projections
now dominate: the two MLP GEMMs take 22.594 and 21.881 ms, QKV projection
takes 17.022 ms, fused attention takes 16.428 ms, and output projection takes
7.527 ms. The independent 528-row block oracle remains the quality gate and
stays below its frozen 1% relative-error ceilings. Two alternate QKV layouts
were measured and removed because they preserved quality but increased
production-shape latency.

The released 1344x768, 124-frame layout contains 37,716 rows. The former
scalar fallback took 12,916.9 seconds for one block. CK BF16 WMMA reduced that
to 36.13 seconds, and AOTriton with row-major interleaved heads reduced it to
15.29 seconds. Writing normalized Q/K/V directly in contiguous head-major
layout lets the native gfx1151 AOTriton FlashAttention image avoid a strided
sequence walk. Pinned byte-equivalent rocBLAS projections, lifetime-aliased
scratch, direct row-major attention output, and a fused gate/RMS handoff reduce
the retained block to 3.29 seconds and 5,051 MiB of shared scratch while
preserving output SHA-256
`b448469ba8c57601e9f8bc3ca599e427528086a7149fc6946c3660f5d212da10`.
A single 50-block forward takes 151.658 seconds, down from 157.155 seconds,
with 43.1 GiB peak residency and zero swap. At 7,136 rows it remains within the
frozen scalar parity budget (`relative_l2=0.003853`,
`normalized_max=0.010417`) with no non-finite values. Sequences at or below
4,096 rows continue to use the unchanged Issue #184 CK kernel.

The associated 92.494 ms wall-time capture peaked at 44 W and 1,192 MHz. Power
is diagnostic here: this isolated block finishes before the APU can ramp
toward its configured 120--140 W envelope. The benchmark is not extended or
repeated merely to produce a larger wattage reading.

### Full-resolution boundary projection pass

A focused kernel trace of one released 1344x768, 124-frame, 37,716-row,
50-block forward attributed the original 154.171-second forward as follows:

| Work | GPU time | Forward share |
| --- | ---: | ---: |
| AOTriton full attention | 65.100 s | 42.2% |
| dense rocBLAS block GEMMs | 47.403 s | 30.7% |
| SwiGLU | 23.161 s | 15.0% |
| scalar patch and final projections | 16.651 s | 10.8% |
| remaining elementwise and transfer work | 1.856 s | 1.2% |

The patch and final heads were still one-thread-per-output scalar F32 dot
products. The retained path uses an atomics-disabled gfx1151 rocBLAS F32 GEMM,
keeps F32 inputs, weights, accumulation, and final-head outputs, preserves the
released BF16 boundary after patch projection, and applies the same F32 biases
after GEMM. One reusable F32 scratch allocation serves video and audio
sequentially.

The focused full-resolution forward falls from 154.171 to 133.263 seconds, a
13.56% reduction. Across the released 49-evaluation exact schedule this
removes about 1,024 seconds (17.1 minutes) of denoising wall time. Peak
residency rises from 42.77 to 43.52 GiB, with zero swap. The frozen 1,872-row
oracle remains green: video velocity relative L2/max are
`0.0108984`/`0.0133581`, and audio velocity relative L2/max are
`0.00867719`/`0.0188007`. The independent 528-row oracle also remains green:
video velocity relative L2/max are `0.0193639`/`0.0458577`, and audio velocity
relative L2/max are `0.0112591`/`0.0216204`.

The post-fix ranking leaves full attention as the primary full-resolution
target, followed by dense block GEMMs and SwiGLU. Decoder work is paid once
per video, while this denoiser cost is paid for every schedule evaluation, so
quality-neutral attention work has the largest remaining end-to-end leverage.

### Full-resolution exact SwiGLU lookup pass

The next profile showed that the vectorized full-resolution SwiGLU kernel still
spent 350.440 ms per block evaluating the same activation for BF16 gate
values. A BF16 gate has exactly 65,536 possible bit patterns. The retained
path builds a 256 KiB F32 SiLU lookup once on the GPU with the existing
`gate / (1 + __expf(-gate))` expression, then performs the unchanged F32
multiply and BF16 rounding. An exhaustive GPU test covers all 65,536 gate bit
patterns and requires byte-identical output from the original and lookup
kernels.

The lookup takes 0.012 ms to build and reduces full-resolution SwiGLU to
15.776 ms per block. The released 37,716-row block falls from 3.279 to
2.846 seconds while preserving output SHA-256
`b448469ba8c57601e9f8bc3ca599e427528086a7149fc6946c3660f5d212da10`.
One 50-block forward falls from 133.263 to 116.199 seconds, a 12.80%
reduction. Across 49 exact evaluations this removes about 836 seconds
(13.9 minutes) of denoising wall time. The lookup is enabled only at 32,768
rows or above, so the 7,136-row path and Issue #184 kernels are unchanged.

### Full-resolution attention admission pass

The retained gfx1151 Triton attention kernel was admitted only at exactly
37,716 rows. That row count is 37,710 non-text rows plus the text rows, so it
required a prompt of exactly six tokens. The released example prompt tokenizes
to 16, which produces 37,726 rows, and `LaunchH3TritonAttention` returned false
and fell through to AOTriton without reporting anything. A kernel trace of the
released 1344x768, 124-frame layout with that prompt showed `attn_fwd`, the
AOTriton image, and no `h3_attention_forward` at all: the specialized kernel
was unreachable for every realistic prompt.

The kernel never required a fixed length. It masks its query rows, peels a
separately instantiated masked key tail, and derives every offset from the
runtime `sequence` argument. Admission is now bounded by the two properties
that do constrain it: the materialized-attention threshold below which the CK
fused kernel is selected, and the signed 32-bit pointer range the kernel is
compiled with, which bounds `heads * sequence * head_dim` and is what keeps it
spill-free.

Measured on the released layout at 37,726 rows with three active blocks, two
evaluations, one binary and an execution toggle:

| Attention image | Forward (3 blocks) | Per block |
| --- | ---: | ---: |
| AOTriton (previous behaviour) | 6752.816 ms | 2250.9 ms |
| Retained Triton kernel | 6533.234 ms | 2177.7 ms |

That is 3.25% of the full-resolution forward, about 3.7 seconds per evaluation
and 179 seconds across 49 exact evaluations. A standalone check confirms the
kernel is equally accurate and equally fast away from the pinned length:
relative L2 against an F32 reference is `2.32e-3` at both 37,716 and 37,726
rows, and throughput stays within 31.8-32.4 TFLOPS for 8,192 to 41,806 rows,
including every non-multiple of `BLOCK_M`. No repository gate exercises
full-resolution attention, so that standalone comparison is the evidence; the
528-row block boundary is byte-identical to the previous implementation and the
1,872-row production forward oracle remains inside every frozen ceiling.

### Rejected full-resolution candidates

Measured on the released 37,716-row shape and retained here so they are not
retried:

- **Attention tile and warp retuning.** All 72 combinations of
  `BLOCK_M` in {32, 64, 128}, `BLOCK_N` in {16, 32, 64, 128}, 2/4/8 warps and
  1/2 stages were compiled and timed. The retained `64/32/4/1` configuration is
  the fastest of all of them at 31.7 TFLOPS; every larger tile either spills or
  is slower. Attention is at a genuine local optimum for this kernel shape.
- **hipBLASLt for the four block projections.** Slower than the pinned rocBLAS
  solutions on every shape: 26.7 against 42.7 TFLOPS on QKV, 21.0 against 37.0
  on the output projection, 26.8 against 42.7 on FC1, 14.6 against 17.9 on FC2.
- **Pre-transposed projection weights** so the GEMM reads `op(A) = none`.
  Slower on all four shapes, by up to 1.7x on the output projection.

A second pass, run without the peak-memory constraint, closed off the remaining
projection and attention avenues. All measured at 37,716 rows.

**Attention is bounded by the register file, not by tuning.** Beyond the 72
tile/warp/stage combinations, all 128 combinations of `kpack`,
`matrix_instr_nonkdim` and `waves_per_eu` were compiled and timed at the
retained tile and its three nearest neighbours. The spread is 0.1%: the
retained configuration is 1281.8 ms against 1283.1 ms for the best alternative.
The reason `BLOCK_M` cannot grow is arithmetic: a `BLOCK_M` x `HEAD_DIM` F32
accumulator at 64 x 128 is 8,192 floats, which is 256 VGPRs per lane across a
wave32 and already the whole register file. Every `BLOCK_M = 128` variant
spills, which is why the query dimension cannot be widened to amortize the
K/V re-reads. 31.8 TFLOPS is the ceiling for a head-dimension-128 FlashAttention
on this part, not a tuning gap.

**FC2 resisted seven approaches.** At 5,376 x 14,336 it sustains about 17
TFLOPS where its siblings reach 36-43, and the output projection at the same `M`
with half the `K` reaches 37:

| Candidate | Result |
| --- | --- |
| N-split into 2/4/8/16 column chunks | Byte-identical and no faster (1.00x to 0.93x) |
| K-split, BF16 accumulation | 1.45x, but rounds the partial sums to BF16 |
| K-split, F32 accumulator via rocBLAS | 0.16x; there is no good BF16-in/F32-out kernel |
| K-split, F32 accumulator via hipBLASLt | About 1.14x after the cast, not worth 811 MiB |
| Transposed orientation, computing C^T | 1.00x, so operand orientation is not the cause |
| hipBLASLt directly, 32 heuristic algorithms | 0.84x |
| A purpose-built BF16 WMMA kernel | 0.75x at best across seven tilings |

The N-split result is the informative one: chunking the token dimension is
numerically free and provably byte-identical, and it changes nothing. So the
shape is not suffering from working-set effects that blocking can fix. The
custom WMMA kernel reached 12.9 TFLOPS against rocBLAS's 17.3 even before its
fragment layout was correct, and rocBLAS reaches 42.7 on the sibling shapes, so
closing that gap is specialist GEMM work rather than a tuning exercise. FC2
remains the largest single opportunity in the forward at about 8%, and it needs
either a competitive hand-written GEMM or acceptance of BF16 partial sums.

### Issue #184 second focused pass

The next retained pass profiles complete phase boundaries once rather than
extrapolating from a full generation:

| Workload | Fresh baseline | Retained | Reduction |
| --- | ---: | ---: | ---: |
| 37-frame stereo AudioVAE decode | 59.4383 s | 2.42326 s | 95.92% |
| five-selected-frame VisualVAE tile | 8.60171 s | 6.12212 s | 28.83% |
| one 1,872-row, 50-block denoiser forward | 9.02751 s | 5.49537 s | 39.13% |

AudioVAE replaces all 129 Conv1d and seven ConvTranspose1d scalar loops with a
bounded reusable F32 im2col workspace and rocBLAS GEMM. Transposed weights are
reordered once after released weight normalization, and alias-free SnakeBeta
still executes the released upsample/filter/sine/downsample sequence. The
waveform and deterministic spectrogram oracles remain green.

VisualVAE computes each Q/K norm once per head, broadcasts the factor, folds
biases only across quality-equivalent F32 rounding boundaries, and vectorizes
SwiGLU. Stable softmax retains its existing F32 reduction order. Peak memory
remains 9.766 GiB and the selected-frame relative L2 is `8.14968e-7`.

A subsequent exact-only decoder pass groups eight independent softmax and RMS
rows per workgroup and uses aligned `float4` residual updates. The selected
tile falls from 6.772 seconds to 5.735 seconds uninstrumented; the retained
trace measures 5.786 seconds. RMS normalization falls from 365 to 138 ms and
residual scale/add from 360 to 115 ms. The frozen selected-frame oracle metrics
remain unchanged. A wave-grouped QKV/RoPE candidate was slower and changed the
oracle slightly, so it was removed.

The denoiser keeps the hidden state in two BF16 device buffers across all 50
blocks instead of copying it through host memory and synchronizing per block.
Its grouped QKV and AdaLN operations are split into F32 factor-reduction and
packed BF16 application kernels. This preserves the previous reduction order
while eliminating 1,240-byte/thread QKV spills and 436-byte/thread AdaLN
spills. The DiT translation unit is compiled at `-O2`: `-O3` retained those
spills, while an unqualified hardware-test compile was not optimized.

The production single-forward oracle remains below every frozen ceiling, peak
memory is 58.880 GiB, and swap remains zero. The final profile shifts the next
denoiser opportunities to SwiGLU, fused attention, and dense GEMMs. They are
future optimization work rather than justification for changing precision or
quality in this pass.

The explicitly requested post-retention aggressive smoke changed the phase
ranking: VisualVAE consumed `188.650` of `229.382` seconds (`82.24%`), while
the optimized denoiser consumed `12.158` seconds (`5.30%`). A single focused
selected-frame profile then attributed `42.815` of `46.685` GPU seconds to the
scalar VisualVAE full-attention kernel.

The retained decoder path replaces that kernel with strided-batched F32
rocBLAS QK/PV GEMMs and an in-place stable F32 softmax. Exact-shape gfx1151
solution indices then reduce the QKV, output, and two MLP projections without
changing their F32 inputs, outputs, or accumulation type. On the same frozen
256x256 tile, profiled wall time fell from `46.6852` seconds to `4.79063`
seconds after attention replacement and finally to `2.82558` seconds. The
final route is `16.522x` faster than the scalar baseline and `1.695x` faster
than the first batched route.

Selected-frame relative L2 remains `8.15249e-7`, the phase peak rises from
9.38 GiB to 9.77 GiB for the 394.2 MiB F32 attention matrix, and sampled
package power reaches `134 W`. The final profile is 88.73% rocBLAS matrix
work. A cached softmax and faster attention-GEMM solution indices were
discarded after their changed reduction order failed the frozen quality gate.
No second MP4 or complete denoising trajectory was run for retention.

### September 2026 exact-route audit

The VisualVAE LayerNorm now assigns eight independent rows to the eight waves
of a 256-thread block and uses the same 32-lane reduction tree through wave
shuffles. Across ten independent exact 1,797-by-2,048 microbenchmark runs, each
containing fifteen 1,000-iteration paired samples, it wins nine times. The
aggregate median falls from `0.12345` ms to `0.11371` ms (`1.086x`) with
bit-exact output. VGPR use moves from 13 to 18, occupancy stays at 16 waves per
SIMD, neither route spills, and the retained route removes the former 1,024-byte
LDS allocation. The real selected-tile oracle remains at relative L2
`8.14968e-7`.

The remaining candidates were discarded:

- Removing the VisualVAE per-block stream synchronization measured
  `2743.60` ms versus `2746.32` ms with the same LayerNorm route, only `0.10%`
  amid tens of milliseconds of run variation.
- Replacing AudioVAE stage synchronizations with events measured `76.45` ms
  versus `76.48` ms (`0.04%`) while preserving spectrogram relative L2
  `3.62714e-6` and waveform relative L2 `1.08936e-5`.
- A direct BF16-input/F32-accumulate final-head GEMM had no supported rocBLAS
  solution for the released shape and failed before producing velocity output.
  The cast-plus-GEMM route remains unchanged.
- Graph capture was not retained: a generation owns a fresh denoiser session,
  and host-built row maps and schedules remain step-dependent. Existing traces
  cap dispatch-only savings near `0.1%`, below the cost and lifecycle
  complexity of a capture cache.

Both synchronization experiments and the mixed-GEMM implementation were
removed rather than left behind switches. Validation used the frozen
single-forward and selected-decoder oracles; no full video generation was run.

The harness schedules each requested preset once, rejects `--rounds` values
other than one, and requires an acknowledgement before it can launch a
complete generation:

```sh
nix develop -c python3 tools/gufo/h3_profile.py \
  --binary ./build-h3-173/gufo \
  --model /var/llms/huggingface/MiniMax-H3 \
  --output-root /var/llms/huggingface/gufo-h3-profiles/bf16-v1 \
  --presets exact \
  --rounds 1 \
  --cooldown-seconds 0 \
  --allow-full-generation
```

If a rare release comparison is requested, run the same command once for the
candidate with its distinct binary and output root. Both observations use the
same prompt, seed, model/reference revisions, and harness. No warm, repeated,
or eviction-requested full video is added.

The generated report combines the versioned parameter and runtime telemetry
with wall time, process read/write throughput, peak total/anonymous/file-backed/
shared-memory RSS, HWM/swap, and available AMDGPU power and selected-clock
samples. Raw profiler captures may retain an available temperature sensor, but
the comparison tool deliberately ignores it because ambient conditions differ
between operators. It records the binary hash, profiling-harness hash,
and kernel release while omitting prompt text, model paths, hostname, and the
private command line. Successful comparison runs require complete lowercase
SHA-256 identities for the binary, harness, prompt, and delivered output;
malformed or abbreviated identity strings fail closed.
The retained pre-selector scalar binary is identified by its pinned SHA-256.
Its older parameter document may omit `attention_kernel`; report and delivery
quality tools infer `scalar` only when the sibling profile binds that exact
binary hash, parameter document, and delivered output hash. Unknown binaries
with missing kernel provenance fail closed.
Timestamped raw monitor samples are retained alongside the aggregate so later
analysis can distinguish sustained behavior from a transient peak.
Every admitted release-comparison run must contain the complete hard-gate
telemetry set. The candidate's direct complete-generation latency, memory
peaks, loaded-clock coverage, and swap are compared with the single scalar
observation; no median or other repeated-run statistic is claimed.
Power and integrated energy are diagnostics rather than efficiency ceilings.
An optimized route may use the configured 120--140 W graphics envelope when
that power performs useful work and reduces complete-generation latency.
Sustained-clock throttling, swap, output quality, and evidence of busy-waiting,
redundant transfer, or recomputation remain promotion gates. Temperature is
neither required telemetry nor a comparison objective.

`tools/gufo/h3_cache_control.py` remains available for a separately requested
cache diagnostic:

```sh
nix develop -c python3 tools/gufo/h3_cache_control.py \
  --model /var/llms/huggingface/MiniMax-H3 \
  --output /var/llms/huggingface/gufo-h3-profiles/cache-control.json
```

The cache-control artifact records files/bytes attempted, errors, memory
snapshots, and elapsed time without private filenames. Each regular target is
opened with `O_NOFOLLOW` and sized through the opened descriptor before the
advisory request. Linux `POSIX_FADV_DONTNEED` is advisory, so the result is
named `eviction-requested`, not asserted to be a physically cold cache. This
utility does not trigger a generation.

For a separately requested fast/aggressive delivery-quality study, compare a
delivered preset MP4 with its matching exact output:

```sh
nix develop -c python3 tools/gufo/h3_preset_quality.py \
  --reference /var/llms/huggingface/gufo-h3-profiles/bf16-v2/000-exact/output.mp4 \
  --candidate /var/llms/huggingface/gufo-h3-profiles/bf16-v2/001-fast/output.mp4 \
  --candidate-label fast \
  --output /var/llms/huggingface/gufo-h3-profiles/bf16-v2/001-fast/quality.json
```

The versioned report hashes both MP4s and compares all decoded RGB frames,
first/middle/last frames, adjacent-frame deltas, stereo waveform, deterministic
spectrogram, frame/audio durations, and A/V duration delta. It contains no
prompt, model path, hostname, or private command line. Promotion reports
require both profile-style `parameters.json` siblings, embed and hash those
privacy-safe documents, and reject mismatched provenance or preset contracts.
Missing parameter documents are accepted only with the explicitly diagnostic
`--allow-missing-parameters` switch. These delivery-level metrics supplement
rather than replace the retained component oracles.

The runtime telemetry separates model inspection, tokenizer work, prompt
weight loading/prefetch wait/submission/GPU time, AdaLN precompute, core load,
every fresh denoiser forward, sampler time, VisualVAE tiles, AudioVAE stages,
media composition, VisualVAE frame-set availability (`first_preview_ms`),
dispatch counts, page faults, swap, and phase memory peaks. That field is
recorded when the requested selected/full frame set returns; it is not a
streaming-first-frame claim. The single-observation comparison also derives
denoiser non-forward time as denoiser wall time minus the sum of fresh
forwards, so old and new binaries remain comparable even when a raw sampler
field changes granularity. Both issue #175 generations use the default
`first-observation` label rather than making an unsupported cache-state claim.

During development, a kernel or residency candidate is retained only when it
passes the independent single-block oracle, the frozen preset contracts, and
a single production-shape block profile showing useful work. A short
sampler analytic gate is used for integration changes that cross the sampler
or schedule boundary; a block-stack change uses one complete forward and does
not advance the denoising schedule. Neither gate is rerun for an
attention-kernel iteration. Fast and aggressive do not add full benchmark
videos.
Byte-identical complete MP4 comparison is reserved for an explicitly requested
release validation, not every implementation iteration.

## Operator Attestation

The operator of the dedicated Strix Halo development machine has stated that
the required MiniMax H3 terms or authorization have been obtained and accepted
for this work. The checkpoint is retained only on that operator-controlled
machine.

gufo does not attempt to adjudicate the operator's authorization,
store private license documents, or transmit credentials. Each downstream
operator must independently obtain the checkpoint from MiniMax and satisfy the
terms applicable to their territory, deployment, users, and outputs.

## Bring-Your-Own-Weights Contract

- The repository and its release artifacts contain no MiniMax H3 weights.
- Build, installation, and normal server startup never download H3.
- The operator supplies an explicit local model directory.
- The loader validates the pinned manifest, shard inventory, sizes, hashes,
  configurations, and tensor metadata before exposing the model.
- Missing, partial, mismatched, or unsupported files fail closed.
- The runtime does not search public mirrors or silently select another H3
  revision.
- Derived quantized weights remain local and are not publication artifacts.

The official Hugging Face acceptance flow or a separate MiniMax authorization
is the source of access. A Gufo command-line flag or configuration entry must
never pretend to replace that external process.

## Distribution and Serving Boundary

gufo distributes numerical engine source and binaries only. It does
not distribute the H3 checkpoint, tokenizer, converted weights, or a hosted H3
service.

An operator who exposes H3 through the asynchronous video API owns the
deployment obligations imposed by the applicable MiniMax terms, including as
relevant:

- binding users to required restrictions;
- acceptable-use and safeguard controls;
- reporting and abuse-handling mechanisms;
- generated-content disclosures and attribution;
- territorial and commercial-use conditions;
- retention and removal procedures after authorization ends.

The server documentation must identify the selected model as MiniMax H3 and
must not imply that the engine authors sublicense the model.

## Release Gate

A source or binary release fails the H3 boundary when any of the following is
true:

- a model weight, tokenizer payload, generated quantized artifact, or private
  authorization file is included;
- startup or build logic automatically downloads the H3 checkpoint;
- the pinned source revision or public license metadata is absent;
- H3 can start from a partial or unvalidated local directory;
- server documentation describes the engine license as covering H3;
- a packaged example points users to an unofficial weight mirror.

Tracked test fixtures may contain only small synthetic tensors generated by
the project. They must not contain extracted checkpoint values.

## Required Runtime Behavior

The H3 implementation issues must preserve this boundary:

1. The loader consumes only an explicit operator-supplied path.
2. Artifact validation runs before device allocation or request admission.
3. API model discovery distinguishes engine support from local model
   availability.
4. A missing H3 checkpoint returns a deterministic configuration error.
5. Logs may report public repository/revision metadata but never access tokens,
   credentials, or private authorization material.
