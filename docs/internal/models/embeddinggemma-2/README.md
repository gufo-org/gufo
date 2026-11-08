# EmbeddingGemma 2 implementation notes

Developer detail behind the [public model page](../../../models/embeddinggemma-2.md).
[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md) · [ROCm](ROCM.md)

## Images

Use llama.cpp's typed input form. Each `content` object produces one embedding;
strings and content objects may be mixed in the same request. Text parts are
concatenated in order, including whitespace; the server adds no task prefix.

```json
{
  "model": "embeddinggemma-2",
  "input": [
    {"content": [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      {"type": "text", "text": "\n"}
    ]},
    {"content": [{"type": "text", "text": "task: search result | query: a red car"}]}
  ],
  "encoding_format": "float"
}
```

This is the request form used by Frigate 0.19's development client. Frigate
0.18 uses llama.cpp's non-OpenAI `POST /embeddings`, which the same process
also serves; see [Embeddings](../../../server.md#embeddings).

Images may be PNG, JPEG, or WebP, supplied as base64 data URLs or public HTTPS
URLs. Shared [image read limits](../../../server.md#embeddings) apply: at most 20 MiB
of encoded images across a request, 32 megapixels per decoded image, and a
15-second read budget. Embedding images additionally have a 32,768-pixel side
limit and a maximum of 32 images per request. The HTTP request-body limit also
applies to base64. Private destinations and local files are refused.

Preprocessing follows the pinned checkpoint's `Gemma4ImageProcessor`: RGB,
antialiased uint8 bicubic resize preserving aspect ratio, no crop, rescale by
1/255, and 16×16 patches. The 3×3 spatial pool yields at most 280 soft tokens
per image by default; the actual count depends on aspect ratio. A request can
pick the processor's other budgets with
`"mm_processor_kwargs": {"max_soft_tokens": N}`, where N is 70, 140, 280, 560
or 1120, as in vLLM; one budget applies to every image of the request, including
mixed batches. Video frames keep their separate budget
(`video_options.max_soft_tokens`, default 140; see [Video](#video)). At 1120
the 8,192-token limit fits at most seven images that use the whole budget. These tokens and their
boundary markers count toward the 8,192-token input limit and reported usage.
The 32-image request cap is a deliberate server resource limit, not a model
limit: it bounds retained decoded patch memory across all inputs in a request.
At budget 70, 32 square images fit in one input (2,114 tokens including
boundaries); a 33rd image is still refused with `too_many_images`. The cap is
shared across inputs, even when each input separately fits the context. Patches
use 27,648 bytes per soft token, so 32 full-budget 1120-token images across
multiple inputs can retain about 945 MiB of patch data, plus decoding and
framing overhead. The shared media byte/read-time and per-image pixel limits
still apply. Smaller budgets reduce vision work and memory at the cost of
spatial detail; [benchmarks](BENCHMARKS.md#image-budgets) record measured costs.
Text parts containing literal `<|image|>` placeholders are refused when images
are present. Unsupported processor or vision configurations are refused.

Gufo's reference-matched resize differs from llama.cpp's preprocessing.
Re-embed an existing llama.cpp image index before switching servers; the
vectors are not interchangeable. [Quality](QUALITY.md) records agreement with
the official float32 model, rather than a retrieval benchmark or full Frigate
integration test.

## Audio

Typed `input_audio` parts may appear alone or interleaved with text and images,
including multiple clips in one input:

```json
{"input":[{"content":[
  {"type":"text","text":"a sound: "},
  {"type":"input_audio","input_audio":{"data":"<base64 WAV>","format":"wav"}}
]}]}
```

The contract follows llama.cpp at `68d9053afd4f4d0752ced6187585f862355a40be`:
`input_audio.data` takes precedence over `input_audio.url`; both are strings.
`format` is an optional string and is ignored; the bytes determine the format.
Raw base64, base64 audio data URLs and public HTTPS URLs are supported. Local
files and private network destinations are refused. Audio data URL media types
are `audio/wav`, `audio/x-wav`, `audio/mpeg`, `audio/mp3`, `audio/flac`, and
`audio/ogg`. Supported containers are WAV (PCM or float), FLAC, MP3, and Ogg.
Audio and images share the request's 20 MiB encoded-byte and 15-second read
budgets. At most 32 clips are accepted, each at most 30 seconds, 8 channels,
and 8–192 kHz; decoding also has a 15-second deadline. The HTTP body limit applies.

The decoder pins the official processor's `librosa` backend: libsndfile float32,
arithmetic channel mean, SOXR HQ resampling to 16 kHz, then zero padding to the
ceil sample length. There is no waveform amplitude normalization. The model
frontend uses 320-sample periodic Hann windows and a 160-sample hop, a 512-point
magnitude FFT, 128 HTK mel filters and `log(mel + 0.001)`. Semicausal left padding,
128-sample waveform padding, full-frame masks and two stride-2 subsampling masks
follow the pinned official implementation. Clips shorter than 161 resampled
samples have no valid feature token and return 400 `invalid_audio_input`.

Each valid soft token is inserted between `<|audio>` and `<audio|>` in the shared
text backbone. Counts come from the actual subsampled validity mask, rather than
an estimated duration or the processor's `audio_seq_length` metadata. Expanded
soft tokens and all boundary tokens count toward the shared 8,192-token context
and usage. Truncation preserves every image and audio item whole, or refuses the
input. Literal `<|audio|>` text is refused when clips are supplied.

The 12-layer audio tower and projector load on the first audio request and reuse
their weights and bounded scratch. Audio computes in float32 using the checkpoint's
BF16 weights widened once; the shared text backbone retains its existing BF16
matrix inputs. An audio request runs in its own device batch. Cancellation is
checked during preprocessing and between audio layers; failures drain queued
work and use the existing device-loss probe before the next request.

## Video

Typed `input_video` parts accept the llama.cpp `data`/`url` contract:

```json
{
  "input": [{"content": [
    {"type": "text", "text": "a camera clip: "},
    {"type": "input_video", "input_video": {"data": "<base64 MP4>"}},
    {"type": "text", "text": " near the entrance"}
  ]}],
  "video_options": {"fps": 1, "max_frames": 32,
                    "overflow_strategy": "uniform", "add_timestamps": false,
                    "max_soft_tokens": 140}
}
```

`data` takes precedence over `url`. Either accepts raw base64, a base64 video
data URL, or a public HTTPS URL. Local paths and private destinations are
refused. Supported containers are MP4/MOV and WebM/Matroska, with codecs supplied
by the linked FFmpeg build (including H.264 and VP9 in the qualified build).
Data URL media types are `video/mp4`, `video/quicktime`, `video/webm` and
`video/x-matroska`. All media share a 20 MiB byte budget and 15-second read
deadline; the server's HTTP body limit also applies before base64 decoding.

Defaults match the pinned processor: 1 FPS, at most 32 frames, uniform overflow,
140 soft-token budget per frame, and timestamps disabled. `video_options` applies
to every video in the request. `fps` is a finite number in `(0, 120]`,
`max_frames` an integer in `[1, 32]`, `overflow_strategy` is `uniform` or
`truncate`, and `add_timestamps` is boolean. `max_soft_tokens` selects the
per-frame vision budget, one of the image budgets 70, 140, 280, 560 or 1120;
each frame is resized like an image at that budget. It is independent of
`mm_processor_kwargs.max_soft_tokens`, which applies only to images.

Sampling uses the decoded frame count and average source FPS: select
`max(1, floor(duration * requested_fps))` indices by flooring source-frame
positions, then apply the cap using uniform linspace/floor indices or take the
first indices for `truncate`. Upsampling repeats selected frames. Timestamps
use the selected source indices divided by source FPS, formatted as `mm:ss`.
The decoded frame count is used even when a container omits its frame count.
FFmpeg decodes RGB pixels without applying display rotation, as the pinned PyAV
reference does. Frame size changes within a clip are refused.

Each frame uses image boundaries around **video** placeholders, preserving its
position among text, images, other videos and separate audio parts. All features
pass through the shared text backbone and produce one normalized vector per
input. Expanded placeholders count toward usage and the 8,192-token context;
`truncate` never drops part of a clip. Vision weights load lazily and remain
shared with image requests. FFmpeg runtime libraries load on the first video
request, preserving startup latency for other services. Missing decoder runtime
libraries are a server error (500), not malformed media.

The video path takes the first video stream and ignores embedded audio tracks,
matching the official visual decoder. To embed sound jointly, provide a separate
`input_audio` part in the desired reference order; Gufo does not automatically
extract or interleave the container's audio. The fixtures cover both an embedded
soundtrack and explicit text/video/image/audio interleaving.

Limits per clip are 300 seconds, 4,096 decoded frames, 4 megapixels per frame,
512 megapixels of decoded work across two passes, eight container streams and a
15-second decode/processing deadline. A request accepts at most 32 clips,
64 selected frames and 8,960 video soft tokens (64 frames at the default
budget), so a 1120-token budget admits at most eight full-budget frames per
request. The soft-token limit is checked frame by frame, before a frame's
patches are kept. Decoding first counts frames, then retains only the selected
frames' unpadded patches; it never stores the entire decoded clip. Cancellation
is checked during decoding, preprocessing and between GPU frame passes.
Malformed or over-budget clips return 400 `invalid_video_input`.

## Requests

Loading uploads the 540 MB text backbone, unmaps the checkpoint so the weights stay only in device memory, and then runs a short warm-up.  The warm-up times each hipBLASLt algorithm that needs no scratch space for each matrix shape and row bucket the encoder uses and keeps the fastest, so the first request pays no tuning cost.  The choice can differ between loads, which changes float32 accumulation order but not the result beyond rounding.

The first image or audio request uploads that modality's tower and projector; subsequent requests reuse them.

One device batch of up to 8,192 tokens runs at a time.  Text requests that wait for the device share a batch, oldest first, and a lone request runs at once.  A request with images or audio runs in a batch of its own, in arrival order, and its soft tokens count toward the batch.  A request that fits in one batch never spans two.  A larger request is split, and after each of its batches the requests that arrived meanwhile go first, so a short request waits for at most one batch of a long one.  A device failure that a probe confirms answers 503 `device_lost` and shuts the server down for its supervisor to restart, as the text server does.  Attention is bidirectional, so there is no cache to reuse between requests.  Gufo checks for cancellation before each batch that would take a request's inputs, so a cancelled request does not use the device again.  `--served-model-name` sets the public model ID (repeatable; extra names are accepted aliases).  Gufo supports only BF16 safetensors, not GGUF or quantized variants.

See [server limits](../../../server.md) and [quality gaps](QUALITY.md).
