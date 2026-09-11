# Qwen3 Audio Benchmarks: gufo vs audio.cpp

Comparison of gufo's native Qwen3-TTS and Qwen3-ASR paths against
[audio.cpp](https://github.com/0xShug0/audio.cpp) on the same hardware, the
same model weights, and the same numeric precision.

**Headline:** gufo is 1.40x faster on TTS and 1.62x faster on ASR at a single
request, and holds a 1.45x / 1.66x throughput lead under concurrency. Both
engines serialize audio work, so neither gains throughput as concurrency rises.

## Environment

| | |
| --- | --- |
| CPU / GPU | AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, Strix Halo) |
| Memory | 125 GiB unified |
| gufo | `ed96ba3` + local audio CLI fixes, `nix build` (HIP, `gfx1151`) |
| audio.cpp | `3174e6b`, `nix build .#rocm-gfx1151` (HIP + `strixHaloOptimizations`) |
| Backend | ROCm/HIP on both sides |
| Date | 2026-09-10 |

Both engines run the same backend family, so the measurement isolates
implementation rather than conflating it with a Vulkan-vs-ROCm difference.

## Model parity

gufo's Qwen3-TTS and Qwen3-ASR loaders are safetensors-only — `src/models/qwen3_tts/`
contains no GGUF support at all — so the two engines cannot share a file. Weight
parity was achieved instead by packing the **exact safetensors directories gufo
loads** into audio.cpp's own GGUF layout at bf16:

```sh
# TTS
audiocpp_gguf \
  --input model_weights=$SRC/model.safetensors \
  --input speech_tokenizer_weights=$SRC/speech_tokenizer/model.safetensors \
  --output qwen3-tts-1.7b-base-bf16.gguf --type bf16 \
  --family qwen3_tts --model-spec model_specs/qwen3_tts.json --root $SRC

# ASR (sharded checkpoint: pass the index under the root namespace)
audiocpp_gguf \
  --input $SRC/model.safetensors.index.json \
  --output qwen3-asr-1.7b-bf16.gguf --type bf16 \
  --family qwen3_asr --model-spec model_specs/qwen3_asr.json --root $SRC
```

| | gufo | audio.cpp |
| --- | --- | --- |
| TTS checkpoint | Qwen3-TTS-12Hz-1.7B-Base | same, packed to GGUF |
| ASR checkpoint | Qwen3-ASR-1.7B | same, packed to GGUF |
| Precision | BF16 | BF16 |
| Container | safetensors | GGUF |

A bf16 tensor is bit-identical whichever container holds it. The ASR results
below confirm this empirically: all six transcripts matched byte-for-byte
between engines.

Pre-built GGUFs from `ggml-org` are **not** usable here — they are
llama.cpp-flavored and fail with `packed GGUF namespace does not exist:
model_weights`. The `audiocpp_gguf` packing step is required, and conveniently
is also what makes the comparison honest.

## Method

- Both engines run as **HTTP servers with the model preloaded**, so model load
  time is excluded on both sides. Requests go to the same OpenAI-compatible
  endpoints: `POST /v1/audio/speech` and `POST /v1/audio/transcriptions`.
- Identical voice presets on both sides, pointing at the same reference WAVs
  and transcripts (gufo `--voice NAME=PATH`, audio.cpp `voice_presets` in its
  server config).
- Greedy decoding on both (gufo `"greedy": true`, audio.cpp
  `"options": {"do_sample": false}`), seed 42.
- **2 warmup requests**, then median of 3. Warmup matters: audio.cpp's first
  greedy request costs 17.8 s against a 2.5 s steady state, because the greedy
  code path triggers its own CUDA-graph warmup on first touch. A single warmup
  run understates it by ~10x.
- Primary metric is **RTF** (wall clock / generated-or-input audio duration);
  lower is better, and < 1.0 means faster than realtime. RTF is used rather
  than wall time because the two engines emit slightly different audio
  durations for identical text.

Run-to-run variance was +/-0.01 s across all cases.

## TTS, single request

RTF (lower is better):

| case | audio | gufo | audio.cpp | gufo faster |
| --- | --- | --- | --- | --- |
| short EN | 4.00 s | 0.509 | 0.687 | 1.35x |
| medium EN | 10.24 s | 0.441 | 0.631 | 1.43x |
| short IT | 4.40 s | 0.499 | 0.700 | 1.40x |
| medium IT | 10.32 s | 0.440 | 0.632 | 1.44x |
| **mean** | | **0.472** | **0.663** | **1.40x** |

In realtime multiples: gufo 2.12x realtime, audio.cpp 1.51x. gufo's lead widens
slightly on longer text (1.35x -> 1.44x), which suggests the advantage is in
sustained decode rather than per-request setup.

## ASR, single request

| case | audio | gufo | audio.cpp | gufo faster | transcript |
| --- | --- | --- | --- | --- | --- |
| short EN | 4.00 s | 0.077 | 0.119 | 1.55x | identical |
| medium EN | 10.24 s | 0.064 | 0.105 | 1.65x | identical |
| short IT | 4.40 s | 0.091 | 0.143 | 1.58x | identical |
| medium IT | 10.32 s | 0.085 | 0.140 | 1.65x | identical |
| ref EN | 10.00 s | 0.060 | 0.099 | 1.66x | identical |
| ref IT | 10.00 s | 0.075 | 0.122 | 1.63x | identical |
| **mean** | | **0.075** | **0.121** | **1.62x** | **6/6** |

ASR is roughly 6x cheaper than TTS per second of audio on both engines: one
encoder pass plus a short text decode, versus autoregressive codec-frame
generation.

## Concurrency

Synchronized burst of C identical requests behind a start barrier. Throughput
is aggregate audio-seconds produced (TTS) or consumed (ASR) per wall second.

### TTS (medium EN, 10.24 s of audio per request)

| C | engine | wall | throughput | latency p50 | latency max |
| --- | --- | --- | --- | --- | --- |
| 1 | gufo | 4.48 s | 2.29 audio-s/s | 4.48 s | 4.48 s |
| 2 | gufo | 8.95 s | 2.29 | 6.72 s | 8.95 s |
| 4 | gufo | 17.90 s | 2.29 | 11.19 s | 17.90 s |
| 8 | gufo | 35.84 s | 2.29 | 20.17 s | 35.84 s |
| 1 | audio.cpp | 6.73 s | 1.58 | 6.73 s | 6.73 s |
| 2 | audio.cpp | 13.45 s | 1.58 | 10.09 s | 13.45 s |
| 4 | audio.cpp | 26.90 s | 1.58 | 16.82 s | 26.90 s |
| 8 | audio.cpp | 53.79 s | 1.58 | 30.25 s | 53.79 s |

### ASR (medium EN, 10.24 s of audio per request)

| C | engine | wall | throughput | latency p50 | latency max |
| --- | --- | --- | --- | --- | --- |
| 1 | gufo | 0.65 s | 15.71 audio-s/s | 0.65 s | 0.65 s |
| 2 | gufo | 1.30 s | 15.73 | 0.98 s | 1.30 s |
| 4 | gufo | 2.60 s | 15.73 | 1.63 s | 2.60 s |
| 8 | gufo | 5.21 s | 15.74 | 2.93 s | 5.20 s |
| 1 | audio.cpp | 1.08 s | 9.49 | 1.08 s | 1.08 s |
| 2 | audio.cpp | 2.16 s | 9.48 | 1.62 s | 2.16 s |
| 4 | audio.cpp | 4.31 s | 9.50 | 2.70 s | 4.31 s |
| 8 | audio.cpp | 8.64 s | 9.48 | 4.86 s | 8.64 s |

**Both engines serialize completely.** Throughput is flat to three significant
figures from C=1 to C=8 on both sides, wall time scales linearly with C, and
max latency equals `C x single-request latency`. On the gufo side this is
`TtsService`'s generation mutex; audio.cpp shows the same shape.

Practical consequence: concurrency buys nothing on either engine for audio
work. It only converts itself into queueing delay — at C=8, TTS tail latency
is 35.8 s (gufo) and 53.8 s (audio.cpp) for what is a 4.5 s / 6.7 s request in
isolation. Size deployments by request rate against single-stream throughput,
not by connection count.

Throughput ratios hold the single-request result: TTS 2.29 / 1.58 = **1.45x**,
ASR 15.73 / 9.49 = **1.66x**.

## Reproduction

```sh
# gufo, TTS
./result/bin/gufo serve --port 8099 --max-connections 32 audio \
  --tts-model models/Qwen3-TTS-12Hz-1.7B-Base \
  --voice narrator_eng=/persist/models/audio/clear-english-voice.wav \
  --voice narrator_ita=/persist/models/audio/clear-italian-voice.wav

# gufo, ASR
./result/bin/gufo serve --port 8099 --max-connections 32 \
  --max-request-bytes 33554432 audio --asr-model models/Qwen3-ASR-1.7B

# audio.cpp (either workload; model declared in the server config)
audiocpp_server --config acpp-server.json --backend hip --device 0 --no-ui
```

Reference transcripts are read from a `.txt` sidecar beside each reference WAV
on the gufo side, and from `voice_presets[].reference_text` on the audio.cpp
side; both were populated from the same source text.

## Scope and caveats

What these numbers do **not** cover:

- **CustomVoice and VoiceDesign** TTS variants. Only the Base checkpoint with
  in-context voice cloning was measured.
- **Quantized weights.** audio.cpp supports Q8_0/Q4_K; gufo's audio paths have
  no quantized route at all. At bf16 gufo wins, but audio.cpp at Q8_0 reads
  roughly half the weight bytes and could narrow or reverse the gap, at some
  quality cost. This is a real capability gufo lacks, not just an untested arm.
- **Streaming.** audio.cpp exposes `/v1/audio/speech/live` and
  `/v1/audio/transcriptions/live`; gufo has no streaming audio endpoint. For
  interactive use, time-to-first-audio may matter more than RTF, and that
  comparison cannot be made today.
- **Long-form input** beyond ~10 s of audio, and text long enough to exercise
  chunking.
- **Output quality.** RTF says nothing about how the audio sounds. Generated
  samples were kept for listening; no MOS or WER evaluation was run. The
  identical ASR transcripts are evidence of correct weight packing, not of
  synthesis quality.

This is one model family on one chip. It should not be generalized to
audio.cpp's other model families or to its CUDA/Metal backends.
