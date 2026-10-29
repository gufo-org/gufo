# LFM3-Audio 2B

Native speech recognition, speech synthesis and voice chat for LiquidAI
LFM3-Audio-2B on Linux x86-64 / Strix Halo `gfx1151`. Implementation:
[src/models/lfm3_audio](../../../src/models/lfm3_audio), on top of the
[LFM3 backbone](../lfm3/README.md).

The checkpoint pairs the LFM3 1.7B backbone with three audio towers stored in
the mmproj GGUF:

- **Encoder** (16 kHz in): log-mel features (128 Slaney bins, 10 ms hop), three
  stride-2 convolutions over the (mel, time) plane, a causal 17-layer LFM2
  stack (short convolutions and 32-token sliding attention) and a projector
  into the backbone. One audio token covers 80 ms.
- **Depth decoder**: a bidirectional 3-layer LFM3 stack over the 12 codebook
  slots of one speech frame. Codebooks are revealed in a random order over
  three MaskGIT passes of four.
- **LiquidCodec decoder** (24 kHz out): FSQ latents, an 8-layer causal LFM3
  stack at 75 Hz with 30-frame sliding attention, and an inverse STFT
  (1280/320). One speech frame is 80 ms (1920 samples).

Each backbone step consumes one text token or one speech frame. Prompts
ending in `<|audio_start|>` speak; `<|mixed_start|>` alternates blocks of six
text tokens and twelve speech frames until `<|text_end|>`, then finishes the
speech. This follows the official `generate` state machine.

## Precision and acquisition

Use the **Q8_0 backbone with the BF16 mmproj**. BF16 is the original
precision of both files. The Q8_0 mmproj is accepted: it speeds up the depth
decoder by about a quarter, but moves codec audio by up to 7% of full scale
against the BF16 towers (see [quality](QUALITY.md)). Activations, attention,
convolution history, the inverse FFT and all accumulation are FP32. Speech
codes are sampled on the device; the frame embedding returned to the backbone
is the exact sum of the codebook embeddings.

```sh
REV=02355c1a141e464a064f2ab2c276638dbcb7f38a
hf download LiquidAI/LFM3-Audio-2B-GGUF \
  LFM3-Audio-2B-Q8_0.gguf mmproj-LFM3-Audio-2B-BF16.gguf \
  --revision $REV --local-dir models/lfm3-audio
```

Weights use the [LFM Open License](https://huggingface.co/LiquidAI/LFM3-Audio-2B/blob/main/LICENSE)
and are downloaded separately. `--mmproj` selects the audio towers; without
it, the single `mmproj*.gguf` with LFM3-Audio towers beside `--model` is used
(several such files are an error).

## Usage

```sh
M=models/lfm3-audio/LFM3-Audio-2B-Q8_0.gguf
P=models/lfm3-audio/mmproj-LFM3-Audio-2B-BF16.gguf
# Speech recognition; --language English (or en) pins the language.
build/release/gufo transcribe --model $M --mmproj $P speech.wav
# Speech synthesis: 18 languages/locales, female or male voice.
build/release/gufo speak --model $M --mmproj $P --language en-gb \
  --voice male --text 'The quick brown fox.' --output fox.wav
# Voice chat: a spoken (or written) message answered in text and speech.
build/release/gufo speak --model $M --mmproj $P --mode chat \
  --audio question.wav --output reply.wav
# HTTP: the same speech services as Qwen3-ASR / Qwen3-TTS.
build/release/gufo serve tts --model $M --mmproj $P
build/release/gufo serve asr --model $M --mmproj $P
```

`serve tts` exposes `/v1/audio/speech` (WAV, PCM, streamed audio and SSE)
and `/v1/audio/voices` (`female`, `male`);
`language` takes a code (`en`, `en-gb`, `fr`, …) or English name.
`greedy: true` selects the most likely speech codes; otherwise the model
card's speech sampling is used (temperature 1.0, top-k 128, repetition
penalty 1.1 over 4096 frames) with the request `seed`. The Qwen-specific
sampling fields do not apply to this model. `serve asr` exposes
`/v1/audio/transcriptions` (JSON, text, verbose JSON, SSE). Recordings are
transcribed in 30 s windows. The realtime WebSockets share these services but
have not been exercised with this model.

Inputs are PCM16/24/32 or float32 WAV, mixed to mono and resampled
(windowed sinc) to 16 kHz. Streaming synthesis emits the first 80 ms frame as
soon as it is decoded, then chunks of four frames; the codec of one chunk runs
while the backbone computes the next frame.

Not supported yet: live (incremental) ASR from a microphone stream inside
the model, tool calls from spoken requests, reasoning prefills in chat mode,
the Omni and VL variants, and Q4 backbones. Digital silence transcribes as a
short word ("Okay."), as in the FP32 reference.

[Quality](QUALITY.md) · [Benchmarks](BENCHMARKS.md) · [Experiments](EXPERIMENTS.md)
