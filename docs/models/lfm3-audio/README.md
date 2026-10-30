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
# Live (English) recognition: text of each 0.96 s chunk as soon as it is heard.
build/release/gufo transcribe --model $M --mmproj $P --live speech.wav
# Speech synthesis: 18 languages/locales, female or male voice.
build/release/gufo speak --model $M --mmproj $P --language en-gb \
  --voice male --text 'The quick brown fox.' --output fox.wav
# Voice chat: a spoken (or written) message answered in text and speech.
build/release/gufo speak --model $M --mmproj $P --mode chat \
  --audio question.wav --output reply.wav
# Text answers, optionally after reasoning (<think> prefill).
build/release/gufo speak --model $M --mmproj $P --mode text --reasoning \
  --audio question.wav
# Spoken requests calling functions: tools.json holds OpenAI function tools;
# calls are printed (and reported with --format json) instead of an answer.
build/release/gufo speak --model $M --mmproj $P --mode chat \
  --tools tools.json --audio request.wav --output reply.wav
# HTTP and realtime WebSockets.
build/release/gufo serve tts --model $M --mmproj $P
build/release/gufo serve asr --model $M --mmproj $P
```

`--system` replaces the model card's prompt of a chat or text mode
("Respond with interleaved text and audio.", none for text, and with tools
"Respond with function calls, and interleaved text and audio.\n\nList of
tools:" or "List of tools:"). The tool definitions follow the system prompt
inside `<tools>`, one function object per line, as the official template.
With tools the model chooses between calls and an answer; a spoken reply
starts with `<tool_call>` or `<|mixed_start|>`.

`serve tts` exposes `/v1/audio/speech` (WAV, PCM, streamed audio and SSE)
and `/v1/audio/voices` (`female`, `male`);
`language` takes a code (`en`, `en-gb`, `fr`, …) or English name.
`greedy: true` selects the most likely speech codes; otherwise the model
card's speech sampling is used (temperature 1.0, top-k 128, repetition
penalty 1.1 over 4096 frames) with the request `seed`. The Qwen-specific
sampling fields do not apply to this model. Concurrent speech requests
(up to eight) share decoding steps: every step runs one backbone pass and
one depth-decoder pass for all of them, so their weights are read once (see
[benchmarks](BENCHMARKS.md)); a lone request takes the single-stream path.
Each request samples with its own seed; batched arithmetic may round
differently from a lone request. `serve asr` exposes
`/v1/audio/transcriptions` (JSON, text, verbose JSON, SSE), transcribing
recordings in 30 s windows, and the realtime WebSockets below.

Inputs are PCM16/24/32 or float32 WAV, mixed to mono and resampled
(windowed sinc) to 16 kHz. Streaming synthesis emits the first 80 ms frame as
soon as it is decoded, then chunks of four frames; the codec of one chunk runs
while the backbone computes the next frame.

## Realtime WebSockets

`serve asr` accepts OpenAI Realtime WebSocket clients on `/v1/realtime`.
Clients capture the microphone, detect turns themselves
(`turn_detection: null`) and stream mono PCM16 at 24 kHz in
`input_audio_buffer.append` events of any size; the server resamples and
encodes the stream causally as it arrives.

**Transcription** (`?intent=transcription`): with no transcription prompt and
English or no language, the session transcribes live. Text deltas
(`conversation.item.input_audio_transcription.delta`) arrive while audio is
still being appended, for the item that the next `input_audio_buffer.commit`
creates; commit flushes the stream and sends `...completed`. The model writes
a word once the audio after it has arrived, so commit after the speaker's
trailing silence. Other languages and prompts transcribe the buffer at
commit. `input_audio_buffer.clear` drops the utterance.

**Conversation** (no `intent`): one spoken conversation per connection, on a
private 8192-slot context (one slot per text token, audio token or 80 ms of
speech; the oldest turns leave when a reply needs room).

| Client event | Effect |
| --- | --- |
| `session.update` | `instructions`, `output_modalities` (`["audio"]` or `["text"]`), `tools` (function tools; `tool_choice` `auto`), `max_output_tokens`, and for text replies the Gufo extension `reasoning: {"effort": "none" \| "low" \| "medium" \| "high"}` |
| `input_audio_buffer.append` | Opens a user turn and prefills its audio while the user speaks |
| `input_audio_buffer.commit` / `.clear` | Closes the turn (`input_audio_buffer.committed`, `conversation.item.created`) or drops it |
| `conversation.item.create` | A user message of `input_text` or one `input_audio` part, or a `function_call_output` |
| `response.create` | Optional `output_modalities` and `max_output_tokens`; the reply streams `response.output_audio_transcript.delta` and `response.output_audio.delta` (base64 PCM16, 24 kHz), or `response.output_text.delta` (and `response.reasoning_text.delta`), then `function_call` items and `response.done` |
| `response.cancel` | Stops the reply within one generation step (`status: cancelled`) |

Without instructions the session uses the model card's prompt for its output
and tools; changing instructions, tools or the session's output modality
recomputes the context. Results of one reply's calls form one tool turn.
Earlier reasoning leaves the context when the user speaks again, as the
official template renders history. Events sent during a reply are handled
after it; at most four realtime sessions (transcription or conversation) run
at once, each on its own context beside the HTTP requests.

Not supported: server voice-activity detection, transcription of the user's
audio inside a conversation session, speech voices other than the model's
own in conversations, the Omni and VL checkpoints (separate models), and Q4
backbones. Digital silence transcribes as a short word ("Okay."), as in the
FP32 reference.

[Quality](QUALITY.md) · [Benchmarks](BENCHMARKS.md) · [Experiments](EXPERIMENTS.md)
