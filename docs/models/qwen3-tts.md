# Qwen3-TTS 12Hz 1.7B

Qwen3-TTS turns text into speech. Gufo runs the three 1.7B BF16 variants: fixed
voices, voices described in words, and voices cloned from a recording.

## Variants

Download the variant you need; Gufo detects which one it is.

| Variant | HTTP model ID | How you choose the voice |
| --- | --- | --- |
| `CustomVoice` | `qwen3-tts-12hz-1.7b-customvoice` | `voice`: one of the checkpoint's speakers |
| `VoiceDesign` | `qwen3-tts-12hz-1.7b-voice-design` | `instructions`: a description of the voice |
| `Base` | `qwen3-tts-12hz-1.7b-base` | A reference recording, with or without its transcript |

The 0.6B, 25Hz and GGUF checkpoints aren't supported.

```sh
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-Base
```

## Run

```sh
gufo serve tts --model models/Qwen3-TTS-12Hz-1.7B-Base \
  --voice narrator=reference.wav --voice-text narrator=reference.txt
curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts-12hz-1.7b-base","voice":"narrator","input":"Hello from Gufo."}' \
  --output speech.wav
```

`GET /v1/audio/voices` lists the voices you can use.

## Cloning a voice

With the Base variant, `--voice NAME=WAV` registers a voice from a recording.
`--voice-text NAME=TEXT` gives its transcript, as text or a file; without it,
Gufo reads a `.txt` file next to the recording. With a transcript, the clone
learns from the recording and its words together, which gives the closest
match; without one, it copies only the speaker's voice.
`--voice-lang NAME=LANGUAGE` sets the voice's default language.

A request can also bring its own `reference_audio` and `reference_text`. Gufo
keeps the last reference ready, so repeated requests with the same voice start
faster; switching voices replaces it. The [audio API](../server.md#other-model-services)
lists accepted formats and limits.

## Sampling

Defaults follow the checkpoint: temperature 0.9, top-k 50, top-p 1 and a
repetition penalty of 1.05. The model has two parts, a **talker** that plans
the speech and a **predictor** that fills in detail, each with its own
controls:

| Talker | Predictor |
| --- | --- |
| `temperature`, `top_k`, `top_p` | `subtalker_temperature`, `subtalker_top_k`, `subtalker_top_p` |
| `greedy: true` turns sampling off for both | `subtalker_dosample` overrides the predictor alone |

`top_k: 0` turns that filter off, and `seed` makes a request repeatable. Greedy
generation can fail to stop, so cap it with `max_new_tokens`; see the
[known limitations](../internal/models/qwen3-tts/QUALITY.md#known-limitations).

## Streaming

WAV responses arrive complete, because the file header holds the length. To
play audio while it is generated, ask for `"response_format": "pcm"`: 16-bit
little-endian mono at 24 kHz.

```sh
curl -N http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts","voice":"narrator","input":"Hello from Gufo.","response_format":"pcm"}' \
  --output speech.pcm
```

Add `"stream_format": "sse"` to receive the PCM as OpenAI `speech.audio.delta`
events, ending with `speech.audio.done`.

## Text that arrives in pieces

When the text comes from an LLM as it is written, open a WebSocket to
`/v1/audio/speech/stream`, which speaks the vLLM-Omni protocol:

```jsonl
{"type":"session.config","model":"qwen3-tts","voice":"narrator","response_format":"pcm","stream_audio":true,"split_granularity":"sentence"}
{"type":"input.text","text":"Hello from Gufo. "}
{"type":"input.text","text":"This text arrived later."}
{"type":"input.done"}
```

Each segment returns `audio.start`, binary PCM frames and `audio.done`.
`session.done` ends the utterance and `session.close` the connection. With
Base, `ref_audio`, `ref_text` and `x_vector_only_mode` pick the voice.

`split_granularity` decides when speech starts. The default, `none`, waits for
`input.done`; `sentence` and `clause` start at each boundary. Splitting starts
sooner but can change the intonation compared with speaking the whole text. A
client that stops reading cancels the generation.

## Serving

`--context` sets token capacity and `--served-model-name` the model ID; repeat
it to accept aliases. Speech recognition runs as a separate server,
`gufo serve asr`; [llama-swap](../server.md#llama-swap) can put both behind
one URL.

## Performance

As of September 20, 2026, on Strix Halo, CustomVoice speaks at up to **2.54×
real time** and starts playing after **201 ms**. See the
[full benchmarks](../internal/models/qwen3-tts/BENCHMARKS.md) and the
[quality report](../internal/models/qwen3-tts/QUALITY.md).
