# Qwen3-TTS 12Hz 1.7B

Native gfx1151 speech synthesis from BF16 safetensors, with an F32 waveform
decoder. Checkpoint configuration selects the variant; 0.6B, 25Hz and GGUF
checkpoints are unsupported. Official Python is an offline oracle only.

[Benchmarks](BENCHMARKS.md) · [Evaluation](EVALUATION.md) · [Experiments](EXPERIMENTS.md)

## Modes

| Checkpoint suffix | HTTP model ID | Conditioning |
| --- | --- | --- |
| `CustomVoice` | `qwen3-tts-12hz-1.7b-customvoice` | `voice`: checkpoint speaker name. |
| `VoiceDesign` | `qwen3-tts-12hz-1.7b-voice-design` | `instructions`: voice description. |
| `Base` | `qwen3-tts-12hz-1.7b-base` | Reference audio and transcript (ICL), or `voice_clone_mode: "speaker_embedding_only"`. |

```sh
nix develop -c hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-Base
nix build
./result/bin/gufo serve audio \
  --tts-model models/Qwen3-TTS-12Hz-1.7B-Base \
  --voice narrator=reference.wav --voice-text narrator=reference.txt
```

Substitute `CustomVoice` or `VoiceDesign` in the download and model directory
for those variants; reference-voice options apply to Base. `--voice NAME=WAV`
registers a reusable clone. `--voice-text` supplies a transcript (text or file),
otherwise a sibling `.txt` is used; absent text selects speaker-only cloning.
`--voice-lang NAME=LANGUAGE` sets that voice's default language.

```sh
curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts-12hz-1.7b-base","voice":"narrator","input":"Hello from Gufo."}' \
  --output speech.wav
```

`GET /v1/audio/voices` lists usable voices. Base also accepts per-request
`reference_audio` and `reference_text`; see [the audio API](../../SERVER.md#other-model-services)
for formats, limits and language/sampling controls. Both audio services can
share a process by adding `--asr-model`.
Repeated Base requests reuse the same reference's encoder features and exact
waveform-decoder state. One reference prefix is retained per runtime; changing
voices replaces it. This preserves generated audio and reduces warm latency.

## Sampling and streaming

Default sampling follows the checkpoint: top-k 50, top-p 1, temperature 0.9
for both talker and predictor, and repetition penalty 1.05. Request controls:

| Talker | Predictor |
| --- | --- |
| `temperature`, `top_k`, `top_p` | `subtalker_temperature`, `subtalker_top_k`, `subtalker_top_p` |
| `greedy: true` disables sampling in both | `subtalker_dosample` can override predictor sampling |

`top_k: 0` disables that filter. `seed` controls request-local replay;
`repetition_penalty` applies to the talker's generated codec tokens.
Greedy generation can fail to reach EOS; bound diagnostic requests with
`max_new_tokens`. Read the [quality limits](EVALUATION.md#known-limitations).

WAV is buffered so its length header is correct. To receive audio as it is
generated, request `response_format: "pcm"`:

```sh
curl -N http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts","voice":"narrator","input":"Hello from Gufo.","response_format":"pcm"}' \
  --output speech.pcm
```

PCM is signed little-endian 16-bit mono at 24 kHz. `stream_format: "sse"`
instead emits OpenAI `speech.audio.delta` events containing base64 `audio`,
then `speech.audio.done`; SSE requires PCM. This streams the same generation
as a complete-text request, without sentence splitting or a server flag.

For text arriving from an LLM, connect a WebSocket to
`/v1/audio/speech/stream` and send the vLLM-Omni speech protocol:

```jsonl
{"type":"session.config","model":"qwen3-tts","voice":"narrator","response_format":"pcm","stream_audio":true,"split_granularity":"sentence"}
{"type":"input.text","text":"Hello from Gufo. "}
{"type":"input.text","text":"This text arrived later."}
{"type":"input.done"}
```

Each segment produces `audio.start`, binary PCM frames and `audio.done`.
`session.done` finishes the utterance; the connection accepts another one.
`session.close` closes it. Base also accepts `ref_audio`, `ref_text` and
`x_vector_only_mode` in this protocol, or a registered voice as above.

`split_granularity` defaults to `none`, buffering text until `input.done`.
`sentence` or `clause` starts generation earlier at text boundaries, independent
of packet sizes. Splitting changes future context and can change prosody; it
does not promise the same audio as synthesizing the complete text. Timestamps,
VAD and forced alignment are not loaded. Slow/disconnected consumers cancel
generation; waiting requests are bounded and cancellable.
