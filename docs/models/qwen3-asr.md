# Qwen3-ASR 1.7B

Qwen3-ASR turns speech into text. Gufo runs the 1.7B BF16 checkpoint, for
recorded files and for live audio over a WebSocket.

## Files

```sh
hf download Qwen/Qwen3-ASR-1.7B \
  --revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --local-dir models/Qwen3-ASR-1.7B
```

The 0.6B checkpoint, forced alignment and audio GGUF files aren't supported.

## Run

```sh
gufo transcribe --model models/Qwen3-ASR-1.7B --audio speech.wav
gufo serve asr --model models/Qwen3-ASR-1.7B
curl -sS http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@speech.wav -F model=qwen3-asr-1.7b -F response_format=json
```

The endpoint accepts `response_format` set to `json`, `text` or
`verbose_json`, plus optional `language` and `prompt`. Decoding is always
greedy, so `temperature` must be 0.

## Audio input

Gufo reads WAV files with 16-, 24- or 32-bit integer or 32-bit float samples,
up to eight channels and 192 kHz, and converts them to 16 kHz mono. A
recording can be up to 30 minutes long; the CLI also caps input at 256 MiB,
and the server applies its request size limit.

Gufo splits long recordings at quiet points, transcribes each piece and joins
the results. `--context` sets the capacity for one piece, and `--max-tokens`
(CLI) or `max_tokens` (HTTP) limits the output **per piece**, so longer
recordings need no extra capacity.

## Streaming a file

Add `-F stream=true -F response_format=json` to receive the transcript while
it is decoded, as OpenAI `transcript.text.delta` events followed by
`transcript.text.done`. The final text is the same as without streaming.

## Live audio

For microphone input, open a WebSocket to
`/v1/realtime?intent=transcription` and use OpenAI's manual-turn messages:

```jsonl
{"type":"session.update","session":{"type":"transcription","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":{"model":"qwen3-asr-1.7b","language":"en"},"turn_detection":null}}}}
{"type":"input_audio_buffer.append","audio":"<base64 PCM16 audio>"}
{"type":"input_audio_buffer.commit"}
```

Send 16-bit mono PCM at 24 kHz, appending as audio arrives. Commit each
utterance when it is complete. The server answers with
`input_audio_buffer.committed`, a conversation item, and
`conversation.item.input_audio_transcription.delta` and `.completed` events.
`input_audio_buffer.clear` drops audio not yet committed. An utterance must be
between 100 ms and 120 seconds long, and one connection can carry many.

Shorter utterances return text sooner, but each is transcribed on its own, so
the joined text can differ from transcribing the whole recording at once.
Gufo doesn't detect speech turns itself, revise earlier text, or return
timestamps.

## Serving

Two requests can prepare audio at the same time; the GPU work runs one piece at
a time in arrival order, and cancelling a request removes its queued pieces.
`--served-model-name` sets the model ID clients send; repeat it to accept
aliases. Speech synthesis runs as a separate server, `gufo serve tts`.

## Performance

As of September 21, 2026, on Strix Halo, transcription runs at **15.27×
real time** once the model is loaded. See the
[full benchmarks](../internal/models/qwen3-asr/BENCHMARKS.md) and the
[quality report](../internal/models/qwen3-asr/QUALITY.md).
