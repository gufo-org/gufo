# Qwen3-ASR 1.7B

Native BF16 safetensors speech recognition on gfx1151. The 1.7B checkpoint is
supported; 0.6B, forced alignment and audio GGUFs are unsupported. Python runs
only offline reference checks. Generation is deterministic greedy decoding.

[Benchmarks](BENCHMARKS.md) · [Evaluation](EVALUATION.md) · [Experiments](EXPERIMENTS.md)

```sh
nix develop -c hf download Qwen/Qwen3-ASR-1.7B \
  --revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --local-dir models/Qwen3-ASR-1.7B
nix build
./result/bin/gufo transcribe --model models/Qwen3-ASR-1.7B --audio speech.wav
./result/bin/gufo serve asr --model models/Qwen3-ASR-1.7B
```

```sh
curl -sS http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@speech.wav -F model=qwen3-asr-1.7b -F response_format=json
```

Inputs: PCM16/24/32 or float32 WAV, up to eight channels and 192 kHz. The
frontend mixes to mono and uses windowed-sinc resampling to 16 kHz.
The WAV loader accepts recordings up to 30 minutes; HTTP request-byte limits
and the CLI's 256-MiB input limit still apply.
HTTP supports `json`, `text`, `verbose_json`, optional `language` and `prompt`,
and temperature zero. Long recordings are split at low-energy boundaries
within the configured context capacity, then transcribed and merged. CLI
`--max-tokens` and HTTP `max_tokens` limit output **per chunk**; increasing the
recording length does not require increasing model capacity.

## Streaming

For an uploaded file, add `-F stream=true -F response_format=json`. The server
emits OpenAI `transcript.text.delta` events while decoding, followed by
`transcript.text.done`. This retains the same transcript as buffered execution.

For microphone input, connect a WebSocket to `/v1/realtime?intent=transcription`:

```jsonl
{"type":"session.update","session":{"type":"transcription","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":{"model":"qwen3-asr-1.7b","language":"en"},"turn_detection":null}}}}
{"type":"input_audio_buffer.append","audio":"<base64 PCM16 audio>"}
{"type":"input_audio_buffer.commit"}
```

Send mono PCM16 at 24 kHz; Gufo resamples to 16 kHz. Append messages can arrive
continuously. Commit an utterance when it is ready for transcription; the server
returns `input_audio_buffer.committed`, an identified conversation item, and
`conversation.item.input_audio_transcription.delta` / `.completed` events.
Further utterances reuse the connection. `input_audio_buffer.clear` discards
uncommitted audio. Each buffer is limited to 120 seconds, with a 100-ms minimum
for commit.

This uses OpenAI's manual-turn interface. It does not implement server VAD,
revisable rolling hypotheses, timestamps or forced alignment. Choosing shorter
utterances reduces latency and future context; their concatenated transcript
need not equal recognition of a complete recording.

Two requests can prepare CPU frontends concurrently. Device state remains
isolated by FIFO admission between audio chunks; cancellation removes queued
work. `--context` controls capacity per chunk; `--served-model-name` sets
the public model ID. TTS runs separately with `gufo serve tts`.
See [server limits](../../SERVER.md) and [quality gaps](EVALUATION.md).
