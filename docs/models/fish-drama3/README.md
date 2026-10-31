# Fish Audio Drama-3 (30B-A3B MoE, 44.1 kHz)

Native gfx1151 speech synthesis and zero-shot voice cloning from the official
`fishaudio/drama-3` checkpoint (`F8_E4M3` routed MoE experts with `128x128`
`F32` block scales, `BF16` dense backbone/tower weights, and an `F32` 44.1 kHz
causal Fish DAC codec loaded directly from `codec.pth`). Python is not a
build or runtime requirement.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Usage

```sh
nix develop -c hf download fishaudio/drama-3 --local-dir models/drama-3
nix build
./result/bin/gufo serve tts \
  --model models/drama-3 \
  --voice narrator=reference.wav --voice-text narrator=reference.txt
```

`gufo serve tts` detects `fish_qwen3_omni` automatically from `config.json`
and folds the weight-normalized convolutions in `codec.pth` on load.
The default HTTP model ID is `fish-drama-3` (override with
`--served-model-name`).

```sh
curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"fish-drama-3","voice":"default","input":"Hello from Gufo on AMD Strix Halo."}' \
  --output speech.wav
```

`GET /v1/audio/voices` lists registered voices (`default`, `voice-clone`, and
any `--voice` presets). Requests may also supply inline base64 WAV
`reference_audio` and `reference_text` for zero-shot in-context voice cloning;
see [the audio API](../../SERVER.md#other-model-services).

## Sampling and streaming

Output audio is signed little-endian 16-bit mono PCM at **44.1 kHz**
(`2048` samples per codec frame, `21.53 Hz` frame rate).
Request `response_format: "pcm"` (or `stream_format: "sse"`) to stream audio in
4-frame causal chunks (`185.8 ms` per chunk) while autoregressive decoding
continues on a separate HIP stream:

```sh
curl -N http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"fish-drama-3","voice":"default","input":"Streaming synthesis on gfx1151.","response_format":"pcm"}' \
  --output speech.pcm
```

WebSocket streaming is available at `/v1/audio/speech/stream` using the same
protocol as [Qwen3-TTS](../qwen3-tts/README.md#sampling-and-streaming).
