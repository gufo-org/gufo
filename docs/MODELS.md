# Supported models

All commands below use the `hf` CLI as provided by the Nix development
environment (`nix develop`), and place artifacts under `models/`, which is
where the benchmark and server examples in this repository expect them.
Strix Halo has unified memory, so "footprint" means what the weights plus
runtime state occupy of your total system memory (32, 64, or 128 GiB).

## Qwen3.8 27B

Download

```sh
# Q8 model
nix develop -c hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --repo-type model \
  --local-dir models/Qwen3.8-27B-GGUF

# Q4 model
nix develop -c hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --repo-type model \
  --local-dir models/Qwen3.8-27B-GGUF

# DFlash2 Q4
nix develop -c hf download z-lab/Qwen3.8-27B-DFlash2-GGUF \
  Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --repo-type model \
  --local-dir models/Qwen3.8-27B-DFlash2-GGUF

# DFlash2 Q8
nix develop -c hf download z-lab/Qwen3.8-27B-DFlash2-GGUF \
  Qwen3.8-27B-DFlash2-Q8_0.gguf \
  --repo-type model \
  --local-dir models/Qwen3.8-27B-DFlash2-GGUF
```

Supported modality: text

Supported tech: DFlash2 speculative decoding. The DFlash2 companion is the
draft model; it is optional but recommended, at ~2.91x speedup at fixed draft
width 7 on the Q8 target. Pair the companion with the target quant:
`UD-Q8_K_XL` + `DFlash2-Q8_0` (2.06 GB) or `UD-Q4_K_XL` + `DFlash2-Q4_K_M`
(1.14 GB). A BF16 companion (3.86 GB) also ships but adds ~3 GB of device
copies for no measured quality gain.

Machine footprint:

| Target       | Weights on disk | Decode weight traffic | Fits                             |
| ------------ | --------------: | --------------------: | -------------------------------- |
| `UD-Q8_K_XL` |       26.12 GiB |         28.0 GB/token | 32 GiB tight; 64 GiB comfortable |
| `UD-Q4_K_XL` |       16.35 GiB |         17.5 GB/token | 32 GiB comfortable               |

Decode is DRAM-bandwidth bound, so the Q4 target is ~1.6x faster to decode
while Q8 gives the best quality. Pick Q8 when memory allows, Q4 when you want
room for something else.

## DeepSeek V4 Flash

Download

```sh
# Target model (Q2-imatrix: IQ2XXS + w2Q2K with Q8 projections)
nix develop -c hf download antirez/deepseek-v4-gguf \
  DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --repo-type model \
  --local-dir models/deepseek-v4-gguf

# Optional DSpark draft support artifact
nix develop -c hf download antirez/deepseek-v4-gguf \
  DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --repo-type model \
  --local-dir models/deepseek-v4-gguf
```

Supported modality: text

Supported tech: DSpark speculative decoding. The DSpark support artifact is
optional and currently a small win at best (~1.1x mean) on this hardware;
leave it off unless experimenting.

Machine footprint: 80.76 GiB of weights; a 64K-context run plans 82.07 GiB
total including KV state and working buffers, and a running server consumes
~90 GiB. **128 GiB machine only.** The model loads in about 21 seconds via
mapped GGUF shards.

## Qwen3 audio

The audio loaders are **safetensors-only**; GGUF files (including pre-built
ones from `ggml-org`) are not supported yet.

Download

```sh
# ASR (sharded checkpoint; download the whole repo)
nix develop -c hf download Qwen/Qwen3-ASR-1.7B \
  --repo-type model \
  --local-dir models/Qwen3-ASR-1.7B

# TTS Base
nix develop -c hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --repo-type model \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-Base
```

The CustomVoice and VoiceDesign TTS variants live in the
[Qwen3-TTS collection](https://huggingface.co/collections/Qwen/qwen3-tts)
and follow the same layout.

Supported modality: audio (ASR: speech to text, TTS: text to speech)

Supported tech: voice presets from reference WAVs (`--voice NAME=PATH`),
served through the OpenAI-compatible endpoints `/v1/audio/speech` and
`/v1/audio/transcriptions`, or directly via `gufo transcribe`.

Machine footprint: 1.7B BF16 models, a few GiB each; fits any machine
configuration and can run alongside a text model.

## MiniMax H3

Download

```sh
nix develop -c hf download MiniMaxAI/MiniMax-H3 \
  --repo-type model \
  --local-dir models/MiniMax-H3
```

The upstream repository is license-gated: sign in with `hf auth login` and
accept the license on the model page before downloading.

Supported modality: text to video

Supported tech: official MiniMaxAI checkpoint only, executed through
`gufo video`. Only MiniMaxAI's official implementation is supported, to
retain full model quality.

Machine footprint: peak device residency has been measured between ~43 and
~59 GiB with zero swap; a 64 GiB machine is the practical minimum and 128 GiB
is comfortable.
