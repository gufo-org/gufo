# Gufo: a Strix Halo inference engine

Gufo is a vertical local inference engine specifically built and optimized for the AMD Strix Halo hardware:
Ryzen AI MAX+ 395 systems with Radeon 8060S (`gfx1151`), an XDNA2 NPU, and up to 128 GiB of unified memory.

Supported models:

- antirez's [DeepSeek-V4-Flash IQ2XXS](https://huggingface.co/antirez/deepseek-v4-gguf/blob/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf) GGUF with [DSpark](https://huggingface.co/antirez/deepseek-v4-gguf/blob/main/DeepSeek-V4-Flash-DSpark-support-0731.gguf) support.
  With its MoE architecture, 284B parameters (13B active), and quantization-aware training techniques, it is the largest and smartest text model that this hardware can run without losing too much of its full-quality accuracy.
- [Qwen3.8-27B:UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-Q4_K_XL.gguf) and [Qwen3.8-27B-UD-Q8_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-Q8_K_XL.gguf) GGUFs from unsloth with z-lab's [DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF). The best choice when you can't saturate your unified memory and want to leave room for something else.
- [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) Safetensors text to video generation model.
  Even though the community has built quicker implementations, as of now we decided to support just MiniMaxAI's official one to retain the full model quality.
- [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) Safetensors for audio to text.
- [Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts) 1.7B models: Base, CustomVoice, and VoiceDesign.

More info in [MODELS.md](./docs/MODELS.md)

## Quickstart

```sh
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --repo-type model \
  --local-dir models/Qwen3.8-27B-GGUF
podman pull ghcr.io/gufo-org/toolboxes/gufo-runtime:latest
podman run --rm \
  --userns=keep-id:uid=1000,gid=1000 \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add keep-groups \
  --ulimit memlock=-1 \
  -p 8080:8080 \
  -v ./models:/models:ro \
  ghcr.io/gufo-org/toolboxes/gufo-runtime:latest \
  gufo serve --host 0.0.0.0 --port 8080 llm \
  --model /models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf
```

Then, from another terminal, ask it something through the OpenAI-compatible API:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B-UD-Q8_K_XL",
    "messages": [{"role": "user", "content": "Say something"}]
  }'
```

The server also exposes `/v1/completions`, `/v1/responses`, `/v1/models`, and
`/health`. Any OpenAI-compatible client can point at `http://localhost:8080`.
For DFlash2 speculative decoding (about 2.9x faster decode) download the
DFlash2 companion as described in [MODELS.md](./docs/MODELS.md).

## Manifest/Philosophy

- The project is vertical on the AMD Strix Halo 128 GiB; our goal is solely to optimize it. This enables optimizations that otherwise wouldn't be possible if we were focusing on other chips as well. Smaller models should fit the 32 and 64 GiB hardware, but no test was conducted on them.
- Quality over speed: we want to squeeze the most out of this chip without compromising on quality compared to other available tools (llama.cpp, audio.cpp, dwarfstar, etc.). To guarantee this we ensure several steps during the development, such as logits checks, internal eval, and a harness + model evaluation framework (coming soon). If at some point a breakthrough novelty brings a lot of speed at the cost of a little accuracy, the feature would be opt-in and the user will be responsible for enabling it, acknowledging the accuracy degradation.
- We only support a few models to allow us to run extremely long optimization sessions to improve kernels based on the Strix Halo architecture. Models are selected based on evidence collected by the community on "the best model for task X for Strix Halo".
- Code duplication over code re-utilization across models and quants. Despite being counterintuitive, it allows us to make models evolve independently without huge refactors when an optimization works only for a model and not for another.
- We would like this project to be the reference for the community using Strix Halo, and every PR is welcome.

## Benchmarks

### Text

#### Qwen3.8-27B

Comparison of gufo's native Qwen3.8-27B path against `llama.cpp` on the same hardware, the same model weights (Q8 with DFlash2 Q8), and the same numeric precision.

| Depth | Gufo `pp2048` | llama.cpp `pp2048` | Gufo / llama.cpp |
| ----: | ------------: | -----------------: | ---------------: |
|     0 |        545.15 |             352.80 |        **1.55x** |
|    4K |        524.11 |             331.70 |        **1.58x** |
|    8K |        498.94 |             309.75 |        **1.61x** |
|   16K |        446.94 |             270.00 |        **1.66x** |

With the DFlash2 draft head

| Policy               |     Speculative |   Speedup |    Median | Acceptance | Average draft | Accepted per step |
| -------------------- | --------------: | --------: | --------: | ---------: | ------------: | ----------------: |
| **fixed 7** (`auto`) | **20.40 tok/s** | **2.91x** | **3.08x** |      45.3% |          7.00 |          **4.17** |
| rolling 1-7          |     19.52 tok/s |     2.79x |     2.86x |      54.7% |          5.11 |              3.79 |
| rolling 3-7          |     19.52 tok/s |     2.79x |     2.86x |      54.7% |          5.11 |              3.79 |
| accepted-EMA 3-7     |     18.72 tok/s |     2.67x |     2.71x |      64.8% |          4.04 |              3.62 |

#### DeepSeek V4 Flash

Comparison of gufo's native DeepSeek V4 Flash path against `dwarfstar` on the same hardware, the same model weights, and the same numeric precision.

| Prepared depth |  Gufo `pp2048` | DS4 `pp2048` | Delta | Gufo `tg128` | DS4 `tg128` | Delta | Snapshot bytes |
| -------------: | -------------: | -----------: | ----: | -----------: | ----------: | ----: | -------------: |
|             2K |         206.33 |       205.49 | +0.4% |        15.63 |       14.76 | +5.9% |     52,184,460 |
|             8K |         204.22 |       197.13 | +3.6% |        14.65 |       13.87 | +5.6% |    136,750,476 |
|            16K |         197.79 |       190.09 | +4.1% |        14.39 |       13.63 | +5.6% |    249,505,164 |
|            32K |         179.18 |       171.83 | +4.3% |        13.66 |       12.93 | +5.6% |    475,014,540 |
|            64K | not remeasured | not supplied |     - |        12.42 |       11.91 | +4.3% |    926,033,292 |

### Audio

**Date: 2026-09-10**

Comparison of gufo's native Qwen3-TTS and Qwen3-ASR paths against [audio.cpp](https://github.com/0xShug0/audio.cpp) on the same hardware, the same model weights (safetensors), and the same numeric precision.

| task | gufo RTF  | audio.cpp RTF | gufo faster |
| ---- | --------- | ------------- | ----------- |
| tts  | **0.472** | **0.663**     | **1.40x**   |
| asr  | **0.075** | **0.121**     | **1.62x**   |

For more information please see [TTS_ASR.md](./docs/benchmarks/TTS_ASR.md).

## Supported Platform

Linux x86-64 on AMD Strix Halo (`gfx1151` GPU and XDNA2 NPU) is the only
planned production platform. Windows, macOS, and CUDA are out of scope.

Build with Nix only; direct host builds are unsupported:

```sh
nix build                          # build default package (gfx1151 + XRT)
./result/bin/gufo diagnose        # run hardware probe & diagnostics
./result/bin/gufo serve           # run server
nix build .#checks.x86_64-linux.pr # canonical PR test command (all gates)
```

For non-Nix users, [toolboxes](https://github.com/gufo-org/toolboxes) for Docker and Podman are available:

```sh
podman pull ghcr.io/gufo-org/toolboxes/gufo-runtime:latest
```

### Why Nix?

Nix provides a reproducible environment by having all the dependencies in a single file, the [flake.nix](./flake.nix). Our primary goal is to provide an engine that performs better on the Strix Halo without sacrificing accuracy. Nix helps us develop this project by removing all those variables that may make experiments not reproducible across several machines (different driver versions, different environment variables, etc.).

## Reference Projects

The initial design is informed by the following open source projects:

- `llama.cpp` for compact model serving, GGUF, and CPU/GPU correctness paths.
- `vLLM` for continuous batching and paged request scheduling.
- `hipEngine` for torch-free HIP execution, and native speculative-cycle work.
- `ROCmFPX` for activation-aware quantization and quality evaluation.
- `DS4` for DeepSeek V4 Flash, MoE scheduling, and DSpark.
- `ypapadop-amd/ggml` `hsa-backend` for XDNA2 HSA dispatch and MLIR-AIE
  integration patterns.
- `audio.cpp` for audio models for tts and asr tasks.
