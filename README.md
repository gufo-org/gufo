# Gufo: The ***Open Source*** Strix Halo AI engine 

<p align="center">
  <img src="assets/gufo-logo.jpg" alt="Gufo logo" width="180">
</p>

Gufo is a vertical local inference engine for Linux and Windows, specifically built and optimized for AMD Strix Halo hardware:
Ryzen AI MAX+ 395 systems with Radeon 8060S (`gfx1151`), up to 128 GiB of unified memory.

**Contributions are welcome!**

See the [changelog](CHANGELOG.md) and [GitHub Releases](https://github.com/gufo-org/gufo/releases)
for user-facing changes and release history.

> [!TIP]
> Check out **[Gufo RDMA for Dual Strix Halo](https://github.com/neuhaus/gufo)**, a Gufo variant that has not been merged yet!

## Contents

- [Models and benchmarks](#models-and-benchmarks)
  - [Embeddings](#embeddings)
  - [Image generation](#image-generation)
  - [Speech recognition](#speech-recognition)
  - [Speech synthesis](#speech-synthesis)
  - [Text generation](#text-generation)
  - [Video generation](#video-generation)
- [Philosophy](#philosophy)
- [Quickstart](#quickstart)
- [Harness notes](#harness-notes)
- [Build from source](#build-from-source)
- [License](#license)
- [Reference Projects](#reference-projects)

## Models and benchmarks

### Embeddings

| Model | Input | License | Latency (one input) | Throughput |
| --- | --- | --- | ---: | ---: |
| [EmbeddingGemma 2](docs/models/embeddinggemma-2/README.md) | Audio | [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) | **100 ms** (10 s of speech, 254 tokens) | TODO |
| [EmbeddingGemma 2](docs/models/embeddinggemma-2/README.md) | Images | [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) | **584 ms** (320×180 image, 269 tokens) | **461 – 479** tokens/s |
| [EmbeddingGemma 2](docs/models/embeddinggemma-2/README.md) | Text | [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) | **13.0 ms** (404 tokens) | **30.4k – 41.5k** tokens/s |
| [EmbeddingGemma 2](docs/models/embeddinggemma-2/README.md) | Video | [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) | TODO | TODO |

Throughput runs from one request at a time to concurrent clients sharing device
batches. Text uses 128-word inputs.

### Image generation

| Model | Modes | License | Speed |
| --- | --- | --- | --- |
| [Qwen-Image-2.1](docs/models/qwen-image-2.1/README.md) | Generation, editing | [Qwen Research (non-commercial)](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/b3179ad355be050328e483a9dfdd9e60cd62adfa/LICENSE) | In progress |

### Speech recognition

| Model | License | Speed (excludes loading) |
| --- | --- | ---: |
| [Qwen3-ASR 1.7B](docs/models/qwen3-asr/README.md) | [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) | **15.27×** realtime |

### Speech synthesis

| Model | Modes | License | Speed (excludes loading) |
| --- | --- | --- | ---: |
| [Qwen3-TTS 1.7B](docs/models/qwen3-tts/README.md) | CustomVoice, VoiceDesign, voice cloning (Base) | [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) | Up to **2.54×** realtime; **201 ms** to first audio (CustomVoice) |

### Text generation

| Model | Weights | Inputs | License | Prompt processing | Single-user decode | 8-user decode |
| --- | --- | --- | --- | ---: | ---: | ---: |
| [DeepSeek V4 Flash](docs/models/deepseek-v4-flash/README.md) | IQ2XXS | Text | [MIT](https://spdx.org/licenses/MIT.html) | **484.62** tok/s | **16.28 – 26.62** tok/s | **55.47** tok/s |
| [Qwen3.8 27B](docs/models/qwen3.8-27b/README.md) | UD-Q4_K_XL | Text, images | [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) | **648.89** tok/s | **12.42 – 71.55** tok/s | **74.95 – 128.42** tok/s |
| [Qwen3.8 27B](docs/models/qwen3.8-27b/README.md) | UD-Q8_K_XL | Text, images | [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) | **479.25** tok/s | **7.21 – 51.69** tok/s | **51.63 – 123.47** tok/s |
| [Qwen3.8 Flash-Next](docs/models/qwen3.8-flash-next/README.md) | UD-Q4_K_XL | Text, images | [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE) | **1,801.01** tok/s | **28.29 – 116.93** tok/s | **125.48 – 245.81** tok/s |

Prompt processing is autoregressive (AR). Decode ranges run from AR to the
repetitive-text speculative run, the best case; mixed prose falls between the
two, as each Benchmarks page shows. 8-user decode sums individual request rates.
The 27B single-user peaks are the 1-user rows of its multi-user table.
DeepSeek's 8-user cell is the AR rate because speculation is no faster there.

### Video generation

| Model | Modes | License | Speed |
| --- | --- | --- | --- |
| [MiniMax H3](docs/models/minimax-h3/README.md) | Text to video and audio | [MiniMax H3 Community (not for the US, EU, UK or Korea)](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE) | In progress |

## Philosophy

- Contributions are welcome! We need the help of Strix Halo community to keep improving gufo!
- We would like this to be the one-stop shop for Strix Halo Local AI enthusiasts: batteries included for text, audio, image, and video models.
- Build and optimize specifically for the Strix Halo 128 GiB hardware. Smaller memory configurations should still work and preserve the speed benefits for models that can fit on memory.
- Support only the best available models for their size that can run on this hardware: less code to maintain, more focused optimization and testing work.
- Preserve quality when optimizing. Each model's quality report records independent numerical checks, execution consistency and unresolved gaps. Don't reuse kernels across different models to limit blast radius of a code change.
- Treat concurrent requests, cancellation and conversation caching as first-class workloads.
- Keep production dependencies small and development tools separate.

## Quickstart

```sh
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 \
  --repo-type model \
  --local-dir models/Qwen3.8-27B-GGUF
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF \
  Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --revision 2d9571f8ce46e151f61c6499c99dee6079e1d610 \
  --repo-type model \
  --local-dir models/Qwen3.8-27B-DFlash2-GGUF
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
  --model /models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --speculative dflash2 \
  --dflash-model /models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

Rootless Podman needs `crun` for `--group-add keep-groups`. Your host user must
have read/write access to `/dev/kfd` and `/dev/dri/renderD*`, usually through the
`render` and `video` groups; log out and back in after changing membership.
Container groups named `video`/`render` do not preserve host supplementary
groups. Check `id`, `ls -l /dev/kfd /dev/dri/renderD*`, and
`podman info --format '{{.Host.OCIRuntime.Name}}'` if ROCm reports no device.
See [Podman's rootless group-access guidance](https://github.com/containers/podman/blob/main/troubleshooting.md#20-passed-in-devices-or-files-cant-be-accessed-in-rootless-container).

On Fedora or another SELinux-enforcing host, GPU enumeration can succeed while
SELinux blocks mapping `/dev/kfd`, causing ROCr to report a misleading
“Memory critical” error. Check the **host** audit log:

```sh
sudo ausearch -m avc -ts recent | grep -E '/dev/kfd|hsa_device_t'
```

If it shows a denied `map` for the container, Podman documents this fix:

```sh
sudo setsebool -P container_use_devices true
```

This persistently allows containers to access device labels for devices passed
into them; it affects all containers on that host. Review that policy scope
before enabling it. See [Podman's device documentation](https://docs.podman.io/en/latest/markdown/podman-run.1.html#device-host-device-container-device-permissions)
and the [SELinux container policy](https://github.com/containers/container-selinux/blob/main/container.te).

Then, from another terminal, ask it something through the OpenAI-compatible API:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B-UD-Q8_K_XL",
    "messages": [{"role": "user", "content": "Say something"}]
  }'
```

For Open WebUI, VS Code, OpenAI SDK and Responses-API clients (for example
Codex), set the API base URL to `http://localhost:8080/v1`. Chat Completions and
Responses support text, images, function tools, structured output and streaming.
See the [API contract](docs/SERVER.md).

The text server uses the model's native context by default and generates until
EOS or the context is full. `--context N` sets context capacity per session;
`--max-tokens N` sets a default response limit that clients can override.
Reasoning tokens count toward that response limit.

## Harness notes

- **Qwen with Codex:** Codex sends developer messages mid-conversation, after
  compaction or a settings change. Qwen's template accepts only one leading
  system turn, so Gufo moves them there. The request that introduces one is
  prefilled again; later requests reuse the cache. See the
  [API contract](docs/SERVER.md).
- **Qwen with Claude Code:** not supported. Claude Code adds a system message
  to `messages` on every turn, so Gufo moves a new one to the leading system
  turn each time and every request prefills the whole history again (#522).

## Build from source

Production packages target Linux x86-64 on AMD Strix Halo (`gfx1151`).
Windows x86-64 is a supported native build target for development; see
[BUILD-WINDOWS.md](BUILD-WINDOWS.md). CMake owns
one production configuration for both Nix and ordinary Linux builds. Tests,
profilers, tuning executables and Python reference runners are not installed
with the production package. No `build.sh` wrapper is needed.

### With Nix

```sh
nix build
./result/bin/gufo diagnose
./result/bin/gufo serve llm --model /path/to/model.gguf
```

[flake.lock](flake.lock) pins the dependencies. `nix develop` adds profiling,
model-download and independent evaluation tools; these are not runtime
requirements. Optional benchmark baselines are selected separately with
`nix shell .#ds4-reference`, `.#llama-cpp-reference` or
`.#llama-cpp-mtp-reference`; see [benchmarking](docs/BENCHMARKS.md).
See [testing](docs/TESTING.md) for the small hosted CI suite and
explicit local quality checks.

Prebuilt store paths are published to the `gufo` Cachix cache. The flake
declares it in `nixConfig`: users listed in `trusted-users` are asked to accept
it, or pass `--accept-flake-config`. Nix ignores it for other users; add it
system-wide with `cachix use gufo` or in `/etc/nix/nix.conf`:

```
extra-substituters = https://gufo.cachix.org
extra-trusted-public-keys = gufo.cachix.org-1:rfZe3n4lHINgyoACKzs5EthIZmBNvEyCCtufUpth0PE=
```

The Nix package and development shell use ROCm 10.1.0 (HIP Clang 24)
from the pinned [nixpkgs ROCm 10.1 update](https://github.com/NixOS/nixpkgs/pull/573885).
Python, Python packages and general build dependencies come from the pinned
`nixos-unstable` input. The ROCm package set comes from the separate fork,
retaining its compiler/runtime dependencies to reuse the gfx1151 binaries on
`gufo.cachix.org`. All Nix workflows configure this cache explicitly.
Both include a [queue-memory patch](.devops/nix/patches/rocr-fixed-cwsr-rocm10.patch)
for Strix Halo. It keeps queue save/restore buffers, launch/copy arenas and
code-loading staging resident during host memory compaction, while retaining
managed-memory support and the ordinary fine-grained host-allocation policy.

### Without Nix

Install a C++20 compiler, CMake 3.21+, Ninja, pkg-config and the
following development libraries. The pinned Nix toolchain uses GCC
16.2 and ROCm 10.1.0 (HIP Clang 24). Attention and audio convolution kernels are compiled
directly from HIP. Python, Triton/AOTriton, Composable Kernel and MIOpen are
not production build or runtime requirements.
For the same protection against queue evictions as the Nix build, apply the
ROCr patch above when building your runtime. Flash-Next's host buffers also use
Linux long-term pins, with ordinary HIP registration retained when system
limits disallow those pins. Gufo leaves `HSA_USERPTR_FOR_PAGED_MEM` unchanged.

| Dependency | Used for |
| --- | --- |
| ROCm HIP compiler/runtime, hipBLAS, hipBLASLt, rocBLAS | GPU execution and matrix multiplication |
| hipCUB, rocPRIM, rocWMMA headers | Compiled GPU kernels |
| ICU, libcurl, OpenSSL, libpng, libjpeg, libwebp | Tokenization, HTTPS, hashing and images |
| FFmpeg development libraries | Video embedding decoding and RGB conversion |
| libsndfile, libsoxr | Audio decoding and reference-compatible resampling |
| FFmpeg and ffprobe | Video/audio output; invoked as separate executables |

Install ROCm using [AMD's Linux instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).
Use the development packages for the libraries above. ROCm normally installs
under `/opt/rocm`.

For example, on Debian/Ubuntu the ordinary system libraries are:

```sh
sudo apt install build-essential cmake ninja-build pkg-config \
  libicu-dev libcurl4-openssl-dev libssl-dev libpng-dev libjpeg-dev libwebp-dev libsndfile1-dev libsoxr-dev libavformat-dev libavcodec-dev libavutil-dev libswscale-dev ffmpeg

# ROCm libraries from the table, named as AMD's repository ships them.
sudo apt install hipblas-dev hipblaslt-dev rocblas-dev \
  hipcub-dev rocprim-dev rocwmma-dev

cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
./build/release/gufo diagnose
./build/release/gufo serve llm --model /path/to/model.gguf
```

Configuring fails at `find_package(hipblas)` when those ROCm packages are
missing. Other distributions name them `-devel` instead of `-dev`.
For nonstandard installations, pass ordinary CMake paths, for example
`cmake --preset release -DCMAKE_PREFIX_PATH="/opt/rocm"`.
If compiler discovery picks a system Clang, also pass
`-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++`.
Use `cmake --install build/release` to install Gufo,
its runtime data and license notices. The GPU driver must allow your user to
access `/dev/kfd` and `/dev/dri`; model weights are acquired separately.

The same source, compiler flags and install rules serve both builds. Nix pins
the complete toolchain for reproducible comparisons; changing the compiler or
math libraries requires the affected model's quality checks.

## License

Gufo's original code is [MIT licensed](LICENSE). Adapted code and dependencies
retain their own notices in [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and `licenses/`, installed under `share/licenses/gufo`. Model weights are not
bundled and retain their publishers' terms.

## Reference Projects

The initial design is informed by the following open source projects:

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for compact model serving, GGUF, and CPU/GPU correctness paths.
- [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp), [Nathanw1014/strix-halo-llamacpp](https://github.com/Nathanw1014/strix-halo-llamacpp), and [gaetan-puleo/llama-cpp-strix-halo](https://github.com/gaetan-puleo/llama-cpp-strix-halo) for Strix Halo optimization inspiration.
- [vLLM](https://github.com/vllm-project/vllm) for continuous batching and paged request scheduling.
- [hipEngine](https://github.com/shisa-ai/hipEngine) for torch-free HIP execution, and native speculative-cycle work.
- [ds4](https://github.com/antirez/ds4) for DeepSeek V4 Flash, MoE scheduling, and DSpark.
- [audio.cpp](https://github.com/0xShug0/audio.cpp) for audio models for tts and asr tasks.
