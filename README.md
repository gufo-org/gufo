# Gufo: the Strix Halo inference engine

<p align="center">
  <img src="assets/gufo-logo.jpg" alt="Gufo logo" width="180">
</p>

Gufo is a vertical local inference engine specifically built and optimized for the AMD Strix Halo hardware:
Ryzen AI MAX+ 395 systems with Radeon 8060S (`gfx1151`), up to 128 GiB of unified memory.

## Models and benchmarks

All model documentation lives under [docs/models](docs/models/README.md):

| Model | Inference modes | Performance and quality |
| --- | --- | --- |
| [Qwen3.8 27B](docs/models/qwen3.8-27b/README.md) | Q4/Q8, images, AR, DFlash2, native MTP | [Benchmarks](docs/models/qwen3.8-27b/BENCHMARKS.md) · [Evaluation](docs/models/qwen3.8-27b/EVALUATION.md) |
| [Qwen3.8 Flash-Next](docs/models/qwen3.8-flash-next/README.md) | Images, AR, MTP | [Benchmarks](docs/models/qwen3.8-flash-next/BENCHMARKS.md) · [Evaluation](docs/models/qwen3.8-flash-next/EVALUATION.md) |
| [DeepSeek V4 Flash](docs/models/deepseek-v4-flash/README.md) | AR, DSpark | [Benchmarks](docs/models/deepseek-v4-flash/BENCHMARKS.md) · [Evaluation](docs/models/deepseek-v4-flash/EVALUATION.md) |
| [Qwen3-ASR](docs/models/qwen3-asr/README.md) | Speech recognition | [Benchmarks](docs/models/qwen3-asr/BENCHMARKS.md) · [Evaluation](docs/models/qwen3-asr/EVALUATION.md) |
| [Qwen3-TTS](docs/models/qwen3-tts/README.md) | CustomVoice, VoiceDesign, Base cloning | [Benchmarks](docs/models/qwen3-tts/BENCHMARKS.md) · [Evaluation](docs/models/qwen3-tts/EVALUATION.md) |
| [Qwen-Image-2.1](docs/models/qwen-image-2.1/README.md) | BF16 image generation and editing | [Benchmarks](docs/models/qwen-image-2.1/BENCHMARKS.md) · [Evaluation](docs/models/qwen-image-2.1/EVALUATION.md) |
| [MiniMax H3](docs/models/minimax-h3/README.md) | Text to video/audio, exact and approximate presets | [Benchmarks](docs/models/minimax-h3/BENCHMARKS.md) · [Evaluation](docs/models/minimax-h3/EVALUATION.md) |

## Manifest/Philosophy

- The project is vertical on the AMD Strix Halo 128 GiB; our goal is solely to optimize it. This enables optimizations that otherwise wouldn't be possible if we were focusing on other chips as well. Smaller models should fit the 32 and 64 GiB hardware, but no test was conducted on them.
- Quality over speed: we want to squeeze the most out of this chip without compromising on quality compared to other available tools (llama.cpp, audio.cpp, dwarfstar, etc.). To guarantee this we ensure several steps during the development, such as logits checks, internal eval, and a harness + model evaluation framework (coming soon). If at some point a breakthrough novelty brings a lot of speed at the cost of a little accuracy, the feature would be opt-in and the user will be responsible for enabling it, acknowledging the accuracy degradation.
- We only support a few models to allow us to run extremely long optimization sessions to improve kernels based on the Strix Halo architecture. Models are selected based on evidence collected by the community on "the best model for task X for Strix Halo".
- Code duplication over code re-utilization across models and quants. Despite being counterintuitive, it allows us to make models evolve independently without huge refactors when an optimization works only for a model and not for another.
- We would like this project to be the reference for the community using Strix Halo, and every PR is welcome.
- We don't to sacrificate multi-agent scenarios, concurrent requests are a first class citizen gufo.

## Quickstart

```sh
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --repo-type model \
  --local-dir models/Qwen3.8-27B-GGUF
nix develop -c hf download z-lab/Qwen3.8-27B-DFlash2-GGUF \
  Qwen3.8-27B-DFlash2-Q8_0.gguf \
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
  --dflash-model /models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q8_0.gguf
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

## Build from source

Linux x86-64 on AMD Strix Halo (`gfx1151`) is the supported target. CMake owns
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
requirements. See [testing](docs/TESTING.md) for the small hosted CI suite and
explicit local quality checks.

### Without Nix

Install a C++20 compiler, CMake 3.21+, Ninja, pkg-config, Python 3.10+ and the
following development libraries. The currently qualified toolchain is GCC
15.3, ROCm 7.2.3, AOTriton 0.11.1b and Triton 3.7.0.

| Dependency | Used for |
| --- | --- |
| ROCm HIP compiler/runtime, hipBLAS, hipBLASLt, rocBLAS | GPU execution and matrix multiplication |
| hipCUB, rocPRIM, rocWMMA, Composable Kernel headers | Compiled GPU kernels |
| MIOpen | ASR audio encoder convolutions |
| AOTriton, including **gfx1151 kernel images** | H3 attention |
| ICU, libcurl, OpenSSL, libpng, libjpeg | Tokenization, HTTPS, hashing and images |
| FFmpeg and ffprobe | Video/audio output; invoked as separate executables |
| Python Triton 3.7.0 | Build-time H3 kernel compilation; not needed to run Gufo |

Install ROCm using [AMD's Linux instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).
Use the development packages for the libraries above. AOTriton's Python package
alone is insufficient: CMake needs its headers, `aotriton-config.cmake`, shared
library and gfx1151 images. If building it from
[the upstream source](https://github.com/ROCm/aotriton/tree/0.11.1b), select
`-DAOTRITON_TARGET_ARCH=gfx1151 -DAOTRITON_USE_TORCH=OFF` and follow its install
instructions. ROCm normally installs under `/opt/rocm`.

For example, on Debian/Ubuntu the ordinary system libraries are:

```sh
sudo apt install build-essential cmake ninja-build pkg-config python3-venv \
  libicu-dev libcurl4-openssl-dev libssl-dev libpng-dev libjpeg-dev ffmpeg
python3 -m venv .venv
. .venv/bin/activate
python -m pip install triton==3.7.0

cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
./build/release/gufo diagnose
./build/release/gufo serve llm --model /path/to/model.gguf
```

Install the ROCm/AOTriton dependencies from the table before configuring.
For nonstandard installations, pass ordinary CMake paths, for example
`cmake --preset release -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/aotriton"`.
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

- `llama.cpp` for compact model serving, GGUF, and CPU/GPU correctness paths.
- `vLLM` for continuous batching and paged request scheduling.
- `hipEngine` for torch-free HIP execution, and native speculative-cycle work.
- `DS4` for DeepSeek V4 Flash, MoE scheduling, and DSpark.
- `audio.cpp` for audio models for tts and asr tasks.
