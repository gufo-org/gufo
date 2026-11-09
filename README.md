# Gufo: The ***Open Source*** Strix Halo AI engine

<p align="center">
  <img src="assets/gufo-logo.jpg" alt="Gufo logo" width="180">
</p>

<p align="center">
  <a href="https://github.com/gufo-org/gufo/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/gufo-org/gufo?label=release&amp;logo=github&amp;style=flat&amp;labelColor=141714&amp;color=2f5d3a"></a>
  <a href="https://gufo.blog/docs"><img alt="Documentation" src="https://img.shields.io/badge/docs-gufo.blog-2f5d3a?logo=readthedocs&amp;logoColor=white&amp;style=flat&amp;labelColor=141714&amp;color=2f5d3a"></a>
  <a href="docs/extension-api.md#versions"><img alt="Extension API series" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgufo-org%2Fgufo%2Fmain%2Finclude%2Fgufo%2Fext%2Fapi-series.json&amp;query=%24.minor&amp;prefix=0.&amp;label=extension%20API&amp;logo=cplusplus&amp;style=flat&amp;labelColor=141714&amp;color=2f5d3a"></a>
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/github/license/gufo-org/gufo?logo=opensourceinitiative&amp;logoColor=white&amp;style=flat&amp;labelColor=141714&amp;color=2f5d3a"></a>
</p>

Gufo is an open-source inference engine for one machine: AMD Strix Halo, the
Ryzen AI MAX+ 395 with its Radeon 8060S (`gfx1151`) and up to 128 GiB of
unified memory. It serves text, embedding, speech, image and video models over
OpenAI-compatible APIs, from a single binary with kernels tuned for that chip.
It runs on Linux, where releases are built and tested, and natively on
Windows.

Gufo supports one model per use case, the best that fits, and checks each one
against the original before optimizing it. Its cache is built for coding agents
that resend long conversations, and extensions let you bring your own model.

## Models

| Model | What it does | Weights license |
| --- | --- | --- |
| [Qwen3.8 Flash-Next](docs/models/qwen3.8-flash-next.md) | Text and images | [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) |
| [Qwen3.8 27B](docs/models/qwen3.8-27b.md) | Text and images | Apache 2.0 |
| [DeepSeek V4 Flash](docs/models/deepseek-v4-flash.md) | Text | MIT |
| [EmbeddingGemma 2](docs/models/embeddinggemma-2.md) | Embeddings for text, images, audio and video | Apache 2.0 |
| [Qwen3-ASR 1.7B](docs/models/qwen3-asr.md) | Speech to text | Apache 2.0 |
| [Qwen3-TTS 1.7B](docs/models/qwen3-tts.md) | Text to speech and voice cloning | Apache 2.0 |
| [Qwen-Image-2.1](docs/models/qwen-image-2.1.md) | Image generation and editing | [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE), non-commercial |
| [Qwen-Image-2.1-Turbo](docs/models/qwen-image-2.1.md#qwen-image-21-turbo) | Eight-step image generation and editing | [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE), non-commercial |
| [MiniMax H3](docs/models/minimax-h3.md) | Video with sound | [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) |

Each model page lists the files to download, the modes and the measured speed.
Gufo's own code is MIT; the weights keep their publishers' terms.

Extensions add more models without changing Gufo. With `gufo` set up as in the
[quickstart](#quickstart), install a reviewed one from the
[extension registry](https://github.com/gufo-org/gufo-extension):

```sh
gufo extension install lfm3
```

[Extensions](docs/extensions.md) explains what that runs and how to trust it;
the [catalogue](https://gufo.blog/extensions) lists the official and community
extensions.

## Quickstart

Download Qwen3.8 Flash-Next and its MTP head:

```sh
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision d96201194f103c906994f283f504209dc8d2a10c \
  --include "UD-Q4_K_XL/*" \
  --include "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" \
  --local-dir models/flash-next
```

Define `gufo` to run in the container, with the GPU and the models folder:

```sh
gufo() {
  podman run --rm -it --network host \
    --userns=keep-id:uid=1000,gid=1000 --group-add keep-groups \
    --device /dev/kfd --device /dev/dri --ulimit memlock=-1 \
    -v ./models:/models:ro \
    -v gufo-extensions:/home/gufo/.local/share/gufo/extensions \
    ghcr.io/gufo-org/toolboxes/gufo-runtime:latest gufo "$@"
}
```

Serve the model, then ask it something from another terminal:

```sh
gufo serve llm --served-model-name flash-next \
  --model /models/flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --speculative mtp --mtp-model /models/flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "flash-next", "messages": [{"role": "user", "content": "Say something"}]}'
```

Any OpenAI client works with the base URL `http://localhost:8080/v1`: Open
WebUI, the OpenAI SDKs, Codex and other agents. If the GPU isn't found, see
[troubleshooting](docs/troubleshooting.md#the-gpu-isnt-found). For the Debian
package, Nix, a source build or Windows, see [install](docs/install.md).

## Documentation

The documentation is on [gufo.blog/docs](https://gufo.blog/docs), one version
per release, and in [`docs/`](docs/README.md) for the development version.

- **Get started:** [install](docs/install.md), [models](docs/models/README.md),
  [command line](docs/cli.md), [server API](docs/server.md).
- **Run it well:** [cache](docs/cache.md),
  [troubleshooting](docs/troubleshooting.md), [benchmarks](docs/benchmarks.md),
  [versions](docs/versions.md).
- **Extend it:** [extensions](docs/extensions.md),
  [writing an extension](docs/writing-extensions.md),
  [extension API](docs/extension-api.md).

The [changelog](CHANGELOG.md) and [releases](https://github.com/gufo-org/gufo/releases)
list what changed.

## Principles

- **One machine.** Every kernel is tuned for Strix Halo with 128 GiB. Smaller
  memory configurations work for the models that fit.
- **The best model per use case.** Fewer models means less code to maintain and
  more time to optimize and test each one. Each model keeps its own kernels, so
  a change to one can't break another.
- **Quality before speed.** Every model has a quality report that compares it
  with an independent reference; an optimization that changes the output isn't
  one.
- **Real workloads.** Concurrent requests, cancellation and long conversations
  are first-class, not afterthoughts.
- **Small production footprint.** Few runtime dependencies; development and
  reference tools stay out of the package.

## Contributing

Contributions are welcome: Gufo needs the Strix Halo community to keep
improving. Start with [development](docs/internal/development.md) and
[testing](docs/internal/testing.md); [AGENTS.md](AGENTS.md) holds the rules
that people and coding agents follow. To add a model without changing Gufo,
[write an extension](docs/writing-extensions.md).

Community projects built on Gufo include
[Gufo RDMA for Dual Strix Halo](https://github.com/neuhaus/gufo), which runs
models across two machines and isn't merged yet.

## License

Gufo's original code is [MIT licensed](LICENSE). Its design builds on other
open-source engines, listed in [reference projects](docs/reference-projects.md). Adapted code and dependencies
keep their own notices in [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`, installed
under `share/licenses/gufo`. Model weights aren't bundled and keep their
publishers' terms.
