# Loom packages

Gufo serves every text package of the
[loom-fun](https://huggingface.co/loom-fun) Hugging Face organization on its
own [AMD Loom](https://github.com/ROCm/hrx-system) kernels, behind Gufo's
usual APIs, scheduler, prompt cache and parsers.

## Supported models

Any loom-fun text package works, with no Gufo update: browse them at
[huggingface.co/loom-fun](https://huggingface.co/loom-fun). As of November
2026, 51 packages pass Gufo's API checks on Strix Halo, among them Qwen2.5,
Qwen3, Qwen3.5, Qwen3.6 and Qwen3.8 (text and vision), gemma 3 and 4,
gpt-oss 20B, Llama 3.2, LFM and Nemotron; see the
[quality report](../internal/models/loom/QUALITY.md). Image generation
packages aren't supported.

## Run

Install a package by its repository and serve it under the same name:

```sh
gufo model install loom-fun/unsloth-Qwen3.5-2B-loom
gufo serve llm --model loom-fun/unsloth-Qwen3.5-2B-loom
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.5-2B","messages":[{"role":"user","content":"Hello!"}]}'
```

`gufo model install` fetches the package revision built for Gufo's Loom SDK
and the GGUF files its card names, and checks each file's SHA-256. `gufo model
update`, `remove`, `verify` and `--model` take the same repository. Requests
name the model by its card ID (`/v1/models` lists it).

To use files you downloaded yourself, put the package and each GGUF origin's
files in one directory at their repository paths and pass it to `--model`:

```sh
hf download loom-fun/ggml-org-Qwen3-4B-loom --local-dir models/qwen3-4b-loom
hf download ggml-org/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --local-dir models/qwen3-4b-loom
gufo serve llm --model models/qwen3-4b-loom
```

## Options

Loom packages take Gufo's own options:

- `--sessions N` and `--context N` size the KV pool, which holds N full
  contexts (the package's own context by default) and is committed at load.
  The startup log reports the device memory and the tokens the pool holds.
- `--mmproj`, and the draft model option with `--speculative`, replace the
  package's vision projector or draft model.
- `--cache-disk DIR` keeps the prompt cache across restarts.

The package's preset for your GPU is chosen automatically (`loom: preset chat
for gfx1151` in the log).

## Platforms

Every install method includes Loom, on Linux and Windows. Loom drives the GPU
through its own driver and needs no ROCm.

- A build from source compiles the Loom runtime, which needs CMake 3.26 (4.2
  on Windows), Python 3.12 and network access. Leave it out with
  `-DGUFO_EXTENSIONS=<list>`.
- In a container, add `--security-opt seccomp=unconfined`. Add
  `--ulimit memlock=-1` too, or the weights load through host memory.
- On Windows, turn on Developer Mode, or keep `TMP` on the models' drive.
- Loom and HIP can't share the GPU in one process: serve a HIP model from
  another `gufo serve`.

## Limits

- Requests that need logits (forced or named `tool_choice`, strict schemas,
  `response_format`, `logprobs`, `min_p`, penalties) sample on the host and
  run without drafting, so they decode somewhat slower.
- gpt-oss and Glimmer: automatic tool choice isn't grammar-constrained.
- Greedy outputs can differ from loom-serve at near-ties.

Implementation notes and benchmarks are in the
[design notes](../internal/models/loom/README.md).
