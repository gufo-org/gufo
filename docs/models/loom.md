# Loom packages

Gufo serves [loom-fun](https://huggingface.co/loom-fun) packages: open GGUF
models, from Qwen3 0.6B to Qwen3.6 35B-A3B, gemma 4, gpt-oss 20B and Llama
3.2, on the package's own [AMD Loom](https://github.com/ROCm/hrx-system)
kernels, behind Gufo's usual APIs, scheduler, prompt cache and parsers.

A **package** is a loom-fun repository: a model card (`model.json`), its
Loom programs and tokenizer, for an unmodified GGUF checkpoint that it names
by origin (`@checkpoint/<file>`), plus a drafter or a vision projector where
it uses one.

## Run

Install a package by its repository and serve it under the same name:

```sh
gufo model install loom-fun/unsloth-Qwen3.5-2B-loom
gufo serve llm --model loom-fun/unsloth-Qwen3.5-2B-loom
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.5-2B","messages":[{"role":"user","content":"Hello!"}]}'
```

Any loom-fun text package works, with no Gufo update. `gufo model install`
takes the package revision built for Gufo's Loom SDK (the `v<SDK>` tag, else
the default branch) and the GGUF files its card names, at the revisions the
card pins, and checks each file's SHA-256 from Hugging Face. `gufo model
update`, `remove` and `verify`, and `--model` in `serve`, `prompt`, `chat`
and `bench`, take the same repository. Requests name the model by its card
ID (`Qwen/Qwen3.5-2B` here; `/v1/models` lists it), unless
`--served-model-name` sets another.

Files you download yourself work too: download the package and each origin's
files into one directory, at their repository paths (the layout `gufo model
install` writes), and pass it to `--model`:

```sh
hf download loom-fun/ggml-org-Qwen3-4B-loom --local-dir models/qwen3-4b-loom
hf download ggml-org/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --local-dir models/qwen3-4b-loom
gufo serve llm --model models/qwen3-4b-loom
```

A package fetched with
`loom-fetch`, which places the files at their card paths, works as it is.
`--mmproj` and the draft model option (`--mtp-model`, `--dflash-model` or
`--dspark-model` with its `--speculative`) replace the package's vision or
draft file. A package built for another Loom SDK release is refused at load
with the revision to fetch.

The package's preset for your GPU is chosen automatically, on Linux and
Windows (`loom: preset chat for gfx1151` in the log). A GPU the package has
no preset for serves the gfx1151 preset `chat`, with a warning. Grammars,
`logprobs`, `min_p` and penalties use the package's logits whenever it
declares them.

Loom commits the KV of `--sessions` full contexts at load (`--context`, the
package's own by default), shared by all sessions; the startup log reports
the device memory and the tokens the pool holds. `--loom-device URI` selects
another Loom device. `--cache-disk DIR` keeps the prompt cache across
restarts; its checkpoint images back the pool elastically instead.

## Platforms

Every install method includes Loom: the container, the Debian package, Nix,
a build from source and the Windows ZIP. Loom drives the GPU through its own
driver and needs no ROCm.

- A build from source compiles the Loom runtime from its pinned release,
  which needs CMake 3.26 (4.2 on Windows), Python 3.12 and network access at
  build time. Leave it out with `-DGUFO_EXTENSIONS=<list>`.
- In a container, add `--security-opt seccomp=unconfined`: Loom's GPU memory
  reservation fails under the default profile.
- Loom streams weights through pinned buffers, which need a locked-memory
  limit of at least 64 MiB (`--ulimit memlock=-1`). Below it, the weights are
  read into host memory at load.
- On Windows, turn on Developer Mode, or keep the
  temporary folder (`TMP`) on the models' drive, so Gufo can link the
  package's files.
- A Loom package runs in its own process: on Linux, Loom and HIP can't share
  the GPU in one process. Serve a HIP model from another `gufo serve`.

## Limits

- Requests that need logits (forced or named `tool_choice`, strict schemas,
  `response_format`, `logprobs`, `min_p`, penalties) sample on the host and
  run without drafting, so they decode somewhat slower.
- gpt-oss and Glimmer: automatic tool choice isn't grammar-constrained.
- Greedy outputs can differ from loom-serve at near-ties: Loom's results
  depend on batch shape.

## Quality and performance

As of November 2026, on Strix Halo, 51 loom-fun packages passed Gufo's API
conformance checks through `gufo model install`: 44 of 51 pass all 29, the
rest miss checks for model behavior. Loom decodes faster than llama.cpp ROCm
on 45 of the 50 measured packages. See the
[quality report](../internal/models/loom/QUALITY.md) and
[design notes](../internal/models/loom/README.md).
