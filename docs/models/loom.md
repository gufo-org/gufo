# Loom packages

Gufo serves [loom-fun](https://huggingface.co/loom-fun) packages: open GGUF
models, from Qwen3 0.6B to Qwen3.6 35B-A3B, gemma 4, gpt-oss 20B and Llama
3.2, on the package's own [AMD Loom](https://github.com/ROCm/hrx-system)
kernels, behind Gufo's usual APIs, scheduler, prompt cache and parsers.

A **package** is a loom-fun repository: a model card (`model.json`), its
Loom programs and tokenizer, for an unmodified GGUF checkpoint that it names
by origin (`@checkpoint/<file>`), plus a drafter or a vision projector where
it uses one.

## Files

`gufo model install` downloads a package and the GGUF files its card names,
pinned by revision, size and SHA-256, into one model directory:

```sh
gufo model list                       # the *-loom IDs
gufo model install qwen3-4b-loom      # 3.1 GB
```

Files you download yourself work too. Pass the package directory to
`--model` and each origin's files with `--loom-checkpoint`, `--loom-drafter`
or `--loom-vision` (a file, or a directory holding them at their repository
paths):

```sh
hf download loom-fun/ggml-org-Qwen3-4B-loom --local-dir models/qwen3-4b-loom
hf download ggml-org/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --local-dir models/qwen3-4b-gguf
gufo serve llm --model models/qwen3-4b-loom --loom-checkpoint models/qwen3-4b-gguf
```

Without these options Gufo also looks for the files in the package
directory, at their origin paths (`hf download` both repositories into one
directory), and a package fetched with `loom-fetch`, which places them itself,
needs none. Use the package revision `models.lock.json` pins: a package built
for another Loom SDK release is refused at load with the revision to fetch.

## Run

```sh
gufo serve llm --model qwen3-4b-loom
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"Hello!"}]}'
```

`gufo prompt`, `gufo chat` and `gufo bench` take the same `--model`. The
package's preset for your GPU is chosen automatically (`loom: preset chat for
gfx1151` in the log); `--loom-preset` picks another.

| Option | Use |
| --- | --- |
| `--loom-preset NAME` | Preset of a package with several |
| `--loom-logits auto\|on\|off` | Logits readback for grammars, `logprobs`, `min_p` and penalties (default `auto`: on when the package declares it) |
| `--loom-memory-bytes N` | Elastic memory pool with that budget, instead of a fixed pool committed at load |
| `--loom-device URI` | Loom device URI |

`--cache-disk DIR` keeps the prompt cache across restarts; its checkpoint
images need the elastic pool, which Gufo then uses.

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
- On Windows, a package with several presets serves `chat` (Strix Halo)
  unless `--loom-preset` names another. Turn on Developer Mode, or keep the
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

As of November 2026, on Strix Halo, every declared package passed Gufo's API
conformance checks through `gufo model install`: 44 of 51 pass all 29, the
rest miss checks for model behavior. Loom decodes faster than llama.cpp ROCm
on 45 of the 50 measured packages. See the
[quality report](../internal/models/loom/QUALITY.md) and
[design notes](../internal/models/loom/README.md).
