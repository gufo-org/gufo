# Loom packages

Gufo serves the text models published by
[loom-fun](https://huggingface.co/loom-fun) on Hugging Face, using AMD's
[Loom](https://github.com/ROCm/hrx-system) kernels. They work with the same
APIs as Gufo's other models.

> **Experimental.** Loom support is new, and some things may not work as
> expected yet. For example, requests with tools or JSON schemas currently
> decode without the draft model, so they run slower
> ([details](https://github.com/gufo-org/gufo/issues/1334)). Please
> [report](https://github.com/gufo-org/gufo/issues) anything else you hit.

## Supported models

Any loom-fun text package, including Qwen3, Qwen3.5, Qwen3.8, gemma, gpt-oss,
Llama 3.2 and more. New packages work without a Gufo update. Image
generation packages aren't supported.

## Run

```sh
gufo model install loom-fun/unsloth-Qwen3.5-2B-loom
gufo serve llm --model loom-fun/unsloth-Qwen3.5-2B-loom
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.5-2B","messages":[{"role":"user","content":"Hello!"}]}'
```

In requests, use the model ID that `/v1/models` lists (here
`Qwen/Qwen3.5-2B`). `gufo model update`, `remove` and `verify` take the same
repository name.

Useful options:

- `--sessions N` and `--context N`: how many requests run at once, and how
  long each can be.
- `--cache-disk DIR`: keep the prompt cache across restarts.

## Platforms

Loom runs on Linux and Windows, on the Strix Halo GPU, and needs no ROCm.

- Building from source also builds the Loom runtime, which needs CMake 3.26
  (4.2 on Windows), Python 3.12 and network access. To skip Loom, pass
  `-DGUFO_EXTENSIONS=<list>` without it.
- In a container, add `--security-opt seccomp=unconfined --ulimit memlock=-1`.
- On Windows, turn on Developer Mode, or keep `TMP` on the same drive as your
  models.
- A Loom model and a non-Loom model can't share one `gufo serve`; run them in
  separate servers.
