# Loom packages in Gufo

`src/models/loom` serves [loom-fun](https://huggingface.co/loom-fun) packages,
AMD Loom's compiled text models, through Gufo's ordinary `TextModelRunner`.
It started as the community
[loom-gufo-extension](https://github.com/loom-fun/loom-gufo-extension)
(release 0.3.6).

```
gufo serve llm --model loom-fun/<name>
 └─ Gufo server, scheduler, prefix cache, parsers   (unchanged)
     └─ loom_runner.cpp: TextModelRunner + cache::Adapter
         └─ loom_text_embed (embed/, C API over Loom's text model layer)
             └─ the package's chat program, tokenizer, kernels and drafter
```

## Who does what

| | Gufo | Loom |
|---|---|---|
| APIs, scheduling, which sessions step together | ✓ | |
| Prefix cache policy (component cache) | ✓ | checkpoints and images it stores |
| Chat rendering, tokenizer, output and tool parsing | | package programs |
| Kernels: prefill chunks, packed decode epochs, drafted verify | | ✓ |
| KV memory | logs it | allocates and pages it |
| Sampling | with logits: grammars, logprobs, min-p, penalties | otherwise, on device |

One Gufo session is one Loom row. Prefill runs in chunks of the package's
prefill capacity; there is no layered prefill.

## Install and load

- **Install.** `gufo model install loom-fun/<name>` resolves the package
  (`loom_source.hpp`, `src/cli/model/source.cpp`). It installs the package
  revision tagged for this runtime's SDK, `v<sdk>`, falling back to the
  default branch. It also installs every Hugging Face origin the card's
  parameter files come from, at the revision the card pins. Every file is
  checked against the Hub's size and SHA-256.
- **Preset.** Chosen from the first AMD GPU, which libamdf reports the same
  way on Linux and Windows (`loom_text_embed_query_gpu`):

  | GPU | Preset |
  |---|---|
  | gfx1151 | `chat` |
  | gfx1150, gfx1152 | `chat-<target>`, else `chat` |
  | gfx1201 with 24 GB or more | `chat-r9700` |
  | other gfx1201 | `chat-16gb` |
  | anything else | `chat`, with a warning |

- **Logits.** A package that declares `llm.logits` opens with them, and
  reopens without them if its state can't return rows.
- **Parameter files.** A file missing at its card path is looked up by
  origin: in `--loom-<origin>`, else in the model directory. Loom then opens a
  private directory of links, `$TMPDIR/gufo_loom_<pid>_*`, shaped like the
  one loom-fetch writes. Missing files are an error that names each one.
- **SDK releases.** Gufo is built for one Loom SDK, pinned in
  `runtime.cmake` and `.devops/nix/hrx/loom-runtime.nix`. A package built for
  another SDK series is refused at install and at load. A new SDK release
  means a new Gufo release, followed by `gufo model update`.

## Memory

All sessions share one KV pool, which Loom allocates and pages. A request
reserves the pages its prompt and output reach. When the pool is short, the
runner releases the oldest prefix-cache checkpoints and tries again.

The pool holds `--sessions` full contexts (`--context`, the package's own by
default), so every session can reach its context at once. It is fixed and
committed at load, except with `--cache-disk`, whose checkpoint images need
elastic backing.

At load the runner logs the device memory, the KV pool's share of it and the
tokens the pool holds (`loom_text_embed_memory`). It claims no per-session
state, because Gufo's per-session claim would count the shared pool once per
session.

## Runtime

`loom_api.hpp` loads `libloom_text_embed.so` (`loom_text_embed.dll` on
Windows) when the first package opens, so Gufo runs without it otherwise.

- **No ROCm.** Loom drives the GPU through libamdf, using KFD on Linux and
  WDDM on Windows.
- **One GPU runtime per process.** The provider sets `serving.hip: false`,
  so `serve` makes no HIP call; libamdf and ROCr can't share a process on
  Linux.
- **Build.** `runtime.cmake` uses a `loom-text-embed` pkg-config module when
  one exists (Nix: `.#loom-runtime`). Otherwise it builds the pinned
  hrx-system release with `ExternalProject`.
- **Install location.** The library goes to `<prefix>/lib/gufo` on Linux, or
  beside `gufo.exe` on Windows.

## Limits

- Strict tools, JSON schemas, logprobs, min-p and penalties need a package
  built with logits. Without them, requests are served unconstrained.
- Greedy output can differ from loom-serve at near-ties, because Loom's
  results depend on batch shape.
- Containers need `--security-opt seccomp=unconfined`.
- On Windows without Developer Mode, the private directory uses hard links,
  so `TMP` must be on the model's drive.

## Tests and results

- **Unit tests, no GPU needed.**
  - `tests/package_test.cpp`: install, presets, files.
  - `tests/runner_logits_test.cpp`: the logits path, against a fake embed
    library.
- **Measured on hardware.**
  - [BENCHMARKS.md](BENCHMARKS.md): throughput.
  - [QUALITY.md](QUALITY.md): per-package conformance.
  - [EXPERIMENTS.md](EXPERIMENTS.md): experiments.
