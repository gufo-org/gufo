# Writing an extension

This guide takes you from the extension template to a model that others can
build, trust and keep updated. It assumes you know C++ and the model you're
porting; the [extension API](extension-api.md) has the contract details.

## What an extension owns

An extension provides the model; Gufo keeps the policy around it.

- **The extension owns** loading its files, tokenization, chat templates,
  tool-call syntax, GPU execution, speculative decoding and the model's part of
  what identifies a cached state.
- **Gufo owns** HTTP and the OpenAI API, scheduling, cache retention, memory
  budgets, diagnostics and the test harness.

The line between them is one **step**: Gufo decides when to prefill a prompt or
decode the next token and which requests share the step; the extension runs
every layer and kernel inside it.

An extension holds one or more **providers**, each implementing one
capability. A provider has:

- an **ID**, such as `flash-next-q4`;
- **artifact keys**, such as `gguf:qwen4exp`, naming the files it claims;
- a **probe**, a cheap check that reads a file's metadata, without the GPU, and
  says whether this provider runs it;
- a **factory** that builds the running model once the probe says yes.

## Core or extension

Gufo's maintainers spend their time on the core: the parts every model relies
on to serve well. Models live in extensions.

- **Core** is everything Gufo owns above, plus shared building blocks such as
  GGUF loading, sampling, quantization helpers and the extension runtime.
  Change the core when the change helps more than one model, or when your
  extension needs something the [extension API](extension-api.md) doesn't offer
  yet. Add it as a general contract, not a hook for one model.
- **An extension** is everything about one model: its engine, kernels,
  loaders, templates, tests and quality evidence. A new model, or a new variant
  of one, starts as an extension in your own repository, and you maintain it.

Gufo ships one or two **official** models per use case. They live in Gufo's
repository and come with the default build; every other extension is
installed separately. Each official model needs its own optimizations to be
the fastest and most accurate on Strix Halo, and maintainer time is limited,
so a model becomes official only when it fills or improves a use case and
someone commits to maintaining it. A model that works well isn't enough on its
own. An official model stays until a better one arrives for the same use case;
it then moves to its own repository as a community extension. A community
extension gets the same runtime, cache and scheduler as an official one.

## Start from the template

Copy [examples/extension_template](../examples/extension_template/README.md)
into a new repository:

| File | Purpose |
| --- | --- |
| `extension.json` | The manifest: identity, API floor, providers, keys, options, targets |
| `extension.cmake` | Targets, sources, dependencies, tests and install rules |
| `src/template_model.cpp` | Registration, using constants generated from the manifest |
| `flake.nix`, `flake.lock` | A pinned Gufo package with this extension added |
| `LICENSE`, `README.md` | License, maintainer, build and support information |

Rename `template_model`, its targets and symbols, the provider ID and the
artifact key together. The template only registers metadata; a model works once
you implement a capability and prove it with tests.

The [LFM3 extension](https://github.com/gufo-org/lfm3-gufo-extension) is a
complete example, with text, speech recognition, speech synthesis and decision
providers that share one backbone.

## The manifest

`extension.json` is plain JSON, so CMake, Nix and CI read it without building
anything:

- `name` and `version`: your extension's identity and version.
- `api_version`: the lowest [extension API](extension-api.md#versions) you need,
  such as `">= 0.1.12"`.
- `registration_target` and `entry`: the library and function that register
  the extension.
- `providers`: each provider's ID, capability, keys, model IDs, library
  (`target`) and registration function.
- `targets[].requires`: what a library needs to build, such as `hip` or
  `testing`. Test libraries use `role: "test"`.
- `depends`: the extensions yours builds on; see
  [depend on another extension](#depend-on-another-extension).
- `options`: command-line options your providers add, with their types.
- `models`: the weights `gufo model` can install, one entry per model. Each
  component names its role, the option its path fills (one fills `--model`),
  whether `install` treats it as `required`, `default` or `optional`, a
  pinned Hugging Face `source` and `include`/`exclude` globs. Run
  `python3 tools/models/lock.py <extension-dir>` to pin every file's size and
  SHA-256 in `models.lock.json`; `gufo model` installs only locked files.
- `packages`, `licenses`, `install_data`, `runtime_libraries`: what the build
  needs and what an install must ship. `packages[].pkg_config` names a
  pkg-config module, which the build then provides as `PkgConfig::<module>`.

Keep host code (probes, metadata parsing) apart from HIP code, so diagnostics
and probes work on machines without a GPU, and keep test code under the
`testing` condition. Declare only the platforms you test.

## Depend on another extension

An extension can build on another one's code, such as a new checkpoint format
for a model another extension already runs. List it in `depends`:

```json
{ "depends": ["qwen38_flash_next"] }
```

Gufo then builds it before yours, in-tree or from `GUFO_EXTENSION_DIRS`, and
refuses to configure if it's missing or unselected, if a lock build locks
yours without it, or if the dependencies form a cycle.

Link only what the other extension **exports**: a library it publishes as
`gufo::<name>` from its `extension.cmake`:

```cmake
gufo_extension_export(TARGET gufo_qwen38_flash_next_hip NAME qwen38_flash_next_hip)
```

`<name>` is the exporting extension's name or starts with it and `_`, and
mustn't fall in a longer extension name's namespace. Only the extension's own
`extension.cmake` can export its targets. For Flash-Next, include only the
experimental `<gufo/ext/families/qwen/models/flash_next/...>` headers; a new
format serves through `CreateQwenFlashNextTextRunner(options, stages, weights,
error)`. Your targets then link the alias:

```cmake
gufo_extension_target(
  NAME my_extension_hip
  SOURCES src/provider.cpp
  LINK_LIBRARIES gufo_ext_runtime gufo::qwen38_flash_next_hip
)
```

After every extension is built, Gufo checks the direct links of each target
your `extension.cmake` defines, helper targets and `INTERFACE` links included:
a link to another extension's library, by export or target name, fails unless
you list that extension in `depends`. For an extension outside Gufo's tree,
every transitive link is checked too. A target disabled by its `requires`
exports nothing, so give targets that link it the same `requires`. A module
may depend only on in-tree extensions its Gufo compiles in, and link only their
exports; not on Windows yet. See [modules](internal/extension-build.md#modules).

> **Warning:** Depending on an export takes you outside the versioned
> [extension API](extension-api.md#the-boundary): its headers and libraries are
> the exporting extension's code, with no compatibility promise, like an
> experimental header. An export from Gufo's own tree passes the external
> include and link checks whole, so it can expose private Gufo headers. Pin the
> dependency to the exact revision you tested. These checks catch mistakes,
> not hostile build code, which runs with your permissions.

## Implement a capability

Pick the interface for your capability:

| Capability | Header |
| --- | --- |
| Text generation | [`text.hpp`](../include/gufo/ext/text.hpp) |
| Embeddings | [`embeddings.hpp`](../include/gufo/ext/embeddings.hpp) |
| Speech recognition | [`asr.hpp`](../include/gufo/ext/asr.hpp) |
| Speech synthesis | [`tts.hpp`](../include/gufo/ext/tts.hpp) |
| Images | [`image.hpp`](../include/gufo/ext/image.hpp) |
| Video | [`video.hpp`](../include/gufo/ext/video.hpp) |
| Decisions | [`decision.hpp`](../include/gufo/ext/decision.hpp) |

Read the interface and an existing provider of the same capability first.
Include only `<gufo/ext/...>` headers and link `gufo_ext_runtime`; the build
rejects private Gufo headers and libraries.

**Probes** must be fast, deterministic and GPU-free. Check the actual file, not
only its key, and never claim files another provider owns with a broad match:
an ambiguous file must stay an error.

**Text models** bring their own tokenizer, chat framing and tool syntax, and
must support continuation through the component cache. Declare only the inputs,
batching and state features you implement, and keep sampler and cache identity
consistent across rollback. Wire the `prompt`, `chat` and `bench` drivers as
well as the server runner. The [extension API](extension-api.md) covers the
cache adapter, sampling defaults, reasoning markers, tool dialects and
options.

**Image models** declare startup options for `serve image` in their manifest.
Declare which request features you support and apply background conditioning
in your runner. See [`image.hpp`](../include/gufo/ext/image.hpp) for the contract.

## Build and test

Point a Gufo checkout at your directory:

```sh
cmake --preset gpu-test -DGUFO_EXTENSION_DIRS=/absolute/path/to/my-extension
cmake --build --preset gpu-test --target gufo my_extension_tests --parallel 4
ctest --preset gpu-full -R '^my_extension\.' --output-on-failure
build/gpu-test/gufo diagnose --section extensions --json
build/gpu-test/gufo prompt --model /models/my-model.gguf \
  --prompt 'What is the capital of France?' --think off
```

A CPU-only build should still run your probes and report the GPU provider as
unavailable, with a reason.

## Validate before publishing

A model that answers one prompt isn't done. The
[conformance guide](internal/extension-conformance.md) lists
what each capability must prove. In short:

1. Test configuration, tokenizer and operators against an independent
   reference.
2. Compare GPU results with a CPU reference.
3. Test real model runs over HTTP: session isolation, streaming, cancellation,
   and every mode you advertise.
4. For text models, test the cache: continuing, editing and restoring
   conversations.
5. Keep a quality report and benchmark results with the model, including
   failures and known limits.

Then install into a fresh prefix and test the installed binary on every
platform you claim. Model weights have their own licenses: document them, and
don't bundle weights you may not redistribute.

## Publish and maintain

Your README should name a maintainer and list the supported Gufo revisions,
platforms, weight files and how to get them, quality gaps and release history.
The [catalogue](https://gufo.blog/extensions) lists GitHub repositories that
a search for `gufo-extension` finds, so put that in your repository's name or
description. To appear in `gufo extension install`, ask for a review in the
[extension registry](https://github.com/gufo-org/gufo-extension).

For each Gufo update:

1. Pin the new Gufo revision and rebuild. Gufo promises that extensions still
   compile, not that old binaries still load.
2. Keep `api_version` at the lowest revision that has everything you use.
3. Rerun the affected numerical, API and performance checks.
4. Check that installs still ship every library, data file and notice.
5. Bump your version and update the quality notes, keeping failed results
   visible.

If your model fits [Gufo's goal](#core-or-extension), you can ask for it to
become official. That needs a named maintainer and the evidence above; the
[support policy](internal/extension-distribution.md#support-states)
describes the process.
