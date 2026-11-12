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
- `options`: command-line options your providers add, with their types.
- `packages`, `licenses`, `install_data`, `runtime_libraries`: what the build
  needs and what an install must ship. `packages[].pkg_config` names a
  pkg-config module, which the build then provides as `PkgConfig::<module>`.

Keep host code (probes, metadata parsing) apart from HIP code, so diagnostics
and probes work on machines without a GPU, and keep test code under the
`testing` condition. Declare only the platforms you test.

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

If your model fits Gufo's goal, the best model for each use case on Strix
Halo, you can ask for it to become official. That needs a named maintainer and
the evidence above; the
[support policy](internal/extension-distribution.md#support-states)
describes the process.
