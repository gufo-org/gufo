# First-refusal file claims (extension API change)

## Motivation

`GgufReader` validates the full tensor table while parsing headers, so a
GGUF that uses storage layouts or types core does not recognize is rejected
before any provider probe runs. An extension that fully supports such a
model — its own parsing, quantization and kernels — therefore never gets a
chance to claim the file.

## Change

`TextProvider` gains an optional `claim_file` callback, and the runtime a
matching `ClaimTextProviderForFile(path)` probe:

- CLI model-open paths (serve, prompt, chat, bench) ask registered providers
  to claim the raw path **before** core opens or inspects the file.
- A claiming provider owns loading the file end to end from
  `options.model_path`. Because core may not be able to parse what the
  provider claims, runners and drivers created for a claimed file receive an
  empty `reader` (`core::GgufReader::Empty()` / a null shared reader) and
  must not depend on it.
- When no provider claims the path, behavior is byte-for-byte the previous
  native flow, including loader diagnostics.

Providers that do not set `claim_file` are never consulted on the raw path
and see no behavior change.

## Migration

Optional, additive. No in-tree provider sets `claim_file`; out-of-tree
model extensions that handle non-core GGUF layouts set it and load the
artifact with their own reader.
