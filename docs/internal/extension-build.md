# Extension build system

How CMake discovers, validates, selects, builds and registers extensions, and
how the same sources become loadable modules. The manifest fields are listed in
[writing an extension](../writing-extensions.md#the-manifest); the design
rationale is in the [model extensions RFC](design/model-extensions/RFC.md).

## Discovery and selection

Configure discovers `src/models/*/extension.json`,
`src/families/*/*/extension.json` and every root in `GUFO_EXTENSION_DIRS`
(an extension root, or a directory of them), with `CONFIGURE_DEPENDS` on the
globs, manifests and lock.

| Selector | Builds |
| --- | --- |
| `GUFO_EXTENSIONS=*` (default) | Every discovered extension the platform supports |
| `GUFO_EXTENSIONS=a;b` | Exactly those, plus nothing else; list every `depends` |
| `GUFO_EXTENSIONS_FROM_LOCK=ON` | Exactly `extensions.lock`; conflicts with an explicit list |

`extensions.lock` is `{ "schema": 1, "extensions": [...] }` and is validated in
every mode, so a stale name fails a development configure too. Missing names,
duplicates, unsupported platforms, omitted dependencies and empty entries are
configuration errors. Built extensions are ordered dependency-first over
lexically sorted names; excluded ones are reported lexically.

Every discovered manifest gets structural and safety checks: names, entry and
target uniqueness, option-type agreement, and no U+0000 anywhere (CMake strings
can't hold NUL). Only selected manifests get API-compatibility and dependency
checks, so an unselected extension may target a future API.

**Collision gate.** Artifact keys, option names and model IDs may not be shared
between an official (locked) and a community extension, whatever is selected;
the error names the first holder. Two extensions in the same tier may share a
key, and probes decide at run time. Provider IDs and entry symbols may never
collide.

## Manifest validation

`gufo_extension(MANIFEST path)` validates and records each selected manifest;
`gufo_extensions_validate()` then checks missing or cyclic dependencies.
Validation is pure CMake; the full JSON schema in `schemas/` runs as a separate
hosted check.

- `api_version` is a floor, `>= 0.MINOR.REVISION`. The minor must equal the
  host's and the revision may not exceed it. The host revision comes from
  `include/gufo/ext/api-series.json` plus the number of change records.
- An extension has an `entry` and an unconditional `registration_target`. Each
  provider has its own `target` and `entry`; a provider whose target is disabled
  registers metadata with `available=false` and a reason, so a CPU build can
  describe a GPU provider without linking it. Family libraries are
  `kind: library` with no entry, providers or options.
- `targets[].requires` is a conjunction of `hip`, `linux`, `windows` and
  `testing` (`BUILD_TESTING`). `role: "test"` creates an executable only under
  `BUILD_TESTING`. Contradictory or undeclared platforms fail.
- Options are declared at extension level; `providers` limits one to some
  providers. Types are `boolean`, `integer` (signed 64-bit), `number` (finite
  double), `string` and `path`. Core reserves its own names in
  `include/gufo/ext/core_options.def`. `--speculative` accepts `off` plus the
  union of available providers' values.

## Building and registering

Selected `extension.cmake` files run dependency-first in isolated
subdirectories, and must define every declared target, disabled ones included,
through the helpers:

- `gufo_extension_target(NAME ... SOURCES ... LINK_LIBRARIES ...)` applies
  role-specific warnings, assertions and sanitizers; production libraries
  strip inherited `-UNDEBUG`.
- `gufo_extension_hip_sources(target SOURCES ... OPTIONS ... DEFINITIONS ...)`
  adds HIP sources with per-source flags.
- `gufo_target_extensions(executable EXTENSIONS ...)` generates one
  registration source and the constants headers for an executable.

The generated registry calls each entry in a fixed order and references entry
symbols directly: no static constructors, no whole-archive. Constants live in
`gufo/ext/generated/<extension>_constants.hpp`, namespace
`gufo::ext::generated::extension_<extension>`, with `kProvider_<id>`
(hyphens become underscores; colliding results are rejected). Generated files
are written only when they change.

External extensions see only the public headers: their targets are checked
against the include and link allowlist (`gufo_ext_runtime`, `gufo_ext_core`,
`gufo_ext_cache`, `gufo_ext_hip`).

## Modules

`GUFO_EXTENSION_MODULES=<names>` builds those extensions as
`extension-modules/gufo-ext-<name>.so` instead of linking them.
`gufo_extension_modules()` compiles their libraries position-independent
against the interfaces of the public targets only; the services resolve from
the `gufo` executable, which exports its symbols. Each module exports
`gufo_ext_module_build_id_v1`, `gufo_ext_module_manifest_v1` and
`gufo_ext_module_register_v1`.

`gufo_extension_build_id()` hashes the public headers in `SURFACE`, the API
version, compilers and their versions, ROCm, global flags, GPU architecture,
build type and sanitizers, and writes `build/<dir>/extension-build-id.txt`.
The loader refuses a module whose identity differs. `gufo_extension_sdk()`
installs the headers, CMake helpers, `profile.cmake` and `sdk.json` that
`gufo extension install` builds against.

## Provenance

`gufo --version` adds API, selection mode, built and excluded extensions.
`gufo diagnose --section extensions [--json]` reports each extension's name,
version, source (where it was built from, or `module:<path>` for a module), provider
availability and excluded artifact keys, without touching the GPU. The full
excluded model-ID and option-name sets stay internal to the collision gate.

## Tests

`tests/extensions/` builds synthetic extensions in a separate CMake project, so
`BUILD_TESTING` never adds fixtures to the production registry. They cover
archive registration, dependency libraries, generated constants, available CPU
and unavailable HIP providers, typed option parsing, selection errors, lock and
discovery provenance, and excluded-key diagnostics.
