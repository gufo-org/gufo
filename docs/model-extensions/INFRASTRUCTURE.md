# Extension infrastructure (cards 04–05)

API 0.1.1 implements explicit metadata registration and typed option parsing.
It does not migrate production models. Card05 adds discovery, exclusion by selection, lock policy and CLI provenance. The public headers are a
development API; the transitive surface classification remains card 10.

`gufo_extension(MANIFEST path)` validates and records a selected metadata file.
Call it for all selected manifests, then `gufo_extensions_validate()` to check
missing/cyclic dependencies and compute dependency-first, lexically stable
ordering. There is no production Python dependency. The full JSON schema is a
separate hosted check, including the optional packaging, documentation and CI
metadata from the RFC. The pure-CMake validator covers production safety and
configuration invariants; it is not a replacement JSON-schema engine.

The RFC's example omits several mappings. The concrete representation makes
these explicit:

- `api_version` is a revision floor spelled `>= 0.MINOR.REVISION`, matching the
  RFC's version policy rather than its illustrative `0.1` example. Runtime and
  headers initially report `0.1.0`. Pre-1.0 minor series must match; requested
  revisions cannot exceed the runtime revision.
- An extension has an `entry` and `registration_target`, an unconditional
  library. Each provider has a `target` and its own `entry` (both required).
  Its entry lives with the model factory and registers the corresponding
  generated `ProviderInfo` constant. The unconditional extension entry may
  register shared extension services. Provider entries are referenced only
  when their actual target is enabled; otherwise generated glue registers
  metadata with `available=false` and a reason. Thus a CPU executable can
  describe a HIP provider without referencing its factory or provider entry.
  Family libraries have `kind: library`, no entry and no providers/options.
- Target `requires` is a conjunction drawn from `hip`, `linux`, `windows` and
  `testing`. The latter follows `BUILD_TESTING`. Target `role` defaults to
  `library`; `test` creates an executable only under `BUILD_TESTING`. A
  contradictory OS conjunction or one outside declared `platforms` fails.
  Unconditional dependencies must support every declared dependent platform.
  Disabling HIP is satisfiable: it excludes the HIP target rather than making
  metadata invalid. Provider targets must be libraries. Extension registration
  targets must have no conditions and be supported on the configured platform.
- Options are declared once at extension level, as in the RFC example.
  Optional `providers` assigns an option to specific provider IDs in that
  extension; omission assigns it to every provider there. Types are `boolean`,
  `integer` (signed 64-bit), `number` (finite double), `string` and `path`.
  Defaults and model-specific validation remain in provider code. The parser
  records only explicit user values so unsupported supplied options can be
  rejected after provider selection. Boolean flags accept `--flag` or
  `--flag=true|false`; other values accept `--name=value` or `--name value`.
  Duplicate options, unknown options, invalid/overflowing numbers and positional
  arguments fail. No aliases are introduced.
- The single core-owned name catalog is
  `include/gufo/ext/core_options.def`, consumed by C++ and CMake. It reserves
  host policy, sampling and command controls. Artifact/draft/framing/voice
  options remain provider-owned. `--provider` selects a registered available
  ID. `--speculative` accepts `off` plus the union of available providers'
  `speculative` values, then validates against the selected provider.
  Commands supply their other core `OptionSpec`s to the parser. Existing CLI
  command parsers do not use the new parser until their migration cards.
- Optional `model_ids` is compile-time metadata for IDs a provider knows in
  advance. Loaded model IDs remain model-owned, as decided by the spike.

Build manifests use `gufo_extension_target(NAME name SOURCES ...
LINK_LIBRARIES ...)` for each declared target. It applies role-specific
warnings/assertions/sanitizers without depending on manifest discovery order;
production libraries strip inherited test-only `-UNDEBUG`/`/UNDEBUG`. Sanitizer
link requirements propagate to final consumers. HIP sources and per-source
flags use `gufo_extension_hip_sources(target SOURCES ... OPTIONS ...
DEFINITIONS ...)`; host HIP toolchain settings remain CMake-owned. Production
migrations must still audit their exact old/new compile settings and ISA.

`gufo_target_extensions(executable EXTENSIONS ...)` creates one explicit
registration source and constants headers for that executable. All selected
dependencies must be present and all declared targets defined through the
helper (including disabled targets). Generated files use write-if-different
semantics. They reference archive entry symbols directly; static registration
constructors and whole-archive are unnecessary. Keys may overlap between
providers for later probe-based selection; provider IDs and entry symbols may
not collide. Card 05 owns discovered-but-unselected metadata and lock checks.

Generated headers are named `gufo/ext/generated/<extension>_constants.hpp`.
Their C++ namespace is `gufo::ext::generated::extension_<extension>`; the
fixed prefix makes even an extension named `namespace` a valid namespace and
is injective over validated extension names. Provider constants remain
`kProvider_<id>` with hyphens replaced by underscores; colliding transformed
IDs are rejected. Explicit extension/provider entry symbols are global C++
identifiers and cannot be C++ keywords. Metadata names such as `off`, `no`,
`false`, `ignore` and `notfound` remain valid; property presence and explicit
empty comparisons, rather than CMake truth conversion, govern ownership and
selection.

U+0000 is rejected throughout metadata, including object keys and optional
packaging/docs fields, before any CMake JSON decoding. CMake strings cannot
represent NUL losslessly. The full schema applies the same recursive
restriction. A literal backslash followed by `u0000` is ordinary help text,
accepted and escaped into the generated C++ without truncation; the scanner
consumes escaped backslashes in pairs. Other generated list strings retain
their existing inert-data restrictions (no backslashes, quotes, list separators
or newlines), independently of the NUL rule.

`Registry` owns copied metadata, tracks extension ownership and checks duplicate
IDs and shared option types. Initial capability kinds are core-owned; adding a
new kind needs a core contract. Model factories and runner contracts remain
later cards, not erased/untyped factory pointers in this initial registry.
`Host` borrows a callback/context; absent a callback it uses the same shared
logger as existing CLI consumers. The logger and trace sources moved verbatim
into `src/core/logging/` with old headers forwarding; the `gufo_logging` target
explicitly links platform facilities and has no HTTP/serving-library dependency.

The synthetic extension fixture exercises actual static archive registration,
a dependency library, generated constants, an available CPU provider, metadata
for an unavailable HIP provider, typed parsing and selected-provider errors.
Its HIP-enabled condition variant uses a C++ synthetic target, not a HIP
compiler or model: it verifies registration decisions and boolean parsing,
without claiming GPU or production migration evidence.


## Discovery and selection (card05)

The root discovers only `src/models/*/extension.json` and
`src/families/*/*/extension.json`, with `CONFIGURE_DEPENDS` on the globs and
metadata/lock files. Platform-only returns first and needs no lock or manifest.
`GUFO_EXTENSIONS=*` means all platform-supported discovered manifests; explicit
semicolon lists (including an empty list) and `extensions.lock` are exact sets
including dependencies. Lock JSON is `{ "schema": 1, "extensions": [] }` with
no other keys. `GUFO_EXTENSIONS_FROM_LOCK=ON` conflicts with an explicit
selector. Missing names, duplicates, unavailable selected platforms and omitted
dependencies are configuration errors, as is an empty entry from a stray `;`.
Names with false-like CMake spelling remain legal. No production Python
dependency was added.

`extensions.lock` is read and validated in every mode, including plain discovery:
unknown locked names and incomplete locked dependencies fail a development
configure too, so a stale lock cannot pass silently. Built extensions are
ordered dependency-first over lexically sorted names, while excluded extensions
are reported in canonical lexical order because dependency order is meaningless
for extensions that are never built.

All discovered manifests receive structural/safety checks, including name and
entry/target uniqueness and option-type agreement. Unselected manifests skip
API-compatibility and dependency-graph requirements, allowing an unshipped
extension to target a future API or an uninstalled dependency without breaking
a lock build. Their JSON must still be structurally valid to generate truthful
excluded metadata. Selected manifests receive full API/graph validation.
Shipped/unshipped collisions in artifact keys, option names (across commands)
and model IDs fail independently of current selection, including lock builds,
and the message names the first recorded holder of the value in the other tier.
Two extensions inside the same tier may share an artifact key: overlapping keys
are decided by probes later. Provider IDs and entry symbols may not collide
(`gufo_extension()`) and shared option types may not disagree
(`Registry::AddProvider`); both fail elsewhere, not in this gate. The collision
prefix is `gufo:`, not a lock-mode claim, because the check runs in both modes.
This collision gate does not establish equivalent runtime behavior.

Selected `extension.cmake` files execute dependency-first in isolated build
subdirectories. Targets must use the existing helpers; relative source paths
are resolved against the manifest directory. Use `CMAKE_CURRENT_LIST_DIR` for
other extension-relative paths. Only selected build manifests execute. Generated
registration links actual registration/provider archives into `gufo`; disabled
HIP providers remain metadata stubs. Built family libraries also carry version
and source provenance. Excluded extensions record version/source/reason plus
keys, model IDs and option names, with `ExcludedKeyDiagnostic()` available to
later provider selectors. `diagnose` deliberately reports excluded provenance
and their artifact keys only; the full model-ID and option-name sets stay
internal to the collision gate and `ExcludedKeyDiagnostic()`. Legacy CLI
provider selection is intentionally not migrated in this infrastructure card.

There are currently zero production manifests; `extensions.lock` is deliberately
empty. Existing legacy model target/routing lists still apply in both builds.
This is an honest bootstrap, not a claim that empty selection excludes legacy
model code. Pilot migration and Card14 populate metadata on their stated
schedule. Synthetic discovery tests create real model/family source directories,
build/link generated registration, compare default/lock runtime provenance and
exercise excluded-key diagnostics. Card04's synthetic alpha/beta/base fixtures
now build in a separate CMake project, so `BUILD_TESTING=ON` cannot contaminate
the production registry or reserve real extension names. Parent warning and
sanitizer settings are forwarded to the isolated project.

`--version` retains its first line and adds API/selection/built/excluded details.
Human default `diagnose` appends the same provenance; the explicit `extensions`
section emits stable human or JSON provenance without probing hardware.
Hardware diagnostic JSON remains its existing schema. Nix's default output and
normal Windows/Devlab paths use discovery. The release package and tagged ZIP
select lock mode; `extensionsFromLock` is exposed for Toolboxes. Production
source filters admit `include/`, `schemas/` and the lock. No external image
publication, model numerical/performance qualification or Windows execution is
claimed by the initial infrastructure checks.
