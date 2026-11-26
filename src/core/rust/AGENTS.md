# Rust components

These rules apply to every Rust component under `src/core/`, including crates
outside this directory. Read them before adding or changing a crate.

## Scope

- Use Rust for components with bytes in and values out (parsers, decoders,
  encoders, compilers of user input), for state behind an opaque handle, and
  for the network edge: transport and protocol decoding and encoding. `tokio`
  and `hyper` are accepted for the HTTP server.
- The engine stays C++: HIP and kernels, model runtimes, the cache (including
  durability), the scheduler and the extension API types. HTTP handlers keep
  running in C++ behind the Rust transport. When an extension API type such as
  `ChatRequest` is the output, Rust builds the C++ value through callbacks and
  the C++ interface stays as it is.
- Refactor tangled engine code that shares GPU or cache state in C++, such as
  `component_text_cache.cpp`, `text_generation_scheduler.cpp`, `src/cache/`,
  the model drivers and `json_constraint.cpp`. A C ABI in the middle adds a
  seam without untangling anything.
- Candidates are GitHub issues; comment on one to be assigned. Propose a
  component without an issue in
  [Discussion #864](https://github.com/gufo-org/gufo/discussions/864) before
  starting it. For a large port, agree on the module split in its issue before
  writing code.

## Layout

- Build crates from small modules that are easy to read, review and update.
  Do not port a tangled C++ file one to one into a single large Rust file.
- Keep one crate per component in `src/core/<name>`. Register it in the root
  workspace and link it through `gufo-rust`, directly or through another
  component. Direct dependencies of `gufo-rust` need an `extern crate` in its
  `src/lib.rs`. Add C header directories to its `CMakeLists.txt`.
  `python3 tools/ci/check-rust-workspace.py` checks these registrations; Nix
  derives the Rust check's source directories from the workspace.
- Put the C header in `include/gufo/<name>.h` within the component. Use only C
  types; mirror every `#[repr(C)]` layout and status value.
- Keep OS facilities in `src/core/platform/`, as the root guidance requires.
  Prefer portable Rust std APIs. When std lacks a facility, concentrate OS
  code in one module per facility, with no scattered `cfg` checks; keep
  Windows building.

## C boundary

- Use `#![deny(unsafe_code)]` and
  `#![deny(clippy::undocumented_unsafe_blocks)]` at crate roots. Allow unsafe
  only on exported functions, narrow helpers and ABI tests, with safety
  contracts and comments. Use `#![forbid(unsafe_code)]` without a C boundary.
- Use `gufo_ffi::guard` for every exported function, including destructors.
  It requires unwinding; panic-abort builds fail. Return the internal failure
  value on a panic. Discard partial results and keep handles safe to free.
- Use `gufo_ffi::input` and `output` for pointer-length slices. Null is accepted
  only at length zero; callers still guarantee alignment, extent, lifetime
  and aliasing rules for nonempty buffers.
- Return borrowed bytes through a callback valid for the call, write into a
  caller-owned buffer, or return an opaque handle with an explicit `_free`.
  C++ never frees Rust allocations with its own allocator.
- Make C++ callbacks `noexcept`. Use `core::ffi::CallbackFailure` from
  `src/core/rust/callback.hpp`, and call `Rethrow()` after the Rust call.
- Distinguish rejected client input (400-class) from internal failures
  (500-class: memory exhaustion, caller errors or parser bugs).

## Input and dependencies

- Check caller limits or named constants before allocating. Use `try_reserve`
  for input-sized allocations. Stream parsers must behave the same one byte
  at a time and avoid rescanning bytes already searched.
- Pin external dependencies exactly (`=x.y.z`), disable default features and
  enable only needed features. Add license text to `licenses/rust.txt`, a row
  to `THIRD_PARTY_NOTICES.md` and the name to `tools/ci/check-dependencies.py`.
- Prefer std-only code when a dependency saves little or changes accepted
  input. Avoid unsafe-heavy dependencies on untrusted input.
- Use released crate versions. Pin a Git revision or an unreleased upstream
  pull request only when there is no alternative.
- For a large dependency tree, such as `tokio` or `hyper`, also measure the
  cold Nix build time it adds.

## Ports and verification

- Prove a C++ port against a frozen copy of the replaced C++, with generated
  grids and deterministic mutations. When the C++ is hard to isolate, such
  as a server endpoint, a golden output generated on unmodified `main` also
  works. Explain every difference; fail on unknown ones. Behavior changes
  require `!` in the PR title and a list in its description.
- Once the port ships, replace the frozen copy and comparison loops with
  expected-result tests covering edge cases and documented differences.
  Keep C ABI and caller tests where integration matters.
- Measure affected timing against main with the same toolchain and workload.
- Rust is pinned by `rust-toolchain.toml` (1.88). Use no newer std or language
  features. Run:

```sh
cargo fmt --all --check
cargo clippy --workspace --locked --offline --all-targets -- -D warnings
cargo test --workspace --locked --offline
nix build .#checks.x86_64-linux.rust
```

The Nix command vendors dependencies and runs all three Cargo checks. Unit
tests use the default test profile, with debug assertions and overflow checks.
The Windows build checks C++ callers but does not run Rust unit tests.
