# Rust CI validation

The Rust gate runs formatting, clippy, workspace registration checks and all
workspace unit tests, with the pinned Rust 1.88 toolchain. Measurements below
were taken on November 12, 2026; [results.json](results.json) retains each run.

The baseline omits only `cargo test` from the candidate gate. Both versions use
private sccache snapshots. Nix toolchains and vendored dependencies were already
available. These are single local observations while the CPU gate also built,
not hosted-runner timings.

| Gate | Empty compiler cache | Warm compiler cache |
| --- | ---: | ---: |
| Formatting, clippy and registrations | 4.302 s | 1.527 s |
| Plus workspace unit tests | 10.478 s | 6.667 s |

The warm candidate includes a comment added to the base64 source: 75 Rust cache
hits, three misses and zero cache errors. It reruns all 31 Rust unit tests.
The extra wall time is about 6.2 seconds cold and 5.1 seconds warm in these
observations. An unchanged successful Nix derivation reuses its result instead;
its measured time is recorded in the JSON.

Each of the nine workspace crates was given a deliberately failing unit test,
one at a time. The Nix gate rejected every one. After changing the cache to
private snapshots, another failing base64 test was rejected with a warm cache.
The temporary failures were removed. A `panic=abort` build of the base64 crate
also failed at the shared boundary's compile-time guard.

`nix flake check` passed: 176 CPU tests passed and 10 tests needing model
fixtures skipped. These skips do not establish model quality.

The five original differential tests passed against the refactored library
before their frozen decoders were removed. The eight final C++ caller and
integration tests passed in 2.22 seconds. Five registration tests verify the
complete tree and missing workspace, linkage, extern and CMake registrations.
The released v0.34.0 workspace contains all five retired parser ports;
independent audio and model numerical references remain.

Reproduce the gate with:

```sh
nix build .#checks.x86_64-linux.rust -L
```

Hosted CI restores `/tmp/gufo-rust-cache`. To exercise that path locally, create
a writable directory there and expose it with
`--option extra-sandbox-paths /tmp/gufo-rust-cache`. Nix builds copy its
`objects-v1` entries into private directories and publish completed entries
with atomic renames; different build users can read the shared entries.
