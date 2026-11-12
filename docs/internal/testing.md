# Testing

Which checks cover which change, and how to run them. Run the smallest check
that covers the change; correctness and model-owned quality limits must pass
before performance results count. CMake and CTest work with system
dependencies or inside `nix develop`.

## Commands

```sh
# Formatting on the editing host, using CI's pinned formatter
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
# Add --fix to apply formatting.

# Documentation changes
nix build .#checks.x86_64-linux.docs

# Hosted CPU/repository checks
nix flake check

# Focused test (replace the target and expression)
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target minimax_h3_sampling_test
nix develop -c ctest --preset gpu-full -R '^minimax_h3_sampling_test$' --output-on-failure

# Host memory/undefined-behavior diagnostics
nix develop -c cmake --preset cpu-sanitizer
nix develop -c cmake --build --preset cpu-sanitizer
nix develop -c ctest --preset cpu-sanitizer --output-on-failure
```

Preset definitions, build options and repository mechanics are in
[development](development.md). Formatting and the Python
documentation/dependency checks run on the editing host without building Gufo;
their entrypoints are in `tools/ci/`.

Selection by preset: `gpu-fast` excludes `slow` and `external-model` tests,
`deepseek-gpu` and `qwen-gpu-kernel-oracle` select model/device suites, and
`gpu-full` covers the complete hardware tree. External-model tests require
their documented local artifacts. A skip due to an absent model or device is
not a quality pass.

## Rust components

Each Rust component is a crate in the Cargo workspace of the root
`Cargo.toml`, with a small C ABI. The `gufo-rust` crate in `src/core/rust`
links them into the one static library C++ links, so the standard library is
linked once. The workspace uses the toolchain pinned in `rust-toolchain.toml`.
`nix build .#checks.x86_64-linux.rust` checks rustfmt, clippy and all workspace
unit tests offline. `nix flake check` includes it; the CMake `check-pr` target does not lint Rust.
Outside Nix, run:

```sh
cargo fmt --all --check
cargo clippy --workspace --locked --offline --all-targets -- -D warnings
cargo test --workspace --locked --offline
```

CMake Debug builds use Cargo's development profile. RelWithDebInfo uses an
optimized profile with full debug information; with `BUILD_TESTING=ON` that
profile also enables debug assertions and overflow checks, matching the C++
test assertions, while shipped builds leave them off. Other configurations use
the release profile. CMake invokes Cargo on every build, leaving incremental
rebuild decisions to Cargo, and links the native system libraries that the
configured `rustc` reports for its standard library. The C++
sanitizer preset does not instrument the Rust library; it uses the development
profile when configured as Debug.

`png_decode_test` checks expected RGB bytes across color types, sample depths,
transparency, Adam7 interlacing and EXIF orientations. It fixes the PNG port's intentional framing
behavior: missing Adler trailers and unknown non-letter chunks are accepted;
bad IEND CRCs, nonempty IEND and stray IDAT after text are rejected.
APNG metadata is ignored while decoding the default image. Libpng only
writes fixtures; no frozen decoder remains.

`wav_decode_test` checks PCM16 values, the strict caller's status mapping,
truncation and limits, and propagation of C++ callback exceptions. Rust WAV
unit tests cover wider sample formats, float clamping, chunk layouts and limits.
Both C++ tests run in `check-pr`.

`audio_decode_test` tests `core::DecodeAudio`, `DownmixMono` and `Resample`
on generated audio: the float value of each WAV encoding (8- to 32-bit PCM,
32- and 64-bit float, A-law, mu-law) on both paths, every 16-bit, A-law and
mu-law value and `WAVE_FORMAT_EXTENSIBLE` on `kAll`, the strict path's float
clamping, the channel mean, also while decoding (`downmix_mono`), sines through the resampler from 8 to 192 kHz to 16 and 48 kHz
(passband error below 1e-4, stopband rms below 1e-5, soxr's output length),
small Ogg Opus and MP3 fixtures, and the status of each rejection. It runs in
`check-pr`. The resampler and audio decoder crates also have Rust unit tests
covered by the workspace command above.
The decoder's tests compare its direct WAV sample conversion with symphonia's
PCM decoder bit for bit.
`embedding_gemma2_audio_decoder_test` compares EmbeddingGemma 2's decoder with
waveforms from the official librosa backend: 16 kHz probes within 2e-6;
resampled probes within 0.02 at the first sample, where soxr starts a
two-stage stream differently, and within 1e-3 and -55 dB of error energy
after it.

`base64_test` checks RFC 4648 vectors through the codec and audio callers,
round trips, canonical padding, decoded-size limits, data URLs and invalid
reference audio. Rust unit tests cover error ordering and the codec itself.

`http_request_test` checks expected head fields and framing through the C ABI,
both at once and one byte at a time. It fixes each intentional difference of
the port: strict request-line whitespace, token methods, valid UTF-8 targets,
no control bytes in targets or values, a 100-field limit, bare LF line ends
and one leading empty line. Rust tests also check query parameters, header
lists and caller errors. `http_server_test` checks parsing over a socket.

`multipart_test` compares the Rust multipart parser, with each API's limits,
with frozen copies of the three 0.33.0 parsers in
`tests/cli/multipart_reference.cpp`. Forms as clients encode them must give
every API the same fields, also when the API stops after one of the first
fields; error messages may differ, and other forms, which RFC 7578 may read
differently, are counted, not compared. The forms cover client-style
boundaries, names, filenames and contents, up to 34 fields, forms with what
clients do not send, and random mutations; the forms captured from curl,
Python requests, Node.js and Go in `tests/fixtures/multipart` must be among
the canonical ones and parse completely. Every rejection the parser has must be
reached. It runs in `check-pr`; the parser's Rust unit tests check the RFC
rules for the other forms.

`websocket_frame_test` checks expected queued messages and reply frames through
the C ABI: fragments, pings, close codes, UTF-8, length encodings, truncation
and queue limits. Rust tests cover protocol rejections and buffer reuse;
`audio_websocket_test` covers sockets. All three parser caller tests run in
`check-pr`. The frozen C++ decoders were retired after their ports shipped in
v0.34.0 or earlier. The audio decoder has no frozen C++ oracle; its independent
symphonia PCM and numerical resampler checks remain.

The hosted Rust check uses Cargo's default test profile, with overflow checks
and debug assertions enabled. The Windows job builds the Rust library and C++
callers but does not run `cargo test`. A workspace registration check runs in
the Rust gate; it verifies every component manifest, linkage through the static
library, direct `extern crate` declarations and CMake header directories.

Hosted checks restore `/tmp/gufo-rust-cache` for sccache, separately from Cargo's
offline vendored sources and C++ ccache. Each Nix derivation runs its own server
against a private snapshot, then publishes complete immutable objects with atomic
renames. Shared entries are readable and writable by different Nix build users;
concurrent servers never write to the same local sccache directory. Keys include the pinned toolchain and
Nix inputs, prefer the same lockfile, and fall back to compatible older lockfiles.
Every run saves a fresh Actions cache key. Test executables still run on every
rebuilt check; an unchanged successful derivation reuses Nix's result cache.
[Validation and timings](rust-ci-validation/README.md) record the cold, warm
and deliberately broken-test checks.


## Hosted versus local checks

GitHub runs `nix flake check`: formatting, documentation, dependency inventory,
server-command checks and one CPU build/test derivation. The CPU suite runs the
`check-pr` selection from [cmake/Checks.cmake](../../cmake/Checks.cmake) with up
to four CTest slots, and validates a separate lock-mode CLI build. It does not build ROCm, download weights, run H3 qualification or run
whole-tree clang-tidy. Missing-model skips remain visible and are not quality
passes. The workflow cancels superseded runs and has a 25-minute limit.

```sh
# Focused CMake contract subset for local iteration (also used on Windows)
cmake --preset cpu-test
cmake --build --preset pr --parallel 4

# Individual CPU check, or explicit qualification tools
nix build .#checks.x86_64-linux.tests
nix build .#static-analysis
nix build .#h3-manifest
nix build .#h3-quality
nix build .#h3-ml-quality
```

Static analysis and the three H3 qualification packages above are explicit-only: `nix flake check`
excludes them and the six CTest suites labelled `h3-qualification`. Their tests
remain available through those packages or an explicit local CTest selection.

The default Nix CPU suite also excludes `serving_bench_test`. Run its mocked
benchmark-client/reporting regressions explicitly when changing those tools:

```sh
nix develop .#checks.x86_64-linux.tests -c python3 tests/tools/serving_bench_test.py
```

The default `.clang-tidy` runs the static analyzer and selected correctness
checks. Run it explicitly against an affected source file:

```sh
nix develop -c clang-tidy -p build/cpu-test src/core/sampling.cpp
```

The compilation database must already be configured (the `cpu-test` preset
exports it).

Full CPU builds need Python NumPy for the offline tool tests. GPU model oracles
need the documented local weights; use focused targets and CTest labels before
running the complete tree. H3 development normally needs an analytic primitive,
one transformer block or one denoiser forward. Full video generation and LPIPS
are release/quality qualification, not routine PR work. `nix flake check` is the canonical hosted CI command.

For Qwen image-prefix changes, build `qwen_vision_serving_test` with `gpu-test`
and run it with `MODEL DRAFT_OR_DASH IMAGE_DIRECTORY --append-only`.
The directory needs `red.png` and `blue.png`. This compares complete target
logits when images are attached early or after a cached prefix, including
snapshot restoration and sampled Flash-Next MTP replay.
Qwen27B also checks exact restoration after cancellation between layers.
Use the [HTTP continuation check](server.md#hip-execution)
for cancellation, thinking controls and disk restart.

## NPU device checks

The dev shell carries the IRON toolchain (`.#iron`: mlir-aie and Peano) and
sets `GUFO_IRON_PYTHON`. CMake then compiles the in-tree NPU designs
(`src/core/npu/kernels`, `cmake/GufoNpuDesigns.cmake`) and adds
`npu_device_test`, which runs `npu::Q8Gemm` against a double reference. It
exits as skipped without an NPU. `gufo diagnose --section npu` reports the
accel node, driver, firmware and whether XRT opens the device.

```sh
nix develop -c cmake --preset cpu-test
nix develop -c cmake --build --preset cpu-test --target npu_device_test
nix develop -c ctest --preset cpu-test -R '^npu_device_test$' --output-on-failure
```

## Boundary and link-graph gates

Three repository gates describe who may reach the serving tier; all three are
`cpu;repository` CTest tests and run in the hosted selection.

```sh
# No src/cli include from a model, family, cache, src/ext or public API source
python3 tools/ci/check-extension-isolation.py --fail-on-serving-boundary
# The same scan with every location listed
python3 tools/ci/check-extension-isolation.py --fail-on-serving-boundary --details
# No link edge from below gufo_ext_runtime to gufo_serving or gufo_http
python3 tools/ci/check-target-graph.py --report build/<dir>/gufo-link-graph.json
```

`gufo-link-graph.json` is written by the configure step
(`cmake/TargetGraph.cmake`) and holds the transitive link closure of every
model and support library declared in a `library.json`, plus `gufo_core`,
`gufo_cache`, `gufo_cache_hip` and `gufo_ext_runtime`. Only explicitly allow-listed HIP libraries may be listed under `absent`,
and only when the report records `hip: false`; missing protected names fail.
Unsupported interface expressions fail configuration. The checker itself has unit coverage in `tests/tools/`.

`ext_runtime_neutral_contracts_test` builds inside the isolated extension
fixture project and links only `gufo_ext_runtime` among Gufo libraries (plus
OpenSSL for public crypto-header coverage): the neutral continuation
bases, host memory budgets, reasoning tags and device-probe default have to
work with no serving or HTTP target even defined. The capture and backing-plan
contract is cache-owned and is covered by `cache_adapter_test`.

For Windows extension-header and runtime changes, the **Check Windows extension
contracts** job runs this fixture independently of the package build. It uses
the hosted ClangCL toolset and only OpenSSL from the application's vcpkg baseline;
it does not require the HIP SDK or media dependencies. Rerun that individual job
while iterating. The full package job still validates the production toolchain.
On a Windows development host with OpenSSL installed, the same focused check is:

```powershell
cmake -S tests/extensions/fixture -B build/windows-extensions -A x64 -T ClangCL -DOPENSSL_ROOT_DIR=C:/path/to/openssl
cmake --build build/windows-extensions --config RelWithDebInfo --parallel 4 --target extension_runtime_test extension_role_test ext_runtime_neutral_contracts_test
ctest --test-dir build/windows-extensions -C RelWithDebInfo --output-on-failure --no-tests=error --timeout 60 -R '^(extension_runtime_test|extension_role_test|ext_runtime_neutral_contracts_test)$'
```

Qualify performance with a `BUILD_TESTING=OFF` artifact: test-preset binaries
keep assertions on inside the serving sources, unlike production.

## Text API functional tests

Use [tests/functional/](../../tests/functional/README.md) for live API regression
checks. Its single runner starts a private server and covers SDK transports,
sampling defaults/overrides, tools, thinking, images, concurrent isolation,
cancellation and RAM/disk replay. Select focused suites and the affected
AR/speculative model modes; full model runs stay outside hosted CI.

## Match validation to the change

| Change | Relevant checks |
| --- | --- |
| Documentation | `check-docs.py`: links, anchors, JSON examples, public-page and figure rules ([docs/AGENTS.md](../AGENTS.md)) |
| Loader/container/tokenizer | independent fixtures, malformed/truncated input, exact token IDs |
| Kernel or quantizer | analytic oracle, affected layer/model boundaries, declared quality limits |
| Attention or state layout | long-context boundaries, snapshot/restore, fresh-versus-reused state |
| Sampling/speculation | distributions, proposal/rejection/residual draws, seeded replay, EOS commit |
| Scheduler/server | direct-versus-HTTP behavior, independent batched requests, multi-turn reuse, cancellation, failure and shutdown |
| Performance | affected quality gates followed by one comparable release measurement |

Broaden testing when a change crosses shared ownership boundaries or a failure
reveals an unresolved risk. Full model sweeps, media generation, and repeated
timing runs are not the default development loop.

Model-owned procedures and outstanding qualification gaps live with the model:

- [DeepSeek V4 Flash](models/deepseek-v4-flash/QUALITY.md)
- [Qwen3.8 27B](models/qwen3.8-27b/QUALITY.md)
- [Qwen3.8 Flash-Next](models/qwen3.8-flash-next/QUALITY.md)
- [MiniMax H3](models/minimax-h3/QUALITY.md)
- [Qwen3-TTS](models/qwen3-tts/QUALITY.md)
- [Qwen3-ASR](models/qwen3-asr/QUALITY.md)
- [EmbeddingGemma 2](models/embeddinggemma-2/QUALITY.md)
- [Qwen-Image-2.1](models/qwen-image-2.1/QUALITY.md)

## Independent references and artifacts

Prefer hand-checkable analytic fixtures or the source checkpoint in a pinned
official runtime. A promoted Gufo build is a regression reference; agreement
with it alone does not establish upstream correctness. Record checkpoint
storage type separately from compute and accumulation precision.

Unchanged-arithmetic refactors retain exact boundaries. Reassociated kernels
need declared numerical limits and model-level checks. Quantization is
evaluated against a full-quality teacher with distribution, perplexity, and
capability gates. Do not loosen tolerances or silently replace goldens to
admit a change.

Golden artifacts record immutable model and oracle revisions, file hashes,
tokenizer/template identity, prompt/token-stream hashes, dtypes, parameters,
and schema version. Large tensor/logit dumps stay outside Git. Keep small
independent fixtures and concise results needed to reproduce qualification;
see [benchmark artifact retention](benchmarks.md).

## Matched-token and layer comparisons

Tokenize once and feed the same predetermined history to teacher and
candidate. Compare full next-token distributions at each position before
appending the next evaluation token. Free-running text diverges after one
different choice and cannot isolate numerical or quantization error.

Relevant metrics include non-finite counts, normalized-logit error, KL,
teacher-token NLL/perplexity, top-1 agreement, and top-k overlap. See
[benchmark quality methodology](benchmarks.md#quality-method-matched-token-per-position).
For a discrepancy, capture the earliest changed embedding, normalization,
attention/recurrent state, routed expert, layer output, or LM-head boundary.
Preserve the failing input and add a focused reproducer before fixing it.

Pin sampling controls, seeds, cache history, and batching when testing replay.
Run longer multi-turn cases when session reuse or EOS handling changes.
Do not average away numerical mismatches or classify device failures as passes.

## Performance evidence

Compare identical model artifacts, prompt/output counts, context depths,
batching, sampling, and cache state. Measure request latency and aggregate
throughput separately. Run baseline and candidate close together; add
repetitions only when noise or a suspected regression requires them.
Record hardware/software identities and the exact timed scope. Use
[profiling tools](performance.md) to explain a gain before retaining it.
