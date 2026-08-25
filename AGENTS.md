# AGENTS.md

## Supported Platform

Linux x86-64 on AMD Strix Halo (`gfx1151` GPU and XDNA2 NPU) is the only
supported production target.

## Build & Test (Nix)

Build and test with Nix only. Direct host builds and Makefiles are unsupported.

```sh
nix build                          # build default package (gfx1151 + XRT)
./result/bin/strix diagnose        # run hardware probe & diagnostics
./result/bin/strix serve           # run server
nix build .#checks.x86_64-linux.pr # canonical PR test command (all gates)
nix develop                        # dev shell
```

- `git add` before `nix build` — Nix sees only tracked files.
- Use binaries under `build/gpu-test` for focused correctness and debugging
  only; that CMake tree is intentionally unoptimized. Run models and measure
  performance with the release binaries produced by `nix build` under
  `result/bin`. Example:
  `./result/bin/strix bench --model <model.gguf> -p 128 -n 16 --validate-prefill 128`.

## Profiling and kernel work

Performance tooling lives in `tools/` and is documented in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) under "Commands":

- `tools/bench/build.sh` builds the standalone microbenchmarks in `tools/bench/`
  with `hipcc` (seconds, not a full `nix build`); each carries an ablation
  harness that reports correctness next to throughput.
- `tools/bench/gfx1151_peak.hip` measures the roofline ceilings every kernel is
  scored against. Do not use spec-sheet numbers.
- `tools/prof.py` wraps `rocprofv3` with a pipeline-stage rollup, GPU-busy versus
  wall-span, and an A/B `diff` mode.
- `tools/isa_mix.py` summarizes one kernel's instruction mix from an assembly
  listing.

## Development

- Use `gh` CLI to retrieve and update issues content. Verify if installed and configured with `gh auth status`, and use it unless the user explicitly says otherwise.
- Follow Conventional Commits format with single-line commit messages (e.g., `feat(scope): summary (#issue)`, `fix(scope): summary (#issue)`).
- Prefer Jujutsu (`jj`) over Git when possible. Verify that it is available by running `jj version`; fall back to Git if it is unavailable or if the user explicitly requires Git.
