---
name: optimize-kernel
description: "Workflow for optimizing model inference kernels on Strix Halo gfx1151: assess a baseline, change one route behind a policy toggle, verify quality against the baseline, and retain or reject with evidence. Rejection with evidence is a valid card completion."
metadata:
  origin: strix-halo.cpp
---

# Optimize Kernel (Agent Skill)

Workflow for GPU kernel optimization cards (gh issue opt-*). Never promote a
fusion that degrades end-to-end layer time or spills scratch. Rejection with
evidence is a valid card completion.

## Card contract (read the issue first)

- The issue's `## Acceptance` checklist is the contract (e.g. #127 requires
  "any active-layer degradation from bus contention rejects/disables
  prefetch"). The `## Verify` snippet may name stale presets; the real ones
  are `gpu-test` (build) and `gpu-full` (test).
- Register the equivalence test under the card's CTest label on the focused
  Qwen HIP target that owns the kernel family in `CMakeLists.txt` (attention,
  SSM, FFN, quant GEMV, dequant, graph/prefetch, module, or basic/BLAS). Add a
  new focused target rather than growing an unrelated executable.

## Task scratch file

Keep an evidence file at repo root (`task-on-going.md`); it is the only
artifact that survives context. Record baselines, profile findings, tool
gotchas, and ideas as they are discovered. DELETE it before finishing - it
only makes sense during the optimization.

## 1. Assess the baseline (before any change)

1. Clean stage state: `git add .` (Nix sees only tracked files).
2. Probe hardware + record revision/fingerprint: `nix build && ./result/bin/strix`.
3. Build:
   - Release: `nix build` (produces `result/`). It builds fully-optimized
     binaries and `./result/bin/strix-server` runs significantly faster with
     consistent performance.
   - Fast incremental loop while editing:
     `nix develop -c cmake --build --preset gpu-test`.
   - Build after EVERY step that edits kernels or launchers; never batch
     several uncommitted kernel edits before compiling this allows to iterate faster and safer.
4. Benchmark with `./result/bin/strix-server bench`, single reps (do NOT pass
   `--repetitions`; alternate baseline/candidate runs instead):
   - Combined headline: `-p 2048 -n 128`
   - Decode only: `--n-prompt 0 --n-gen 128`
   - Prefill only: `--n-prompt 2048 --n-gen 0`
   - Depth scaling: `--n-prompt 2048 --n-depth 0,4096,8192,16384`, plus
     `--n-gen 128 --n-depth 0,4096,8192,16384` for tg. Read
     `src/server/bench_cli.cpp` to interpret values before trusting them.
   - Until the implementation works do NOT test with big numbers, focus on `--n-prompt 128 --n-gen 16`
5. Logits comparison (correctness before performance counts):
   `--validate-prefill 1024 --n-prompt 1024 --n-gen 0`; record the envelope
   (current model: rmse `0.02007260`, cosine `0.99997753`, top-1 `198`). A
   candidate must not regress it. Comparing logits can be extremely slow, do it ONLY at the end to verify end to end correctness of the change.
6. Profiling (separate pass; tracing changes timing so never profile the
   headline run):
   ```
   nix develop -c rocprofv3 \
     --kernel-trace --scratch-memory-trace --summary \
     --output-directory /tmp/strix-profile-<tag> \
     -- ./result/bin/strix-server bench --model "$MODEL" --n-prompt 0 --n-gen 128
   ```
   Collect per-kernel launch count, elapsed time, memory traffic, and resource
   table (VGPR, SGPR, LDS, scratch, occupancy, waves/SIMD). Headline latency
   always comes from an unprofiled run.

Record the baseline artifact in `task-on-going.md` (fingerprint, revision,
raw samples, register allocation, scratch, occupancy) before touching
code.

## 2. Introduce the change

- One fused kernel / one route at a time, behind a new policy toggle in
  `src/core/hip/detail/qwen_attention_policy.hpp` (e.g.
  `ShouldFuseSSMGateResidual`, `ShouldPrefetchNextLayer`), default OFF. The
  unfused route stays wired as the independent reference (comparison and
  revert); a rejected route remains runnable behind the toggle.
- Reuse existing kernel patterns in `src/core/hip/` (and llama.cpp CUDA
  kernels in `llama.cpp/` as reference sources). Copy-paste the pattern into
  that model's own implementation; do not modularize shared code. Only the
  model being optimized may change.
- Kernel must sustain >= 4 waves per SIMD and no scratch spilling to be
  retained; check `__launch_bounds__` and resource usage.
- When editing `__global__` kernels, re-balance braces before compiling:
  count `{`/`}` per line skipping `//` and `#` lines. A stray `}` closing the
  namespace early surfaces as `use of undeclared identifier '<KernelName>'` at
  the launch site (secondary cascade); a missing `}` surfaces as `function
  definition is not allowed here`. Fix the FIRST structural error and rebuild
  before chasing later ones.
- Register the equivalence test under the card's CTest label.
- Graph-capture awareness (decode): if the change touches launch structure,
  verify capture still succeeds with
  `STRIX_DISPATCH_TELEMETRY=1 ./build/gpu-test/strix-server bench -p 16 -n 16`
  -> expect `hip_graph` `miss_captured` then `hit`. Cross-stream rules:
  - Only the side-stream-records -> main-stream-waits direction is
    capture-compatible; the reverse fails `hipStreamEndCapture`
    (`miss_end_failed`).
  - Work launched on a NON-captured side stream during capture is NOT part of
    the captured graph (only event edges are baked in); on replay the side
    kernels never run. Verify with rocprof that new work actually dispatches
    per token, or it only fires during capture.
- Record new ideas in `task-on-going.md` as they are discovered (and the
  model's `benchmarks/<model>/README.md` `# TODOs` dotted list at the end).

## 3. Verify against the baseline

1. Correctness gate first:
   ```
   nix develop -c cmake --build --preset gpu-test
   nix develop -c ctest --preset gpu-full --output-on-failure --no-tests=error -L 'opt-c0XX-...'
   ```
   Fused output must match the unfused reference under the arithmetic contract
   (CPU oracle vs GPU kernel).
2. Logits: same `--validate-prefill` as baseline; require finite logits,
   identical top-1, no material envelope regression.
3. End-to-end interleaved A/B with release binaries. Keep
   `result-base`/`result-cand` symlinks and alternate binaries:
   ```
   val=$(./result-<tag>/bin/strix-server bench --model "$MODEL" --n-prompt 0 --n-gen 128 2>/dev/null \
     | rg '\| *tg128' | sed -E 's/.*\|\s*([0-9.]+) ±.*/\1/')
   ```
   Depth rows are named `tg128@d4096` etc. Report medians, tail, and raw
   samples. The host is noisy (background sidekiq/agy spikes produce pp2048
   outliers ~150-300 tok/s), so a signal is real only if it reproduces across
   interleaved samples outside the noise band. Include `pp2048` in the same
   runs as an "untouched-path" sanity check.
4. Always A/B BOTH paths: decode silently switches to split-K (non-graph) at
   context >= 4K, and a regression can hide in one path only. Test `tg128`
   (graph) and `tg@depth 4K/8K/16K` (split-K); to force the non-graph path at
   shallow depth run `STRIX_ENABLE_HIP_GRAPH=0`.
5. rocprofv3 with the same command as baseline; compare launch count,
   eliminated memory traffic, layer latency, VGPR/LDS/scratch, occupancy,
   waves/SIMD, and raw repetitions (sqlite query below).
6. Decision:
   - Retain only if end-to-end evidence improves beyond noise under the
     acceptance contract AND the kernel meets resources (>= 4 waves, no
     scratch). If retained, flip the toggle on and consider making the route
     default.
   - Reject (valid completion) if neutral-to-regressive or any active-layer
     degradation; leave the toggle OFF and the code + test behind it for
     re-evaluation.
7. Full gates only on the final tree, right before wrap-up:
   - clang-format edited files first:
     `nix shell nixpkgs#clang-tools -c clang-format -i <edited .cpp/.hpp>`
   - `nix build .#checks.x86_64-linux.pr`
   - `nix flake check`

## 4. Wrap up (evidence-based decision)

1. Update `benchmarks/<model>/README.md` existing sections only: add a row to
   the Experiment Summary table (retained/rejected) and append follow-up ideas
   to the `# TODOs` dotted list.
2. Post the decision to the issue with `gh` and close it if done (rejection is
   valid). Comment format that worked across cards:
   - `## Result: REJECTED/...` header with the one-line conclusion
   - Correctness (CTest label passing; validate-prefill envelope)
   - rocprofv3 kernel table (VGPR/SGPR/LDS/scratch/wgs/grid + notes)
   - Interleaved A/B raw samples + medians, memory placement
   - Mechanism / why
   - Decision sentence referencing the card contract
3. Commit locally: `jj commit -m "perf(hip): evaluate <change> (#<issue>)"`,
   then move the `main` bookmark to the commit and to NOTHING else
   (`jj bookmark move main --to <commit-id>`; the bookmark lives on the real
   commit, never on the empty working copy). Do not push.
4. Delete `task-on-going.md` and squash the deletion into the card commit
   (`jj squash --into <card-commit>`).
5. Re-run the full gates on the amended tree, then send the completion
   notification if the user's convention uses one (e.g. ntfy).

## Platform knowledge (Strix Halo gfx1151, Qwen3.8-27B BF16)

- Weights are `mmap(PROT_READ, MAP_PRIVATE)` + `madvise(MADV_SEQUENTIAL)` then
  `hipHostRegister(Mapped|ReadOnly)` via `MapRegisteredRegion` (mapped mode is
  the default on the integrated APU). The shard is fully page-cached, so
  steady-state decode is DRAM-bandwidth bound (~85 GB/s re-reading ~10-24 GiB
  of weights per token). Page-touch prefetch only re-reads the same DRAM.
  `STRIX_GPU_WEIGHT_MODE=copy` makes weights device-resident (slow H2D load).
- Decode hot kernels (tg128): `Wave32FusedSwiGLUGEMV_2Rows`,
  `FastGEMVBlockKernel`, `Wave32GEMVKernel_1Row`,
  `Wave32FusedSSMInputProjections`, `DeltaNetRecurrenceKernel`.
- Decode attention: online softmax below 4K; split-K (non-graph) at 4K+.
  HIP-graph capture runs only when `!use_split_k_decode`.
- Reference: `llama.cpp/` contains llama-bench on the same model; the
  remaining gap is ~1.07-1.09x decode and ~1.15-1.34x prefill at depth (see
  `benchmarks/qwen3.8-27b/README.md`).
- Baseline envelope: pp2048 ~335-341, tg128 3.72-3.74,
  validate rmse `0.02007260` / cosine `0.99997753` / top-1 `198`.
