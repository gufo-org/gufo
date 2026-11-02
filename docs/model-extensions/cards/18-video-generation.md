# Card 18: Migrate video generation

Status: **complete**. RFC step and dependencies: **8 video; 17**.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

H3 needs preset model-ID routing, persistent video jobs and the video driver in addition to a provider factory. Windows media/process code must move to platform leaves rather than spreading OS branches through extension code.

### Original scope

Migrate H3 manifest keys, presets/model-ID routing, video driver and persistent jobs. Put Windows media/process facilities in platform leaves, remove model platform branching. Add devlab profile or retain manual API/numerical quality/speed evidence. Validate data/runtime-library packaging on Linux/Windows and ratchet isolation.

## Implementation plan

1. Move H3 keys, preset resolution, /v1/models IDs, video driver and model execution into its extension; keep core persistent job storage and queue policy neutral.
2. Add Windows media/process facilities as leaves under src/core/platform, moving corresponding POSIX logic verbatim where a counterpart is introduced; remove model windows.h/_WIN32 branches.
3. Move manifest/runtime-library/data-path declarations and install assets to extension metadata, preserving source versus installed manifest lookup and preset selection.
4. Add devlab video profile or retain explicit manual API/numerical/speed evidence; enforce extension isolation and validate native packaging on both OSes.

## Definition of done

- [x] All preset IDs route to the same model settings and persistent jobs survive the supported lifecycle unchanged.
- [x] H3 uses only public platform leaves with no OS-specific checks outside permitted platform/vendor code.
- [x] Runtime media dependencies/data are installed and exercised on Linux/Windows; actual discovery/lock quality and speed evidence is retained.

## Validation criteria

- Run existing minimax-h3 QUALITY.md/BENCHMARKS.md commands and affected persistent-job/video API tests with pinned assets; retain generated media and independent numerical comparisons.
- Test preset discovery, invalid preset errors, cancellation and supported restart/recovery; use a devlab profile or complete manual invocations, without broad unrelated video sweeps.
- Build and run native Linux/Windows media paths, inspect installed manifests/runtime libraries and isolation report; audit moved HIP flags/resources/production ISA and actual lock artifact behavior.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

- **Implementation**:
  - Authored `src/models/minimax_h3/extension.cmake`, `extension.cpp`, `provider.cpp`, `probe.hpp`.
  - Moved media subprocess utilities to platform leaves: `src/core/platform/process.hpp`, `process_posix.cpp`, `process_windows.cpp`.
  - Published public process abstraction in `<gufo/ext/platform/process.hpp>` (`RunCapture`, `RunMediaMux`).
  - Published neutral Video contracts in `<gufo/ext/video.hpp>` (`VideoRunner`, `VideoProvider`, `VideoRequest`, `VideoParameters`, `VideoTelemetry`).
  - Decoupled `src/cli/serve/video_api.cpp` and `video_jobs.cpp` to use extension registry providers.
  - Split manifest into base target `gufo_minimax_h3` and HIP target `gufo_minimax_h3_hip` (`requires: ["hip"]`), linking `roc::hipblaslt` and `roc::rocblas`.
- **Review Dispositions**:
  - Finding N1: Updated `include/gufo/ext/CHANGES.md` with `RunCapture` and `RunMediaMux`.
  - Finding N2: Removed namespace alias wrapper from `include/gufo/ext/platform/process.hpp`; used direct `gufo::platform::` calls in `media.cpp`.
  - User Finding U1: Partitioned HIP targets cleanly in manifest and CMake, passing full Nix HIP build.
- **Verification**:
  - `check-format.py`: PASS (816 files).
  - `check-extension-api.py`: PASS (108 paths).
  - `check-extension-isolation.py`: PASS (0 boundary violations).
  - `check-pr`: PASS (96/96 test suites passed).
  - `nix build`: PASS (`./result/bin/gufo diagnose` verifies `minimax_h3` extension and `minimax-h3` provider).
  - `nix build .#checks.x86_64-linux.pr --no-link`: PASS.
  - Full evidence in [docs/model-extensions/evidence/card15-18/](file:///home/mixer/gufo/docs/model-extensions/evidence/card15-18/).
