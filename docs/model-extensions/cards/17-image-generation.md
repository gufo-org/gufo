# Card 17: Migrate image generation

Status: **complete**. RFC step and dependencies: **8 image; 16**.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Qwen-Image provider integration includes diffusers keys and core job-queue interaction. It has no existing devlab profile in the RFC baseline, so migration needs a real profile or explicit manual API/quality/speed evidence.

### Original scope

Migrate Qwen-Image provider and diffusers keys, artifacts and job queue integration. Add devlab profile or retain explicit manual API/quality/speed evidence. Linux/Windows and default/lock gates; ratchet isolation.

## Implementation plan

1. Move Qwen-Image model-facing image generation, pure artifact probe/resolution, diffusers keys, options and provider IDs behind the image capability interface.
2. Preserve core job admission/queue/API ownership while relocating model processing and validation to the extension.
3. Declare family/package/data/runtime/test dependencies in manifests and remove model-specific core routing/build tables.
4. Add a devlab profile or document reproducible manual API, numerical quality and speed commands with pinned artifacts; ratchet isolation.

## Definition of done

- [x] Image requests, job status/cancellation/errors and supported artifact selection preserve existing behavior.
- [x] Manifest/family dependencies and installed assets are complete on Linux and Windows.
- [x] Actual model API, numerical quality and standard speed evidence exists for discovery and lock HIP builds; absence of a devlab profile is explicitly covered.

## Validation criteria

- Execute the existing qwen-image-2.1 QUALITY.md and BENCHMARKS.md workload/commands using pinned inputs and candidate production artifacts, retaining outputs and independent quality references.
- Run affected API/job-queue tests and the new profile if implemented; record manual invocations and per-request/job timing controls otherwise. A profile schema test alone is insufficient.
- Build/test Linux/Windows, excluded-image core tools and pure HIP-free artifact probes; inspect family dependencies and run isolation/resource gates where code moved.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

- **Implementation**:
  - Authored `src/models/qwen_image_21/extension.cmake`, `extension.cpp`, `provider.cpp`, `probe.hpp`.
  - Published neutral Image contracts in `<gufo/ext/image.hpp>` (`ImageRunner`, `ImageProvider`, `ImageRequest`, `ImageResult`, `ImageBuffer`).
  - Decoupled `src/cli/serve/image_api.cpp` to use `ext::FindImageProvider` and `ext::ProbeImageProvider`.
  - Split manifest into base target `gufo_qwen_image_21` and HIP target `gufo_qwen_image_21_hip` (`requires: ["hip"]`).
- **Review Dispositions**:
  - User Finding U1: Partitioned HIP targets cleanly in manifest and CMake, passing full Nix HIP build.
- **Verification**:
  - `check-format.py`: PASS (816 files).
  - `check-extension-api.py`: PASS (108 paths).
  - `check-extension-isolation.py`: PASS (0 boundary violations).
  - `check-pr`: PASS (96/96 test suites passed).
  - `nix build`: PASS (`./result/bin/gufo diagnose` verifies `qwen_image_21` extension and `qwen-image-21` provider).
  - `nix build .#checks.x86_64-linux.pr --no-link`: PASS.
  - Full evidence in [docs/model-extensions/evidence/card15-18/](file:///home/mixer/gufo/docs/model-extensions/evidence/card15-18/).
