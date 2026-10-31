# Card 17: Migrate image generation

Status: **pending**. RFC step and dependencies: **8 image; 16**.

Future scope; do not execute during the cards 01–14 milestone.

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

- [ ] Image requests, job status/cancellation/errors and supported artifact selection preserve existing behavior.
- [ ] Manifest/family dependencies and installed assets are complete on Linux and Windows.
- [ ] Actual model API, numerical quality and standard speed evidence exists for discovery and lock HIP builds; absence of a devlab profile is explicitly covered.

## Validation criteria

- Execute the existing qwen-image-2.1 QUALITY.md and BENCHMARKS.md workload/commands using pinned inputs and candidate production artifacts, retaining outputs and independent quality references.
- Run affected API/job-queue tests and the new profile if implemented; record manual invocations and per-request/job timing controls otherwise. A profile schema test alone is insufficient.
- Build/test Linux/Windows, excluded-image core tools and pure HIP-free artifact probes; inspect family dependencies and run isolation/resource gates where code moved.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
