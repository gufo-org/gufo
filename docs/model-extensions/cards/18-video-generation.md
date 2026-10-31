# Card 18: Migrate video generation

Status: **pending**. RFC step and dependencies: **8 video; 17**.

Future scope; do not execute during the cards 01–14 milestone.

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

- [ ] All preset IDs route to the same model settings and persistent jobs survive the supported lifecycle unchanged.
- [ ] H3 uses only public platform leaves with no OS-specific checks outside permitted platform/vendor code.
- [ ] Runtime media dependencies/data are installed and exercised on Linux/Windows; actual discovery/lock quality and speed evidence is retained.

## Validation criteria

- Run existing minimax-h3 QUALITY.md/BENCHMARKS.md commands and affected persistent-job/video API tests with pinned assets; retain generated media and independent numerical comparisons.
- Test preset discovery, invalid preset errors, cancellation and supported restart/recovery; use a devlab profile or complete manual invocations, without broad unrelated video sweeps.
- Build and run native Linux/Windows media paths, inspect installed manifests/runtime libraries and isolation report; audit moved HIP flags/resources/production ISA and actual lock artifact behavior.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
