# Card 24: Prove capability extensibility with second providers

Status: **pending**. RFC step and dependencies: **11; 23**.

Future scope; do not execute during the cards 01–14 milestone.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

A migrated provider only proves its own interface fits. Every capability needs an independent second implementation to demonstrate that an existing-capability model can be added without touching core, build or registry source lists.

### Original scope

Add genuinely external test-only second providers for every capability (or genuine new models), run conformance and prove adding an existing-capability model changes only its own directories plus allowed generated output/lock release decision. Include zero core/build/preset/workflow/Nix/registry source changes. Cover ambiguity, excluded keys, provider override and HIP-free probes; builds exclude any single extension.

## Implementation plan

1. Create external test-only second providers for text, embeddings, ASR, TTS, image and video, or use genuine new model ports; do not clone migrated model internals as the contract oracle.
2. Run each capability conformance suite, covering advertised IDs, request validation, step/stream/cancellation contracts and capability-specific scheduling boundaries.
3. Exercise pure HIP-free probes, ambiguous matches, explicit provider override and discovered-but-excluded key diagnostics.
4. Record the exact addition diff: only extension directories plus allowed generated output and explicit lock release decision may change; test every single-extension exclusion against core/tools.

## Definition of done

- [ ] Every capability has a genuinely external second provider with retained conformance evidence.
- [ ] Adding those providers requires zero core/build/preset/workflow/Nix/registry source edits.
- [ ] Selection ambiguity/overrides/exclusions and HIP-less probes behave deterministically; all any-single-extension-excluded core/tool builds pass.

## Validation criteria

- Build external fixtures in clean source trees and run to-be-added capability conformance tests; retain a capability-by-provider matrix and concrete response/stream assertions.
- Compare jj diffs/source inventories before and after each addition, classifying only allowed generated/lock changes; a manually patched core invalidates the proof.
- Configure/build the full exclusion matrix and run selection/probe negative fixtures without HIP hardware. Stub conformance establishes interface behavior, not real-model numerical quality or speed.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
