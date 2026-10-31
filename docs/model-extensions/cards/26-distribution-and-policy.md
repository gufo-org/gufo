# Card 26: Complete distribution recipes and conformance policy

Status: **pending**. RFC step and dependencies: **distribution/conformance; 25**.

Future scope; do not execute during the cards 01–14 milestone.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Source extensions execute arbitrary native code and distribution spans CMake, Nix, Windows and external Toolboxes images. Repository recipes alone cannot prove builder/runtime image publication or edge lock integration.

### Original scope

Publish source/CMake/Nix/Windows recipes, arbitrary-native-code trust and tier policy, capability conformance suites and diagnostics provenance. Track Toolboxes builder/runtime images and lock-mode edge integration as an external deliverable; retain actual evidence or leave pending (repository-only changes do not prove images). Verify packaging/install/licensing and all generated-output drift gates.

## Implementation plan

1. Publish source/CMake/Nix/Windows build/install recipes, capability conformance invocation and diagnostic provenance guidance, with explicit native-code trust assumptions.
2. Resolve terminology, community in-tree acceptance and abandoned-maintainer policy; distinguish core release gates from community conformance/evidence expectations.
3. Verify data/runtime libraries/licenses and generated output across supported package paths, including third-party recipes.
4. Track Toolboxes builder/runtime images and rolling edge lock-mode integration as an external deliverable with image revisions/digests and actual runtime results; leave incomplete external work pending.

## Definition of done

- [ ] Recipes are reproducible, licensing/install data is complete and arbitrary-code/tier policy is explicit.
- [ ] Conformance suites and provenance diagnostics are documented for each capability and distribution path.
- [ ] Toolboxes builder/runtime and edge integration have actual external evidence or the card stays pending; no repository-only completion claim.

## Validation criteria

- Execute documented source/CMake/Nix/Windows recipes in appropriate clean environments, inspect installed binaries/data/runtime libraries/licenses and run diagnose/provenance plus capability smoke checks.
- Run all generated-output drift gates and capability conformance suites, including external template consumers.
- Inspect published Toolboxes image digests/build logs and run lock-mode runtime checks against those images; retain external identifiers and outputs, not only links to proposed recipes.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
