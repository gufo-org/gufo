# Card 23: Support external extension source trees

Status: **pending**. RFC step and dependencies: **11; 22**.

Future scope; do not execute during the cards 01–14 milestone.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

An extension is genuinely external only when it builds without an in-tree copy or Gufo source edits. Configure-time include policy must account for inherited flags/root include paths, and family-library stability scope must be resolved first.

### Original scope

Implement external directories, configure-time allowlist (including inherited directory flags/root includes), Nix extensions override and template repository. Add external registry fragment schema/merge support. Externalize pilot and remove its in-tree copy from both raw and Nix source trees for proof; build/test with zero Gufo diff. Add Windows reusable-workflow repository/revision inputs and template third-party ZIP recipe.

## Implementation plan

1. Implement GUFO_EXTENSION_DIRS discovery and configure-time public include allowlist for external targets, including inherited directory flags and root include escapes; record family dependency policy.
2. Add experimental Nix extensions override and validated external functional registry fragment schema/merge support without changing the public existing flake contract accidentally.
3. Create a template repository with source/CMake/Nix/tests and a third-party Windows ZIP recipe; extend reusable Windows workflow inputs for repository/revision selection.
4. Externalize the EmbeddingGemma pilot, physically remove its in-tree copy from raw and Nix-filtered trees for proof, and build/run its suites with zero Gufo diff.
5. Add rejection fixtures for private/serving/model includes, conflicting fragments and missing dependencies; document arbitrary native code trust.

## Definition of done

- [ ] External pilot source/CMake/Nix builds and tests pass without hidden in-tree code, Gufo modifications or root-include escape.
- [ ] Allowlist checks reject prohibited dependencies even through inherited compilation settings.
- [ ] Template and Windows inputs are usable at pinned revisions; fragment merging validates and preserves deterministic selection/provenance.

## Validation criteria

- Build the pilot from a clean Gufo tree with its in-tree source removed and the external directory passed explicitly, then repeat with Nix extensions; retain source inventories, diff proof, compile commands and binary provenance.
- Run the pilot full API/numerical/speed suites from that external artifact and focused to-be-added allowlist/registry-merge fixtures; do not treat the template build alone as behavioral proof.
- Execute the pinned repository/revision Windows workflow and ZIP recipe, inspect bundled assets, and verify no external include reaches private paths via inherited -I or compiler flags.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
