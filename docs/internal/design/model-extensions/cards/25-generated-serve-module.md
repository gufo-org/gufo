# Card 25: Generate Nix serving arguments and validation

Status: **implementing**. RFC step and dependencies: **11b; 24**.

Implementation authorized after the cards 01–14 milestone.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

mkServe still has hand-maintained per-model arguments during migration. Metadata must now drive drafts, modalities, speculative values and voices while preserving host module policy and explicit release/default selection.

### Original scope

Generate mkServe model arguments and validation for drafts/modalities/speculation/voices, apply recorded canonical option decision without aliases; use `!` for renamed public flake arguments. Run module evaluation/NixOS serve checks, correct release/default distinctions and external extension configuration.

## Implementation plan

1. Generate model argument definitions/validation from manifests in the existing Nix mk-serve implementation, keeping core host configuration separate.
2. Apply the recorded canonical draft-option decision across CLI-to-module translation with no aliases; renamed public flake arguments require ! release treatment.
3. Validate unsupported provider-specific combinations, voices/modalities/speculation and required draft/model paths before launch.
4. Support external extension metadata and default/discovery versus shipped lock configuration; update module evaluation/serve tests and docs.

## Definition of done

- [x] No per-model draft/modality/speculation/voice lists remain hand-maintained in mkServe source.
- [x] Valid current configurations generate the expected CLI, invalid combinations fail with provider-specific diagnostics, and aliases are absent.
- [x] Public argument changes are documented, external extensions configure successfully and NixOS serving works with correct artifact selection.

## Validation criteria

- Run existing Nix module evaluation checks and NixOS serve tests selected from current flake outputs; add fixtures for every generated option type and invalid combination.
- Use nix eval to inspect representative Qwen/Flash-Next/DeepSeek drafts, multimodal and TTS voice configurations plus an external extension, comparing generated argv to declared canonical options.
- Build/run representative default and release module configurations, inspect provenance and readiness, and verify public unrelated flake check names are preserved.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

### Implementation

- Added manifest-driven Nix metadata discovery, following the selected package's
  source root, external extension roots and discovery/lock selection.
- Generated provider-owned argument names and types; removed hardcoded draft
  paths and speculative mode inventories from `mkServe`.
- Added provider-specific combination checks, required draft paths, voice
  validation and rejection of options for incompatible modalities.
- Recorded the canonical decision: retain the existing distinct draft option
  names without aliases or public argument renames.
- Added rendering/metadata fixtures, default/release NixOS service evaluation,
  CMake/schema checks and a live external-LFM3 systemd/HTTP smoke.
- Added change-record instructions with `AGENTS.md`/`CLAUDE.md` symlinks and
  normalized existing records to descriptive names/headings without assigning
  versions. CMake still derives the revision; `baseline.md` does not count.
- Retained commands, results, initial smoke failure and limitations in
  [Card 25 evidence](../evidence/card25/README.md). User documentation is in
  [Nix serving](../serving.md).

Both production package builds pass, with discovery/development identity for
`default` and lock/release identity for `release`. The implementation gates above
are verified and prepared for pull-request review; independent closing review
remains pending. New external runtime controls still require CLI integration beyond Nix generation;
the existing canonical controls and the external LFM3 provider are covered.
