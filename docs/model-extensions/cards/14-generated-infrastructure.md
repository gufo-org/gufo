# Card 14: Generate infrastructure from model metadata

Status: **pending**. RFC step and dependencies: **9; 13**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Before remaining implementations move, every model needs metadata so generic tooling can replace hand-maintained lists. Metadata-only registration must remain truthful while legacy code is still linked through old paths.

### Original scope

Add metadata-only manifests for every remaining model before its code moves. Generate hosted/Windows target lists, labels, ZIP devlab registry input, Nix filters/packages/install checks/devshell/reference/check fragments, CI resource/vendoring data, docs/notices and committed functional registries with drift checks. Preserve public flake check names and test-source production-derivation independence. Hosted checks and pilot-selected devlab run.

## Implementation plan

1. Inventory every remaining model and its existing target/test/fixture/package/resource/docs/functional declarations; add metadata-only extension.json files without pretending unmigrated providers already use factories.
2. Generate bounded hosted and Windows target lists, CTest labels and ZIP devlab.json registry input; preserve BUILD_TESTING/platform/HIP conditions and current bundled-test assets.
3. Generate Nix production filters, dependency/install declarations, extension.nix imports for devshell Python/reference/check fragments, keeping public check names and test-source derivation independence.
4. Replace tools/ci model resource/vendoring tables, docs/notices indexes and committed functional registries with manifest-derived output plus drift checks. Add fixtures for missing/duplicate/stale/conditional metadata.
5. Run hosted checks and a pilot-selected devlab run, review all generated diffs, and stop the authorized implementation milestone after this card is verified and committed.

## Definition of done

- [ ] Every current model has accurate metadata before its code migration; generated consumers no longer need parallel model lists.
- [ ] Windows ZIP uses generated registry data; hosted tests/labels and Nix output names remain compatible.
- [ ] Editing model test/functional/tool sources leaves production derivations unchanged while metadata/production edits are admitted; generated-output drift fails checks.
- [ ] Pilot-selected devlab evidence and bounded hosted checks are retained, with discovery/release distinctions intact.

## Validation criteria

- Run nix build .#checks.x86_64-linux.pr and affected generator/functional registry fixture tests; generator second runs must produce no diff and deliberate stale fixtures must fail.
- Use nix eval and derivation comparisons to verify public check names, build inputs, install-data/licenses and unchanged production derivation after an isolated test-source-only edit; restore only that owned fixture edit afterwards.
- Build/check generated Windows target/ZIP registry paths on Windows and run the pilot-selected devlab command using the current tests/functional/README.md CLI; inspect real embedding subreports, not only successful scheduling.
- Run task-scoped docs/notices links, resource/vendoring consistency checks and default/lock metadata reports. Retain exact commands, revisions and artifact hashes; do not begin Card 15.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
