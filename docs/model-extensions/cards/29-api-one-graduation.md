# Card 29: Graduate the extension API to 1.0

Status: **pending**. RFC step and dependencies: **12 graduation; 28**.

Future scope; do not execute during the cards 01–14 milestone.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

API 1.0 depends on measured elapsed time and real additive serving-feature evolution, not finishing the migration code or counting release commits. This future card cannot be accelerated by simulating dates.

### Original scope

Require actual 60-day stable text header and pilot external+CMake/Nix period, at least five merged serving features affecting API with only additive revision bumps, all public-only in-tree models and external text fixture compatibility. Audit all 1.0 criteria, then graduate/publish policy. Experimental hooks require explicit reviewed graduation after their own 60 unchanged days. Do not simulate elapsed time or count non-API commits. Binary loading remains deferred until separate demand and 90-day stable-API criteria.

## Implementation plan

1. Audit every RFC 1.0 criterion using Card 28 dated source/build/conformance evidence and the current tree; require every in-tree model to use public API only.
2. Verify the stable text header has at least 60 actual unchanged-compatible days and at least five merged serving features affecting the API with only additive revision bumps; exclude non-API commits.
3. Verify the external pilot builds unchanged from source/CMake and Nix for the same period, pinned external text compatibility, second provider per capability and enforced experimental-symbol policy.
4. Only after all criteria are proven, review/publish API 1.0 source-compatibility, versioning and previous-major adapter/deprecation policy. Graduate each experimental hook explicitly after its own 60 unchanged days.
5. Leave binary loading deferred unless separate concrete demand and at least 90-day stable API criteria are independently satisfied; do not implement it as part of this card.

## Definition of done

- [ ] All actual time, additive-feature, public-only, external compatibility and second-provider requirements are demonstrated with dated authoritative evidence.
- [ ] API 1.0 publication/version/policy occurs only after the audit passes; no feature or day is invented to close the card.
- [ ] Experimental hooks remain experimental until separately reviewed and eligible; the guarantee remains source compatibility after rebuild, never binary ABI.

## Validation criteria

- Audit commit/merge history and release/compatibility ledger dates against pinned fixture builds, checking the API effect and version bump for each of at least five counted features.
- Run public-surface isolation and external text/pilot CMake/Nix compatibility builds plus capability conformance, and verify the entire real 60-day period rather than one current success.
- Inspect published policy/version/CHANGES and gate behavior for incompatible majors/adapters. If elapsed time or independent evidence is missing, retain pending status and name the exact unmet criterion.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
