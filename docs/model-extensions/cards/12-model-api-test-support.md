# Card 12: Add in-tree model API test support

Status: **pending**. RFC step and dependencies: **7b; 11**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Model API tests currently reach production CLI/serving internals. An in-tree fixture library can support them without making HTTP testing machinery part of the extension source API.

### Original scope

Add in-tree-only gufo_test_support for HTTP/serving fixtures, remove model-test production CLI includes. Public extension API excludes this support. Run all affected model API tests and report remaining coupling explicitly.

## Implementation plan

1. Inventory model API test includes of src/cli/ and identify reusable HTTP/server lifecycle and serving fixtures.
2. Add in-tree-only gufo_test_support wrapping current gufo_http and gufo_serving fixtures, with explicit test-only dependencies and build conditions.
3. Migrate model API tests to the fixture surface and remove production CLI includes; keep extension production libraries independent of support.
4. Document any remaining test coupling and keep support out of include/gufo/ext/SURFACE and installed extension contracts.

## Definition of done

- [ ] Affected model API tests use test support, with no production CLI includes remaining at the migrated boundaries.
- [ ] Test support is absent from production link/install and is not a public external-extension dependency.
- [ ] Fixture lifecycle preserves request/error/cancellation assertions rather than weakening model API coverage.

## Validation criteria

- List affected API CTests with ctest --preset gpu-full -N and run every migrated model API test with its required weights; missing-model skips remain pending.
- Configure/build with BUILD_TESTING=OFF and inspect production link/install inputs to prove test support is excluded.
- Run isolation inventory for tests and inspect SURFACE to ensure support is private; retain explicit remaining coupling rather than a global clean claim.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
