# Card 10: Publish and measure the public extension API

Status: **pending**. RFC step and dependencies: **7; 09**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Forwarding headers can expose a large transitive core/cache closure. Directory names alone do not define a stable API; measure and govern the declarations actually reachable before promising compatibility.

### Original scope

Classify every model-used core header: stable/platform/experimental forwarding API, host wrapper or private. Publish headers and generated transitive `include/gufo/ext/SURFACE`, measure paths/declarations and coupling before committing stability. Enforce changes via version/CHANGES/ext-api gate, including forwarded src/cache headers. Test transitive closure and declaration changes; record the measured surface and any narrowing required.

## Implementation plan

1. Enumerate every core header used by model code and classify it as stable API, platform API, experimental forwarding, Host wrapper or private; record rationale and intended owner.
2. Publish forwarding/contract headers under include/gufo/ext, including platform leaves and experimental contracts, without accidentally exposing serving policy.
3. Generate the full transitive SURFACE using tools/ci/check-extension-api.py --write; measure paths, declarations and coupling, including forwarded src/core and cache declarations.
4. Extend version/CHANGES/ext-api diff tests for transitive additions/removals and forwarded declaration changes; narrow wrappers or explicitly document unresolved coupling before dependent include migrations.

## Definition of done

- [ ] Every inventoried model-used header has a classification and public replacement or approved private removal plan.
- [ ] Measured SURFACE closure is reproducible; forwarded source declarations are governed equally with public headers.
- [ ] Public CMake/schema contracts and API version/changelog gates reject missing required policy changes; no unsupported stability claim is made from wrapper paths alone.

## Validation criteria

- Run python3 tools/ci/check-extension-api.py --write, inspect the resulting closure and then run the check without --write; output must be stable.
- Run python3 tests/tools/check_extension_api_test.py and focused extension schema/configure tests; add to-be-written fixtures for newly classified closure/declaration edge cases.
- Compile representative HIP-less and HIP public-header consumers, record declaration/path counts and confirm no serving/private headers leak beyond the explicitly published closure. Run formatting for changed C++ and both platform builds where affected.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
