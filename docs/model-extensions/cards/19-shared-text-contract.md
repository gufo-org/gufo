# Card 19: Finalize the shared text-generation contract

Status: **pending**. RFC step and dependencies: **10 shared text; 18**.

Future scope; do not execute during the cards 01–14 milestone.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

The serving split provides a draft seam; shared text migration now fixes its public contract. Core owns scheduling/cache/client envelopes and providers own formatting, execution, syntax and option values. The unresolved JSON dialect fallback must be decided and qualified.

### Original scope

Finalize public runner contract versus pool/cache policy split, optional diagnostics/capabilities and experimental grammar-builder, provider-valued options, neutral inputs and tool syntax. Preserve schema helpers/client envelope ownership and apply spike decisions. Shared four profiles, long-context/cache, forced tool choice, goldens, matched per-phase timings, default/lock dialect behavior.

## Implementation plan

1. Finalize public step-level runner versus pool/cache policy types, neutral input/continuation contracts, optional diagnostics/capabilities and defaults in gufo_ext_runtime.
2. Publish experimental grammar-builder and tool-syntax contracts for grammar construction, streaming marker scanning, parsing and reasoning separation; keep schema helpers and client envelope closers host-owned.
3. Apply SPIKE.md driver/provider decisions and recorded canonical draft-option naming without aliases; implement provider-valued --speculative plus selected-provider validation.
4. Resolve active-model-only versus all-built-dialect JSON fallback with explicit default/lock forced-tool-choice expectations. Coordinate cache RFC sequencing and retain native snapshots.
5. Update API SURFACE/version/CHANGES and tests without creating mandatory virtual methods for optional new features.

## Definition of done

- [ ] Contract cleanly separates model primitives from scheduler/pool/cache policy and optional features have compatible defaults.
- [ ] Tool-syntax fallback and provider-valued options have explicit tested behavior, including lock selection differences.
- [ ] All four profiles, exact identities, actual drafts, continuation and matched per-phase timings satisfy shared gates before model migrations start.

## Validation criteria

- Run shared runner/grammar/parser/options unit tests plus all four text profiles using explicit affected suites, forced tool choice, long-context and cache in tests/functional/README.md.
- Run exact identity goldens and independent numerical checks; inspect native DeepSeek continuation behavior and real speculative counts rather than requested settings alone.
- Qualify HTTP and CLI controls with matched histories on default and actual lock HIP artifacts. Run API/SURFACE/isolation checks and policy fixtures; preserve inconclusive timings and fix reproduced regressions.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
