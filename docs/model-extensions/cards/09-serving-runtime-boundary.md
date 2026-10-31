# Card 09: Separate serving policy from extension execution

Status: **pending**. RFC step and dependencies: **6; 08**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Models must use neutral execution contracts while HTTP semantics, scheduler, pool and retention policy stay core-owned. This split coordinates with the cache RFC and preserves both component-cache and native-snapshot continuation behavior.

### Original scope

Extract gufo_serving and extend gufo_ext_runtime with neutral runner/chat/tool/tokenizer/reasoning/continuation types, default HIP probing, host budgets, cache backing/resource contracts and injected logging/budget callbacks. Keep HTTP semantics/scheduler/pool/cache policy in core. All four profiles, long-context/cache, native DeepSeek snapshots, exact goldens and matched timings.

## Implementation plan

1. Extract gufo_serving from currently executable-owned scheduler, runner pool, component text cache and InferenceBackend sources; retain gufo_http as a separate consumer.
2. Extend gufo_ext_runtime with neutral runner/chat/tool/tokenizer/reasoning and continuation state/snapshot bases plus default HIP probing; remove concrete model types from those contracts.
3. Move host snapshot budget queries and neutral backing/resource contracts to runtime/cache owners; place CreateHipComponentCacheResources next to gufo_cache_hip with injected budget/log callbacks.
4. Keep queueing, API envelopes, cache retention/admission and policy in serving/core. Migrate consumers and add focused dependency/behavior tests before publishing the larger surface in Card 10.

## Definition of done

- [ ] Models/runtime/cache compile without serving or HTTP implementation headers.
- [ ] Shared runner step/batch, cancellation, publication waits, budgets and snapshot/resource semantics are unchanged.
- [ ] Exact identities and all four profile continuation behaviors are proven, including native DeepSeek snapshots; no cache cutover is inferred.

## Validation criteria

- Build all affected runtime/cache/serving/HTTP targets and run focused unit tests chosen from ctest --preset gpu-full -N; check link dependencies for forbidden serving cycles.
- Run all four text profiles with explicit long-context/cache and affected scheduler/tool suites using tests/functional/README.md. Inspect actual loaded modes/drafts, restore/prefill work and local DeepSeek persistence.
- Run exact identity goldens, affected numerical quality and matched HTTP/CLI timing controls; retain default and actual lock-mode production artifacts separately and fix confirmed regressions.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
