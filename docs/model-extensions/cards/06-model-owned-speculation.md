# Card 06: Move model-specific speculation out of core

Status: **pending**. RFC step and dependencies: **4; 02, 05**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Core speculation currently knows Qwen implementations. Move those algorithms to their owner while keeping core sampling/constraint contracts neutral. The open cache RFC owns its cutover; native snapshots remain supported until the all-model replacement is proven.

### Original scope

Move Qwen-specific speculative code out of core; neutralize draft tokenizer types and preserve rollback sampler/constraint state. Coordinate cache RFC and keep native snapshots. All four profiles plus long-context/cache (DeepSeek locally), identity goldens and matched timings.

## Implementation plan

1. Inspect core speculative sources, draft tokenizer types and serving callers against the current cache RFC state; record ownership and sequencing before moving interfaces.
2. Move Qwen-specific speculation to model/family-owned code and replace tokenizer dependencies with neutral contracts. Keep core prompt lookup as a host primitive.
3. Preserve sampler, RNG and constraint copies on speculative verification/rollback, cancellation and stop handling; retain native snapshot and persistence paths including DeepSeek.
4. Ratchet the moved boundary, update dependencies and API policy where public contracts change; run exact identities and affected shared text histories before committing.

## Definition of done

- [ ] Core speculative code has no Qwen implementation dependency and the neutral tokenizer contract is sufficient for actual modes.
- [ ] Rollback preserves sampler/constraint/RNG semantics and all existing native continuation paths remain reachable.
- [ ] All four text profiles, affected long-context/cache histories and matched phase evidence are retained; confirmed regressions are fixed.

## Validation criteria

- Run the existing identity target/CTest selection from Card 02 and affected speculative unit tests discovered with ctest --preset gpu-full -N; require exact bytes and no model skips.
- Use tests/functional/run.py and compare.py as documented in tests/functional/README.md with all four profiles, explicit affected suites including long-context and cache, actual drafts and fresh output directories. DeepSeek uses verified local models/deepseek-v4-flash target plus sidecar from HANDOFF.md.
- Run existing numerical quality commands and standard speed benchmarks from the affected model QUALITY/BENCHMARKS documents; qualify default and actual lock-mode HIP artifacts separately. Apply shared per-request/phase controls, never pooled averages.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
