# Card 27: Audit every RFC success criterion

Status: **pending**. RFC step and dependencies: **success criteria; 26**.

Future scope; do not execute during the cards 01–14 milestone.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

The final audit must prove the original RFC success criteria rather than infer completion from green narrow checks. Every migrated model, release artifact, build bound and ownership boundary needs direct evidence.

### Original scope

Final requirement-by-requirement audit: zero prohibited coupling and model knowledge, any-single-extension-excluded core/tool builds, clean/incremental build bounds, every model's actual numerical/API/performance evidence, CLI benchmark continuity, unchanged cache goldens, diagnostic provenance, unshipped discovery and shipped HIP lock artifact evidence. Fix confirmed regressions before closing.

## Implementation plan

1. Build a requirement-to-evidence matrix for all RFC goals/success criteria, cards and approved decisions; classify each as proven, contradicted, incomplete or missing.
2. Audit include and compiler/symbol isolation, declared family dependencies and model knowledge in core/build/CI/Nix/registries, exempting only generated output and explicit release decisions.
3. Measure clean production build overhead against matched main and incremental model-touch rebuild scope; run every single-extension exclusion for core/tests/tools.
4. Audit every model's existing numerical/API/speed evidence, CLI continuity, exact identity goldens, unshipped discovery and actual shipped HIP lock artifact results; fix confirmed regressions and rerun affected evidence.

## Definition of done

- [ ] Every requirement has current authoritative scoped evidence with exact revisions/toolchains/artifacts; no unresolved failure is called complete.
- [ ] Clean build overhead is no more than 5%, and touching one model rebuilds only its targets plus final link.
- [ ] Zero prohibited model/core coupling is proven beyond lexical inventory, and every migrated model/default/lock quality gate is complete.

## Validation criteria

- Run isolation and generated drift checks, inspect actual include/symbol/link outputs and run the any-single-extension-excluded build matrix with retained logs.
- Measure matched clean/incremental builds under identical toolchain, concurrency and cache conditions; retain raw timings and rebuilt target lists rather than estimates.
- Replay only missing/stale/affected numerical, functional and per-request/phase timing evidence using the established harnesses. Audit every artifact's provenance and keep noisy timings visibly inconclusive until sufficiently resolved.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
