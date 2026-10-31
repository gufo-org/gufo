# Card 11: Migrate models to public extension headers

Status: **pending**. RFC step and dependencies: **7; 10**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

The classified public surface becomes meaningful only when models consume it. Keep this card mechanical: change one model boundary at a time and preserve temporary root-relative own includes until each actual extension migration.

### Original scope

Mechanical public-header include migrations one model at a time, preserving root-relative own includes temporarily: EmbeddingGemma, ASR, TTS, Image, H3, Qwen 27B, Flash-Next, DeepSeek. Record per-model Linux/Windows and affected unit checks; keep each rewrite reviewable and ratchet only migrated boundaries.

## Implementation plan

1. For EmbeddingGemma, ASR, TTS, Image, H3, Qwen 27B, Flash-Next and DeepSeek in that order, replace private core includes with the Card 10 public equivalents.
2. Keep each model rewrite reviewable and compile it before advancing; do not mix arithmetic, provider routing or target ownership redesign into the include change.
3. Update SURFACE after include changes and apply required API policy when declarations/contracts change rather than only paths.
4. Ratchet isolation only for boundaries already migrated, recording remaining intentional root-relative own/family includes for later cards.

## Definition of done

- [ ] All eight models use published core/platform/experimental paths in place of classified private core paths.
- [ ] Every rewrite has per-model Linux/Windows and affected unit evidence; remaining own/family coupling is explicitly inventoried.
- [ ] No new private dependency is hidden in the isolation baseline or accepted as a migration exception.

## Validation criteria

- Build each model's current targets on Linux and Windows and run the corresponding affected CPU/unit tests before proceeding to the next model.
- Run python3 tools/ci/check-extension-isolation.py --json-report /tmp/gufo-card11-isolation.json; inspect per-model moved edges and the ratchet, not just command exit.
- Run python3 tools/ci/check-extension-api.py and required formatting. Confirm the final diff is mechanical; broaden numerical/functional checks only if the diff changes behavior or exposes risk.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
