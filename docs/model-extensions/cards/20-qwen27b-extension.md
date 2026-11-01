# Card 20: Migrate Qwen 27B text generation

Status: **pending**. RFC step and dependencies: **10 Qwen 27B; 19**.

Future scope; do not execute during the cards 01–14 milestone.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Qwen 27B becomes the first production text extension using the finalized interface. Its permissive artifact fallback is deliberately replaced with a qwen35 key/shape probe; that user-facing tightening requires breaking-change release treatment.

### Original scope

Move runner, prompt formatting/tokenizer/reasoning, vision inputs/cache, artifact resolution, identity ordering, speculation and prompt/chat/bench drivers into extension. Replace permissive fallback with supported qwen35 key/shape probe and clear errors; document CLI break with `!`. Run Qwen modes, forced tool choice, long-context/cache, goldens, CLI/HTTP continuity and ISA/resource checks; isolation ratchet.

## Implementation plan

1. Move Qwen runner, chat template/tokenizer/reasoning, vision inputs/cache and artifact handling into the extension with explicit family dependencies.
2. Move persistence/process identities in their exact field order, speculative ownership and prompt/chat/bench drivers; keep fingerprint emitters and shared report/output policy core-owned.
3. Implement HIP-free supported qwen35 key/shape probing with explicit unsupported/malformed errors and provider disambiguation rather than permissive fallback.
4. Declare metadata/options/model IDs and resources; update public CLI docs and Conventional Commit/PR title with ! for stricter selection, remove migrated core knowledge and ratchet isolation.

## Definition of done

- [ ] Qwen AR/speculative and multimodal paths route through extension factories/drivers with unchanged valid-input outputs and identities.
- [ ] Supported artifact shape selection is deterministic and unsupported artifacts fail clearly with documented CLI change.
- [ ] Actual Qwen modes, numerical quality, forced tool choice, cache histories and CLI/HTTP speed continuity are qualified on discovery/lock artifacts.

## Validation criteria

- Run Qwen 27B QUALITY.md numerical checks, all affected functional modes with forced tool choice, long-context/cache and exact identity goldens; inspect actual DFlash drafts and vision identity behavior.
- Run matched bench/prompt/chat controls as applicable plus standard HTTP speed, retaining exact output hashes, work and request/phase comparisons. Historical BENCHMARKS.md values are not a baseline.
- Build/test Linux/Windows and excluded-Qwen core/tools; verify HIP-free key/shape rejection fixtures, isolation ratchet and unchanged moved-kernel resources/stripped production ISA.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
