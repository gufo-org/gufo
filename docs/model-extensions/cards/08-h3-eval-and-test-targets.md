# Card 08: Extract H3, evaluation and model-dependent targets

Status: **pending**. RFC step and dependencies: **5; 07**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

After Qwen extraction, H3 and utility/test targets still couple core to models. Every such target needs an owner or a build condition so excluding a model does not break unrelated tools. DeepSeek tool_calls remains in HTTP until Card 22.

### Original scope

Extract H3, comparison support and eval targets; move eval dataset to core fixture with Nix filter exception. Give DeepSeek CPU text target without duplicating tool calls (still in HTTP until text migration). Assign or condition every model-dependent core test/tool via manifests. Record sanitizer/assertion state. Linux/Windows, affected CPU/model checks, H3 moved-HIP gate and exclusion-build coverage.

## Implementation plan

1. Extract H3 into its model library and src/testing/compare into separate comparison support; extract src/eval/{dataset,extract,regrade}.cpp into an eval target.
2. Move eval dataset to a core fixture location, update references and the specific Nix source-filter exception without bringing model tests into production sources.
3. Create a CPU-only DeepSeek text target for chat_template.cpp and sampling.cpp, linking rather than duplicating tool_calls.cpp still owned by gufo_http.
4. Inventory every model-dependent core test/tool and assign it to a model/family manifest or condition it on selection. Record old/new sanitizer and assertion state and ratchet the extracted boundaries.

## Definition of done

- [ ] H3, eval and comparison targets have explicit ownership and dependency graphs.
- [ ] Every inventoried model-dependent test/tool has manifest registration or a justified condition; no single-extension exclusion leaves unresolved model references.
- [ ] DeepSeek CPU text support has one implementation per source and no premature HTTP tool-syntax migration.

## Validation criteria

- Build affected production/test/tool targets on Linux and Windows; run eval/comparison and affected model CPU/unit checks selected from current CTest inventory.
- Run H3 moved-HIP resource and stripped production ISA comparisons with audited source/target flags, plus existing H3 numerical checks appropriate to the moved code.
- Configure/build exclusion cases for each affected extension and the corresponding core tests/tools. Record actual target selections, sanitizer/assertion state and Nix dataset admission evidence.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
