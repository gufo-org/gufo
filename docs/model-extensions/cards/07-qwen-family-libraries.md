# Card 07: Extract acyclic Qwen family libraries

Status: **pending**. RFC step and dependencies: **5; 06**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Qwen headers form cycles and reach model-specific execution policy. Target extraction must resolve the real graph instead of hiding it in a large family library. ModelConfig and TextModelPreset remain in core until text migration.

### Original scope

Extract acyclic Qwen family libraries (tokens with independent Python mirror, config/text/vision/kernels and any justified common layer), make GGUF config extraction CPU-only, extract Qwen 27B and relink CLI/kernel tools. Resolve header-graph unassigned pieces explicitly. Keep ModelConfig/TextModelPreset until text migration. Linux/Windows, unit, Qwen/Flash-Next functional/performance and moved-HIP ISA/resource gates.

## Implementation plan

1. Use Card 01 graph and src/models/qwen/AGENTS.md to assign tokens, config, text, vision and kernels under src/families/qwen with explicit manifests/dependencies.
2. Resolve token-to-vision RoPE, vision-to-GEMM, prefill-to-QwenExecutionPolicy and continuation_hooks edges. Assign gemm_route.hpp and quant_ops.hpp explicitly; justify an acyclic common layer or merged kernel/vision layer.
3. Move GGUF config extraction out of GgufReader into a free CPU-only qwen_config function and preserve config tests. Move Qwen 27B code into its own library and relink gufo_llm_cli and gufo-kernel-bench.
4. Update independent tools/gufo/control_tokens.py mirror and AGENTS links as needed; preserve per-source HIP flags, target includes/definitions and assertion/sanitizer settings.

## Definition of done

- [ ] Family graph is acyclic, every inventoried header is assigned, and manifests declare direct family dependencies.
- [ ] CPU config users do not pull HIP, and token fixture expectations remain independent from production definitions.
- [ ] Qwen 27B/Flash-Next functional and performance behavior is qualified; moved HIP resources and production ISA match or any intentional non-layout change is separately justified and tested.

## Validation criteria

- Build affected family/model/CLI/kernel-tool targets on Linux and native Windows; run CPU config/token and affected model unit tests, retaining actual build logs.
- Run affected Qwen 27B and Flash-Next modes with their existing numerical, HTTP and CLI speed controls; include continuation suites where the move touches that code and exact identity goldens.
- Audit compile_commands.json and CMakeCache.txt; run tools/ci/check-kernel-resources.py using its current CLI and compare stripped NDEBUG gfx1151 ISA for moved sources. Merely matching source text or successful compilation is insufficient.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Pending. Record implementation, independent review and third-agent integration, finding dispositions, evidence and the closing jj commit here before advancing.
