# Card 24: Prove capability extensibility with second providers

Status: **implementing**. RFC step and dependencies: **11; 23**.

Future scope; do not execute during the cards 01–14 milestone.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

A migrated provider only proves its own interface fits. Every capability needs an independent second implementation to demonstrate that an existing-capability model can be added without touching core, build or registry source lists.

### Original scope

Add genuinely external test-only second providers for every capability (or genuine new models), run conformance and prove adding an existing-capability model changes only its own directories plus allowed generated output/lock release decision. Include zero core/build/preset/workflow/Nix/registry source changes. Cover ambiguity, excluded keys, provider override and HIP-free probes; builds exclude any single extension.

## Implementation plan

1. Create external test-only second providers for text, embeddings, ASR, TTS, image and video, or use genuine new model ports; do not clone migrated model internals as the contract oracle.
2. Run each capability conformance suite, covering advertised IDs, request validation, step/stream/cancellation contracts and capability-specific scheduling boundaries.
3. Exercise pure HIP-free probes, ambiguous matches, explicit provider override and discovered-but-excluded key diagnostics.
4. Record the exact addition diff: only extension directories plus allowed generated output and explicit lock release decision may change; test every single-extension exclusion against core/tools.

## Definition of done

- [ ] Every capability has a genuinely external second provider with retained conformance evidence.
- [ ] Adding those providers requires zero core/build/preset/workflow/Nix/registry source edits.
- [ ] Selection ambiguity/overrides/exclusions and HIP-less probes behave deterministically; all any-single-extension-excluded core/tool builds pass.

## Validation criteria

- Build external fixtures in clean source trees and run to-be-added capability conformance tests; retain a capability-by-provider matrix and concrete response/stream assertions.
- Compare jj diffs/source inventories before and after each addition, classifying only allowed generated/lock changes; a manually patched core invalidates the proof.
- Configure/build the full exclusion matrix and run selection/probe negative fixtures without HIP hardware. Stub conformance establishes interface behavior, not real-model numerical quality or speed.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

### Implementation (`text_generation` via `lfm3-gufo-extension` — 2026-11-01)

- **Scope completed**: Ported the Liquid LFM3 1.7B model (`gguf:lfm3`, model IDs `["LFM3-1.7B", "lfm3-1.7b"]`) from the `lfm3` branch into the external extension repository `git@github.com:gufo-org/lfm3-gufo-extension.git` (commit `6360b0fceacc7e98910c6d686c29984dc90374ea`), built and tested against Gufo's public extension API (`<gufo/ext/...>`, `>= 0.1.8`) via `-DGUFO_EXTENSION_DIRS=/home/mixer/gufo/lfm3-gufo-extension`.
- **Zero-core-diff proof**: Adding and exercising `lfm3-gufo-extension` required **zero** edits to Gufo core, build, preset, workflow, Nix, or registry source files (`jj status` / `jj diff` across `src/`, `include/`, `cmake/`, `CMakeLists.txt`, `CMakePresets.json`, `flake.nix`, and `extensions.lock` is empty).
- **Retained evidence**: See [`docs/model-extensions/evidence/card24/README.md`](../evidence/card24/README.md).
  - **GPU build & diagnostics (`gpu-test`)**: `lfm3_config_test`, `lfm3_ops_test`, `lfm3_session_test`, `lfm3_probe`, and `gufo` compiled with 0 warnings and 0 errors; `gufo diagnose --section extensions --json` reports `lfm3` built from `"source": "lfm3-gufo-extension"` with provider `"id": "lfm3", "available": true`.
  - **Unit, HIP kernel, session & CPU/GPU logit probe tests**:
    - `lfm3_config_test`: `PASS LFM3 metadata and chat framing`
    - `lfm3_ops_test`: `PASS LFM3 convolution, RoPE, gated GQA and sliding ring against independent formulas`
    - `lfm3_session_test models/lfm3/LFM3-1.7B-Q8_0.gguf`: `513-token batched/serial max_abs=3.19481e-05`, `2/3/4/8-session batch/serial max_abs <= 9.17912e-06` (all 3 `PASS` assertions).
    - `lfm3_probe` CPU vs GPU 128,000-vocab logit comparison: `max_abs=3.458261e-04`, `mean_abs=5.191900e-05`, `top1=4741 (" Paris")` on both CPU and GPU.
    - CLI `gufo prompt` and `gufo bench` (`pp128`: `4725.66 tok/s`, `tg32`: `106.153 tok/s`).
  - **Live HTTP serving conformance (`gufo serve llm -j 2 -c 1024`)**: Verified `GET /v1/models` (`id: "LFM3-1.7B"`), non-streaming `POST /v1/chat/completions` (`enable_thinking=false` and `enable_thinking=true`), streaming SSE (`data: [DONE]`), request validation (`404 model_not_found`, `400 invalid_messages` for image input), and 2-way concurrent session batching.
  - **HIP-free CPU build & probe (`cpu-test`)**: Verified `lfm3_config_test`, `lfm3_probe ... cpu`, and `gufo diagnose --section extensions --json` (`"available": false, "reason": "requires hip"`).
- **Remaining scope**: External second providers for non-text capabilities (`embeddings`, `speech_recognition`, `speech_synthesis`, `image_generation`, `video_generation`), ambiguity/override/excluded-key negative fixtures across capabilities, and the full any-single-extension-excluded build matrix remain pending.
