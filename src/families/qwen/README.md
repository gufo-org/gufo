# Qwen family libraries

Code several Qwen extensions share, split into small acyclic libraries so no
family library becomes a second core (RFC step 5). `src/core` contains no Qwen
code either: a family library may use `gufo_core`, never the other way round,
and family code never includes a model header.

| Library | Target | Kind | Contents | Used by |
| --- | --- | --- | --- | --- |
| `control_tokens` | `gufo_qwen_control_tokens` | interface | Shared control-token spellings in `gufo::tokenization`. | Qwen 27B, Flash-Next, Qwen3-TTS, Qwen-Image-2.1, MiniMax H3 |
| `hip_common` | `gufo_qwen_hip_common` | interface | Execution policy, attention policy, recurrent-state layout, GEMM routing, quantization helpers, the RoPE layout shared by attention and vision, and the continuation-hook interface. Below `kernels` and `vision`. | kernels, vision, Qwen 27B, Flash-Next |
| `config` | `gufo_qwen_config` | static (CPU) | `ExtractQwenConfig`: Qwen GGUF configuration extraction and structural validation. Reaches no HIP. | Qwen 27B, text (chat-template shape check), `prompt` |
| `text` | `gufo_qwen_text` | static (CPU) | Tokenizer and bounded deterministic ChatML template. | Qwen 27B, Flash-Next, Qwen-Image-2.1 |
| `kernels` | `gufo_qwen_kernels` | static (HIP) | Shared Qwen HIP operators: `ops.hpp`, `ops/` launcher interfaces and the kernel translation units. | Qwen 27B, Flash-Next, Qwen3-TTS |
| `vision` | `gufo_qwen_vision` | static | Vision encoder, device image input, image prompt expansion and image cache. | Qwen 27B, Flash-Next |

Dependency order (each library may use the ones above it):

```text
control_tokens   hip_common
      \            |      \
       text -- config      kernels
        |                    |
      vision <--------------+
        |
   (Qwen 27B model, src/models/qwen)
```

## Manifests

`<library>/library.json` is the manifest:

| Field | Meaning |
| --- | --- |
| `name` | Library name; must equal the directory name. |
| `target` | The CMake target the directory defines. |
| `kind` | `interface` (header-only, defines no object code) or `static`. |
| `purpose` | One-sentence responsibility, shown by `diagnose`-style reports and review. |
| `depends` | Sibling libraries this library includes files from. Direct edges only. |

The graph is enforced, not documented prose:

```sh
python3 tools/ci/check-extension-isolation.py --fail-on-family-graph
```

CTest `extension_family_graph_gate` fails when a header under `src/families/`
is not inside a declared library, when `depends` and the real include edges
differ in either direction, when the declared graph has a cycle, or when family
code reaches `src/models/**`. `--fail-on-core-model-includes` keeps `src/core`
free of model and family headers.

## Why these boundaries

Card 01's header inventory found the old `src/models/qwen/` graph reaching
across layers in four places: `hip/ops/token.hpp` and `hip/ops/attention.hpp`
included `vision/rope.hpp`; `vision/encoder.hip` included `hip/ops/gemm.hpp`;
`hip/kernels/prefill_quant_gemm.hpp` reached the execution policy through
`hip/detail/attention_policy.hpp`; and `vision/device_input.hpp` reached
`continuation_hooks.hpp`. Directory-level layers would therefore have cycled.

The unassigned headers got explicit homes: `gemm_route.hpp`, `hip/quant_ops.hpp`
and `hip/detail/recurrent_state.hpp` → `hip_common`;
`hip/execution_policy.hpp`, `hip/detail/attention_policy.hpp` → `hip_common`;
`continuation_hooks.hpp` → `hip_common`; `vision/rope.hpp` → `hip_common` as
`rope.hpp`, because both attention ops and the vision encoder need that layout.
`hip_common` is header-only, so it is the small common layer the RFC allowed
for this purpose rather than a second core, and it compiles no code of its own:
every consumer compiles these headers with its own per-source options.
