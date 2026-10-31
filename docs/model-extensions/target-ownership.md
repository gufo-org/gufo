# Target ownership

Card 08 (RFC step 5) moved the MiniMax H3 model, the evaluation tier and the
logit comparison helper out of `gufo_core` into their own targets, and gave the
DeepSeek text tier a single CPU-only implementation. A library can only be
excluded, moved or split later if every target that still reaches it is known,
so each model and support library now records those targets in its
`library.json`, and one gate compares the record with the CMake files:

```sh
python3 tools/ci/check-extension-isolation.py --fail-on-target-ownership
python3 tools/ci/check-extension-isolation.py --target-ownership-report
```

The first command is CTest `extension_target_ownership_gate`
(`cpu;repository`); the second prints the whole graph for review.

## How a dependency is found

A target depends on a library when it

- links the library target (including through a variable such as
  `GUFO_QWEN_LIBRARIES`),
- compiles a source file below the library directory, or
- includes a header below the library directory from one of its own sources.

Every such target declared outside the library directory must appear in that
manifest. Targets declared inside `tests/models/<model>/` stay with the model
test tree — CARDS.md leaves that physical placement open — and are reported in
the graph without needing a claim. Family libraries under `src/families/` are
covered by `--fail-on-family-graph` and are excluded here.

Manifest fields, all checked by the gate:

| Field | Meaning |
| --- | --- |
| `name`, `target`, `kind`, `purpose` | Library identity; names and targets are unique across the repository. |
| `consumers` | Shipped targets (libraries and executables) that depend on the library. |
| `checks` | Test executables that depend on the library. |
| `tools` | Developer and benchmark executables that depend on the library. |
| `compiled_elsewhere` | `target` + `source` + `reason` for each library source that another target compiles. |

A claim that names no target or no derived dependency, a `compiled_elsewhere` entry whose source that
target does not compile, and a model directory without a manifest all fail the
gate, so the record cannot rot silently.

## Libraries and their owners

| Directory | Target | Production consumers | Tests and tools |
| --- | --- | --- | --- |
| `src/eval` | `gufo_eval` | gufo | 1 |
| `src/models/deepseek_v4_flash` | `gufo_deepseek_v4_flash` | gufo, gufo_http, gufo_llm_cli | 1 |
| `src/models/deepseek_v4_flash/text` | `gufo_deepseek_v4_flash_text` | gufo_deepseek_v4_flash | 1 |
| `src/models/embedding_gemma2` | `gufo_embedding_gemma2` | gufo, gufo_http | 0 |
| `src/models/minimax_h3` | `gufo_minimax_h3` | gufo, gufo_http | 16 |
| `src/models/qwen` | `gufo_qwen27b` | gufo, gufo_http, gufo_llm_cli | 6 |
| `src/models/qwen38_flash_next` | `gufo_qwen38_flash_next` | gufo, gufo_llm_cli | 0 |
| `src/models/qwen3_asr` | `gufo_qwen3_asr` | gufo, gufo_http | 0 |
| `src/models/qwen3_tts` | `gufo_qwen3_tts` | gufo, gufo_http | 11 |
| `src/models/qwen_image_21` | `gufo_qwen_image_21` | gufo_http | 0 |
| `src/testing/compare` | `gufo_compare` | gufo_llm_cli | 5 |

### `gufo_eval` (`src/eval/library.json`)

- Production consumers: `gufo`.
- Tests and tools: `eval_regrade_test`.
- Compiled outside the directory: `src/eval/http_client.cpp` in `gufo` and in
  `eval_http_client_test`. `gufo_eval` is the offline dataset, extraction and
  regrading library; the OpenAI-compatible client behind `gufo eval --serve`
  stays in the executable so the library keeps no socket dependency, and the
  client test drives that source against a local listener.
- The default dataset moved to the core fixture
  `tests/fixtures/eval/antirez-ds4.json`. `tests/models/deepseek_v4_flash/`
  keeps the provenance test `ds4.dataset` and `ds4_eval_test`, which now link
  `gufo_eval` instead of relying on `gufo_core`.

### `gufo_minimax_h3` (`src/models/minimax_h3/library.json`)

- Production consumers: `gufo`, `gufo_http`.
- Tests and tools (16):
  `minimax_h3_audio_vae_hip_test`, `minimax_h3_denoiser_hip_test`,
  `minimax_h3_dit_hip_test`, `minimax_h3_generation_test`, `minimax_h3_media_test`,
  `minimax_h3_prompt_encoder_hip_test`, `minimax_h3_runtime_hip_test`,
  `minimax_h3_runtime_test`, `minimax_h3_sampling_hip_test`,
  `minimax_h3_sampling_test`, `minimax_h3_tokenizer_test`,
  `minimax_h3_video_vae_hip_test`, `minimax_h3_video_vae_test`, `video_api_test`,
  `video_cli_test`, `video_jobs_test`.
- The ROCm sources join this target only when `ENGINE_ENABLE_HIP` is ON, and
  `GUFO_FFMPEG_EXECUTABLE` / `GUFO_FFPROBE_EXECUTABLE` are private definitions
  of this target: only `media_posix.cpp` and `media_windows.cpp` read them.

### `gufo_compare` (`src/testing/compare/library.json`)

- Production consumers: `gufo_llm_cli` (`src/cli/bench/bench.cpp`).
- Tests and tools: `qwen27b_target_test`, `qwen_l1_ffn_module_test`,
  `qwen_l1_norm_module_test`, `qwen_l1_ssm_module_test`, `qwen_oracle_test`.
  These are the targets that call `CompareLogits`, directly or through
  `tests/testing/test_common.hpp`; other tests include that header without
  reaching the comparison symbol and therefore do not link the library.

### `gufo_deepseek_v4_flash_text` (`src/models/deepseek_v4_flash/text/library.json`)

- Production consumers: `gufo_deepseek_v4_flash`.
- Tests and tools: `openai_chat_test`. `ds4_sampling_test` and
  `ds4_chat_template_test` live in the model test tree and link this library
  instead of recompiling the sources.
- `chat_template.cpp` and `runtime/sampling.cpp` compile once here. The name
  carries the model prefix because `text` is already the Qwen family library.
  The target keeps the shipped sampling code generation
  (`-O3 -ffast-math -fno-finite-math-only`) in HIP builds only; CPU-only
  builds retain configuration optimization without fast-math. It adds `-UNDEBUG` when
  `BUILD_TESTING` is ON, which is what the duplicated test copies had.
- `tool_calls.cpp` is not part of this library; it stays in `gufo_http` until
  Card 22, and `chat_template.cpp` needs no symbol from it beyond the
  header-only `DeepSeekToolSyntax` constants.

### Models whose tests are all model-local

`gufo_embedding_gemma2`, `gufo_qwen38_flash_next`, `gufo_qwen3_asr` and
`gufo_qwen_image_21` claim no target outside their own directory beyond
production consumers: their tests and probes are registered under
`tests/models/<model>/` (`src/models/qwen3_asr/CMakeLists.txt` registers its
own), which the graph reports without a claim. `gufo_qwen27b` is the exception
because core-owned tests and tools reach it: `exact_token_test`,
`hipblaslt_runtime_cache_test`, `openai_chat_test`,
`persistent_deltanet_test`, `text_model_runner_test` and the installed
`gufo-kernel-bench`.

## Build conditions

`library.json` records identity and ownership; when a target exists stays with
CMake, which the graph and this table summarize:

| Target | Registered when |
| --- | --- |
| `gufo_deepseek_v4_flash`, `gufo_deepseek_v4_flash_mmq` | `ENGINE_ENABLE_HIP` |
| H3 ROCm sources inside `gufo_minimax_h3` | `ENGINE_ENABLE_HIP` |
| `gufo_deepseek_v4_flash_text`, `gufo_minimax_h3`, `gufo_eval`, `gufo_compare` | always, so CPU builds keep DeepSeek text support and `gufo bench` |
| `minimax_h3_*_hip_test`, `video_*_test`, `qwen3_tts_*_hip_test` | `BUILD_TESTING` and `ENGINE_ENABLE_HIP` |

## Limits of the audit

- The CMake scan is lexical: it resolves plain paths, `${CMAKE_CURRENT_SOURCE_DIR}`,
  `${PROJECT_SOURCE_DIR}` and simple list variables. Generator expressions are
  skipped, and a target created inside a helper function is reported as
  unattributable unless the helper is called from `tests/models/<model>/`.
- Includes are read from a target's own sources. A model header that arrives
  only through a shared header is still visible in the coupling report
  (`--mode report`), not as a target edge; a link edge normally covers the same
  dependency.
- The audit is static. Whether a claimed target actually builds is shown by
  the build and CTest gates, not here.
