# Card 08 — model-dependent target inventory

Inventoried with `python3 tools/ci/check-extension-isolation.py
--target-ownership-report` (183 dependency edges over 17 libraries: 6 family
libraries from Card 07, 11 manifests added by this card). A target depends on a
library by linking its target, compiling a source below its directory, or
including a header from it; the rules and the exempt model test trees are
documented in [target-ownership.md](../../target-ownership.md).

## Core tests and tools outside a model test tree

Every row is now a claim in the owning `library.json`; `extension_target_ownership_gate`
fails if a row disappears, a claim goes stale, or a new dependent is unclaimed.

| Target | Owner library | Why it depends on the model/support library | Registration |
| --- | --- | --- | --- |
| `gufo` | minimax_h3, eval, deepseek_v4_flash, qwen, qwen38_flash_next, qwen3_tts, qwen3_asr, embedding_gemma2 | `gufo eval`, video generation and the provider wiring; links the model tier directly and through `gufo_http` | `consumers` |
| `gufo-server` | — (via `gufo`) | symlink/copy of `gufo`; no direct model reference | not claimed (no edge) |
| `gufo_http` | minimax_h3, deepseek_v4_flash (+ `tool_calls.cpp`), qwen, qwen3_tts, qwen3_asr, embedding_gemma2, qwen_image_21 | video, chat-completions tool syntax, TTS/ASR/embedding/image endpoints | `consumers`, one `compiled_elsewhere` |
| `gufo_llm_cli` | qwen, qwen38_flash_next, deepseek_v4_flash, compare | `src/cli/bench/bench.cpp` reports through `CompareLogits` | `consumers` |
| `openai_chat_test` | deepseek_v4_flash, deepseek_v4_flash_text, qwen | exercises the shared chat template and HTTP tool syntax; previously compiled `chat_template.cpp`, `tool_calls.cpp` and `openai_chat.cpp` a second time | `checks` in three manifests |
| `text_model_runner_test` | qwen | links `gufo_qwen27b` for the text runner | `checks` |
| `hipblaslt_runtime_cache_test` | qwen | links `gufo_qwen27b` for the plan-database route | `checks` |
| `persistent_deltanet_test` | qwen | includes `src/models/qwen/hip` headers | `checks` |
| `exact_token_test` | qwen | includes `src/models/qwen` headers | `checks` |
| `gufo-kernel-bench` | qwen | installed benchmark tool over the Qwen kernels | `tools` |
| `qwen_oracle_test`, `qwen_l1_norm_module_test`, `qwen_l1_ffn_module_test`, `qwen_l1_ssm_module_test`, `qwen27b_target_test` | compare | call `CompareLogits` (directly or through `tests/testing/test_common.hpp`) | `checks` |
| `eval_regrade_test` | eval | links `gufo_eval` | `checks` |
| `eval_http_client_test` | eval | compiles `src/eval/http_client.cpp` | `checks` + `compiled_elsewhere` |
| `minimax_h3_runtime_test`, `minimax_h3_tokenizer_test`, `minimax_h3_sampling_test`, `minimax_h3_generation_test`, `minimax_h3_video_vae_plan`, `minimax_h3_media_test` | minimax_h3 | H3 CPU unit tests, previously satisfied by `gufo_core` objects | `checks` |
| `minimax_h3_runtime_hip_test`, `minimax_h3_prompt_encoder_hip_test`, `minimax_h3_sampling_hip_test`, `minimax_h3_dit_hip_test`, `minimax_h3_denoiser_hip_test`, `minimax_h3_video_vae_hip_test`, `minimax_h3_audio_vae_hip_test` | minimax_h3 | H3 ROCm tests (`ENGINE_ENABLE_HIP` condition on registration) | `checks` |
| `video_cli_test`, `video_jobs_test`, `video_api_test` | minimax_h3 | compile `src/cli/video/video.cpp`, `src/cli/serve/video_jobs.cpp`, `src/cli/serve/video_api.cpp`, which include H3 headers | `checks` |
| `qwen3_tts_audio_test`, `qwen3_tts_audio_api_test`, `qwen3_tts_config_test`, `qwen3_tts_loader_test`, `qwen3_tts_prompt_test`, `qwen3_tts_tokenizer_test`, `qwen3_tts_base_encoders_hip_test`, `qwen3_tts_speech_decoder_hip_test`, `qwen3_tts_synthesis_hip_test`, `qwen3_tts_talker_hip_prefill_test`, `qwen3_tts_variant_talker_hip_test` | qwen3_tts | declared in the root test block, so they are core-owned tests of a model library | `checks` |

## Conditions instead of claims

| Group | Condition |
| --- | --- |
| `ds4_*`, `qwen27b_*`, `qwen38_flash_next_*`, `embedding_gemma2_*`, `qwen_vision_serving_test`, `qwen_serving_test_support`, `inference_backend_gpu_test` | declared under `tests/models/<model>/`, which stays with the model test tree (open decision in `CARDS.md`); the gate reports them in the graph and requires no claim. Counts: deepseek_v4_flash 13, embedding_gemma2 15, qwen38_flash_next 13, qwen 5, deepseek_v4_flash_text 2, eval 1 |
| H3 ROCm sources inside `gufo_minimax_h3`, `minimax_h3_*_hip_test`, `video_*_test`, `qwen3_tts_*_hip_test`, `gufo_deepseek_v4_flash*` | registered only when `ENGINE_ENABLE_HIP` is ON; the CPU configuration builds without them (see [gates.md](gates.md)) |
| family libraries (`gufo_qwen_config`, `gufo_qwen_control_tokens`, `gufo_qwen_hip_common`, `gufo_qwen_kernels`, `gufo_qwen_text`, `gufo_qwen_vision`) | owned by the Card 07 family gate (`--fail-on-family-graph`), excluded from the target-ownership claims so the two gates have one owner each |

## Remaining duplicate compilation (recorded, not silently accepted)

```
src/models/deepseek_v4_flash/tool_calls.cpp   compiled by gufo_http
src/eval/http_client.cpp                      compiled by gufo and eval_http_client_test
```

These are the only library sources compiled outside their own directory in the
whole tree, each justified in its manifest. `tool_calls.cpp` is the Card 22
step-10 boundary this card must not cross; `http_client.cpp` stays with the
executable so `gufo_eval` has no socket dependency. Everything else that used to
be duplicated — DeepSeek `chat_template.cpp`, `runtime/sampling.cpp`,
`openai_chat.cpp` — is now linked from one target.
