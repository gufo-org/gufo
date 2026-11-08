# Card 11: Model Include Inventory

## Overview

Mechanical migration of private `src/core/` and `src/cache/` include directives to public `<gufo/ext/...>` headers across all eight model families in `src/models/`.

| Model | Source Directory | Modified Files | Migrated Includes | Remaining Own Root-Relative | Remaining Family Coupling | Test-Only Serving Edge |
|---|---|---|---|---|---|---|
| EmbeddingGemma | `src/models/embedding_gemma2` | 9 | 11 | 38 | 0 | 0 |
| ASR | `src/models/qwen3_asr` | 7 | 11 | 46 | 0 | 2 |
| TTS | `src/models/qwen3_tts` | 6 | 10 | 45 | 2 | 0 |
| Image | `src/models/qwen_image_21` | 4 | 7 | 19 | 2 | 0 |
| H3 | `src/models/minimax_h3` | 7 | 11 | 57 | 1 | 0 |
| Qwen 27B | `src/models/qwen` | 31 | 53 | 114 | 52 | 0 |
| Flash-Next | `src/models/qwen38_flash_next` | 19 | 29 | 58 | 8 | 0 |
| DeepSeek | `src/models/deepseek_v4_flash` | 12 | 20 | 9 | 0 | 0 |
| **Total** | | **95** | **152** | **386** | **65** | **2** |

---

## Per-Model Include Migration Details

### EmbeddingGemma (`src/models/embedding_gemma2`)

| File | Prior Include | Migrated Public Include |
|---|---|---|
| `src/models/embedding_gemma2/audio_processor.cpp` | `"src/core/json.hpp"` | `<gufo/ext/json.hpp>` |
| `src/models/embedding_gemma2/config.cpp` | `"src/core/json.hpp"` | `<gufo/ext/json.hpp>` |
| `src/models/embedding_gemma2/hip/audio_runtime.hip` | `"src/core/hip/hip_utils.hpp"` | `<gufo/ext/experimental/hip_utils.hpp>` |
| `src/models/embedding_gemma2/hip/encoder_runtime.cpp` | `"src/core/hip/hip_utils.hpp"` | `<gufo/ext/experimental/hip_utils.hpp>` |
| `src/models/embedding_gemma2/image_processor.cpp` | `"src/core/json.hpp"` | `<gufo/ext/json.hpp>` |
| `src/models/embedding_gemma2/image_processor.hpp` | `"src/core/image.hpp"` | `<gufo/ext/image.hpp>` |
| `src/models/embedding_gemma2/loader.cpp` | `"src/core/json.hpp"` | `<gufo/ext/json.hpp>` |
| `src/models/embedding_gemma2/loader.cpp` | `"src/core/platform/mapped_file.hpp"` | `<gufo/ext/platform/mapped_file.hpp>` |
| `src/models/embedding_gemma2/tokenizer_loader.cpp` | `"src/core/json.hpp"` | `<gufo/ext/json.hpp>` |
| `src/models/embedding_gemma2/video_processor.cpp` | `"src/core/json.hpp"` | `<gufo/ext/json.hpp>` |
| `src/models/embedding_gemma2/video_processor.cpp` | `"src/core/platform/library.hpp"` | `<gufo/ext/platform/library.hpp>` |

### ASR (`src/models/qwen3_asr`)

| File | Prior Include | Migrated Public Include |
|---|---|---|
| `src/models/qwen3_asr/config.cpp` | `"src/core/json.hpp"` | `<gufo/ext/json.hpp>` |
| `src/models/qwen3_asr/hip/audio_encoder_runtime.cpp` | `"src/core/hip/weight_upload.hpp"` | `<gufo/ext/experimental/weight_upload.hpp>` |
| `src/models/qwen3_asr/hip/text_decoder_runtime.cpp` | `"src/core/hip/weight_upload.hpp"` | `<gufo/ext/experimental/weight_upload.hpp>` |
| `src/models/qwen3_asr/hip/text_decoder_runtime.cpp` | `"src/core/mapped_prefetch.hpp"` | `<gufo/ext/mapped_prefetch.hpp>` |
| `src/models/qwen3_asr/hip/transcription_runtime.cpp` | `"src/core/cancellable_gate.hpp"` | `<gufo/ext/cancellable_gate.hpp>` |
| `src/models/qwen3_asr/hip/transcription_runtime.cpp` | `"src/core/utf8.hpp"` | `<gufo/ext/utf8.hpp>` |
| `src/models/qwen3_asr/loader.cpp` | `"src/core/json.hpp"` | `<gufo/ext/json.hpp>` |
| `src/models/qwen3_asr/loader.cpp` | `"src/core/platform/mapped_file.hpp"` | `<gufo/ext/platform/mapped_file.hpp>` |
| `src/models/qwen3_asr/loader.cpp` | `"src/core/platform/prefault.hpp"` | `<gufo/ext/platform/prefault.hpp>` |
| `src/models/qwen3_asr/loader.hpp` | `"src/core/gguf_reader.hpp"` | `<gufo/ext/gguf_reader.hpp>` |
| `src/models/qwen3_asr/tokenizer.cpp` | `"src/core/json.hpp"` | `<gufo/ext/json.hpp>` |

### TTS (`src/models/qwen3_tts`)

| File | Prior Include | Migrated Public Include |
|---|---|---|
| `src/models/qwen3_tts/config.cpp` | `"src/core/json.hpp"` | `<gufo/ext/json.hpp>` |
| `src/models/qwen3_tts/hip/speech_decoder_runtime.cpp` | `"src/core/hip/weight_upload.hpp"` | `<gufo/ext/experimental/weight_upload.hpp>` |
| `src/models/qwen3_tts/hip/speech_decoder_runtime.cpp` | `"src/core/mapped_prefetch.hpp"` | `<gufo/ext/mapped_prefetch.hpp>` |
| `src/models/qwen3_tts/hip/talker_runtime.cpp` | `"src/core/hip/weight_upload.hpp"` | `<gufo/ext/experimental/weight_upload.hpp>` |
| `src/models/qwen3_tts/hip/talker_runtime.cpp` | `"src/core/mapped_prefetch.hpp"` | `<gufo/ext/mapped_prefetch.hpp>` |
| `src/models/qwen3_tts/loader.cpp` | `"src/core/json.hpp"` | `<gufo/ext/json.hpp>` |
| `src/models/qwen3_tts/loader.cpp` | `"src/core/platform/mapped_file.hpp"` | `<gufo/ext/platform/mapped_file.hpp>` |
| `src/models/qwen3_tts/loader.cpp` | `"src/core/platform/prefault.hpp"` | `<gufo/ext/platform/prefault.hpp>` |
| `src/models/qwen3_tts/loader.hpp` | `"src/core/gguf_reader.hpp"` | `<gufo/ext/gguf_reader.hpp>` |
| `src/models/qwen3_tts/tokenizer.cpp` | `"src/core/json.hpp"` | `<gufo/ext/json.hpp>` |

### Image (`src/models/qwen_image_21`)

| File | Prior Include | Migrated Public Include |
|---|---|---|
| `src/models/qwen_image_21/hip/kernels.hip` | `"src/core/hip/weight_upload.hpp"` | `<gufo/ext/experimental/weight_upload.hpp>` |
| `src/models/qwen_image_21/image.cpp` | `"src/core/image.hpp"` | `<gufo/ext/image.hpp>` |
| `src/models/qwen_image_21/image.cpp` | `"src/core/crypto/sha256.hpp"` | `<gufo/ext/sha256.hpp>` |
| `src/models/qwen_image_21/weights.cpp` | `"src/core/mapped_prefetch.hpp"` | `<gufo/ext/mapped_prefetch.hpp>` |
| `src/models/qwen_image_21/weights.cpp` | `"src/core/platform/mapped_file.hpp"` | `<gufo/ext/platform/mapped_file.hpp>` |
| `src/models/qwen_image_21/weights.hpp` | `"src/core/gguf_reader.hpp"` | `<gufo/ext/gguf_reader.hpp>` |
| `src/models/qwen_image_21/weights.hpp` | `"src/core/json.hpp"` | `<gufo/ext/json.hpp>` |

### H3 (`src/models/minimax_h3`)

| File | Prior Include | Migrated Public Include |
|---|---|---|
| `src/models/minimax_h3/audio_vae.hip` | `"src/core/platform/mapped_file.hpp"` | `<gufo/ext/platform/mapped_file.hpp>` |
| `src/models/minimax_h3/audio_vae.hip` | `"src/core/platform/os_info.hpp"` | `<gufo/ext/platform/os_info.hpp>` |
| `src/models/minimax_h3/denoiser.hip` | `"src/core/platform/mapped_file.hpp"` | `<gufo/ext/platform/mapped_file.hpp>` |
| `src/models/minimax_h3/denoiser.hip` | `"src/core/platform/os_info.hpp"` | `<gufo/ext/platform/os_info.hpp>` |
| `src/models/minimax_h3/dit.hip` | `"src/core/platform/mapped_file.hpp"` | `<gufo/ext/platform/mapped_file.hpp>` |
| `src/models/minimax_h3/generation.cpp` | `"src/core/platform/os_info.hpp"` | `<gufo/ext/platform/os_info.hpp>` |
| `src/models/minimax_h3/prompt_encoder.hip` | `"src/core/mapped_prefetch.hpp"` | `<gufo/ext/mapped_prefetch.hpp>` |
| `src/models/minimax_h3/prompt_encoder.hip` | `"src/core/platform/mapped_file.hpp"` | `<gufo/ext/platform/mapped_file.hpp>` |
| `src/models/minimax_h3/runtime.hpp` | `"src/core/platform/mapped_file.hpp"` | `<gufo/ext/platform/mapped_file.hpp>` |
| `src/models/minimax_h3/video_vae.hip` | `"src/core/platform/mapped_file.hpp"` | `<gufo/ext/platform/mapped_file.hpp>` |
| `src/models/minimax_h3/video_vae.hip` | `"src/core/platform/os_info.hpp"` | `<gufo/ext/platform/os_info.hpp>` |

### Qwen 27B (`src/models/qwen`)

| File | Prior Include | Migrated Public Include |
|---|---|---|
| `src/models/qwen/continuation_adapter.cpp` | `"src/core/hip/snapshot_copy.hpp"` | `<gufo/ext/experimental/snapshot_copy.hpp>` |
| `src/models/qwen/continuation_adapter.hpp` | `"src/cache/adapter.hpp"` | `<gufo/ext/adapter.hpp>` |
| `src/models/qwen/continuation_adapter.hpp` | `"src/core/hip/transfer_pool.hpp"` | `<gufo/ext/experimental/transfer_pool.hpp>` |
| `src/models/qwen/continuation_layout.hpp` | `"src/cache/adapter.hpp"` | `<gufo/ext/adapter.hpp>` |
| `src/models/qwen/continuation_layout.hpp` | `"src/cache/types.hpp"` | `<gufo/ext/model_config.hpp>` |
| `src/models/qwen/continuation_layout.hpp` | `"src/core/model_config.hpp"` | `<gufo/ext/types.hpp>` |
| `src/models/qwen/dflash_weights.hpp` | `"src/core/gguf_reader.hpp"` | `<gufo/ext/gguf_reader.hpp>` |
| `src/models/qwen/forward.cpp` | `"src/core/quant/ggml_dequant.hpp"` | `<gufo/ext/experimental/ggml_dequant.hpp>` |
| `src/models/qwen/forward.cpp` | `"src/core/quant/ggml_gemm.hpp"` | `<gufo/ext/experimental/ggml_gemm.hpp>` |
| `src/models/qwen/generator.hpp` | `"src/core/gguf_reader.hpp"` | `<gufo/ext/gguf_reader.hpp>` |
| `src/models/qwen/generator.hpp` | `"src/core/sampling.hpp"` | `<gufo/ext/sampling.hpp>` |
| `src/models/qwen/hip/arena.cpp` | `"src/core/hip/hip_utils.hpp"` | `<gufo/ext/experimental/hip_utils.hpp>` |
| `src/models/qwen/hip/arena.cpp` | `"src/core/hip/snapshot_transfer.hpp"` | `<gufo/ext/experimental/snapshot_transfer.hpp>` |
| `src/models/qwen/hip/batched_decode.cpp` | `"src/core/hip/hip_utils.hpp"` | `<gufo/ext/experimental/hip_utils.hpp>` |
| `src/models/qwen/hip/decode.cpp` | `"src/core/hip/detail/dispatch_telemetry.hpp"` | `<gufo/ext/experimental/dispatch_telemetry.hpp>` |
| `src/models/qwen/hip/decode.cpp` | `"src/core/hip/hip_utils.hpp"` | `<gufo/ext/experimental/hip_utils.hpp>` |
| `src/models/qwen/hip/decode_step.cpp` | `"src/core/hip/detail/dispatch_telemetry.hpp"` | `<gufo/ext/experimental/dispatch_telemetry.hpp>` |
| `src/models/qwen/hip/decode_step.cpp` | `"src/core/hip/hip_utils.hpp"` | `<gufo/ext/experimental/hip_utils.hpp>` |
| `src/models/qwen/hip/dflash.hpp` | `"src/core/gguf_reader.hpp"` | `<gufo/ext/draft_backend.hpp>` |
| `src/models/qwen/hip/dflash.hpp` | `"src/core/speculative/draft_backend.hpp"` | `<gufo/ext/gguf_reader.hpp>` |
| `src/models/qwen/hip/dflash.hpp` | `"src/core/speculative/prompt_lookup.hpp"` | `<gufo/ext/prompt_lookup.hpp>` |
| `src/models/qwen/hip/dflash/backend.cpp` | `"src/core/sampling.hpp"` | `<gufo/ext/sampling.hpp>` |
| `src/models/qwen/hip/dflash/batch.cpp` | `"src/core/hip/hip_utils.hpp"` | `<gufo/ext/experimental/hip_utils.hpp>` |
| `src/models/qwen/hip/dflash/executor.cpp` | `"src/core/hip/hip_utils.hpp"` | `<gufo/ext/experimental/hip_utils.hpp>` |
| `src/models/qwen/hip/dflash/executor.cpp` | `"src/core/hip/snapshot_transfer.hpp"` | `<gufo/ext/experimental/snapshot_transfer.hpp>` |
| `src/models/qwen/hip/dflash/model.cpp` | `"src/core/hip/hip_utils.hpp"` | `<gufo/ext/experimental/ggml_dequant.hpp>` |
| `src/models/qwen/hip/dflash/model.cpp` | `"src/core/mapped_prefetch.hpp"` | `<gufo/ext/experimental/hip_utils.hpp>` |
| `src/models/qwen/hip/dflash/model.cpp` | `"src/core/quant/ggml_dequant.hpp"` | `<gufo/ext/mapped_prefetch.hpp>` |
| `src/models/qwen/hip/executor.cpp` | `"src/core/hip/detail/dispatch_telemetry.hpp"` | `<gufo/ext/experimental/dispatch_telemetry.hpp>` |
| `src/models/qwen/hip/executor.cpp` | `"src/core/hip/hip_utils.hpp"` | `<gufo/ext/experimental/hip_utils.hpp>` |
| `src/models/qwen/hip/executor.hpp` | `"src/core/model_config.hpp"` | `<gufo/ext/model_config.hpp>` |
| `src/models/qwen/hip/executor.hpp` | `"src/core/hip/detail/hip_graph_decode_executor.hpp"` | `<gufo/ext/experimental/hip_graph_decode_executor.hpp>` |
| `src/models/qwen/hip/model_loader.cpp` | `"src/core/hip/heap_placement.hpp"` | `<gufo/ext/experimental/heap_placement.hpp>` |
| `src/models/qwen/hip/model_loader.cpp` | `"src/core/hip/weight_upload.hpp"` | `<gufo/ext/experimental/weight_upload.hpp>` |
| `src/models/qwen/hip/model_loader.cpp` | `"src/core/platform/pages.hpp"` | `<gufo/ext/platform/pages.hpp>` |
| `src/models/qwen/hip/model_loader.cpp` | `"src/core/platform/prefault.hpp"` | `<gufo/ext/platform/prefault.hpp>` |
| `src/models/qwen/hip/mtp.hpp` | `"src/core/speculative/draft_backend.hpp"` | `<gufo/ext/draft_backend.hpp>` |
| `src/models/qwen/hip/mtp/detail/allocation.hpp` | `"src/core/hip/managed_alloc.hpp"` | `<gufo/ext/experimental/managed_alloc.hpp>` |
| `src/models/qwen/hip/mtp/model.cpp` | `"src/core/quant/ggml_dequant.hpp"` | `<gufo/ext/experimental/ggml_dequant.hpp>` |
| `src/models/qwen/hip/prefill.cpp` | `"src/core/hip/hip_utils.hpp"` | `<gufo/ext/experimental/hip_utils.hpp>` |
| `src/models/qwen/hip/prefill_chunk.cpp` | `"src/core/hip/detail/dispatch_telemetry.hpp"` | `<gufo/ext/experimental/dispatch_telemetry.hpp>` |
| `src/models/qwen/hip/prefill_chunk.cpp` | `"src/core/hip/hip_utils.hpp"` | `<gufo/ext/experimental/hip_utils.hpp>` |
| `src/models/qwen/hip/prompt_hidden_buffer.hpp` | `"src/cache/ledger.hpp"` | `<gufo/ext/experimental/host_allocations.hpp>` |
| `src/models/qwen/hip/prompt_hidden_buffer.hpp` | `"src/core/hip/host_allocations.hpp"` | `<gufo/ext/ledger.hpp>` |
| `src/models/qwen/hip/qwen_speculative_verifier.hpp` | `"src/core/speculative/speculative_verifier.hpp"` | `<gufo/ext/speculative_verifier.hpp>` |
| `src/models/qwen/hip/ssm_replay.cpp` | `"src/core/hip/hip_utils.hpp"` | `<gufo/ext/experimental/hip_utils.hpp>` |
| `src/models/qwen/modules/module_ctx.hpp` | `"src/core/model_config.hpp"` | `<gufo/ext/model_config.hpp>` |
| `src/models/qwen/mtp_reference.cpp` | `"src/core/quant/ggml_dequant.hpp"` | `<gufo/ext/experimental/ggml_dequant.hpp>` |
| `src/models/qwen/mtp_reference.hpp` | `"src/core/gguf_reader.hpp"` | `<gufo/ext/gguf_reader.hpp>` |
| `src/models/qwen/ssm.hpp` | `"src/core/model_config.hpp"` | `<gufo/ext/model_config.hpp>` |
| `src/models/qwen/state.hpp` | `"src/core/gguf_reader.hpp"` | `<gufo/ext/experimental/ggml_dequant.hpp>` |
| `src/models/qwen/state.hpp` | `"src/core/model_config.hpp"` | `<gufo/ext/gguf_reader.hpp>` |
| `src/models/qwen/state.hpp` | `"src/core/quant/ggml_dequant.hpp"` | `<gufo/ext/model_config.hpp>` |

### Flash-Next (`src/models/qwen38_flash_next`)

| File | Prior Include | Migrated Public Include |
|---|---|---|
| `src/models/qwen38_flash_next/config.hpp` | `"src/core/gguf_reader.hpp"` | `<gufo/ext/gguf_reader.hpp>` |
| `src/models/qwen38_flash_next/continuation_adapter.cpp` | `"src/core/hip/snapshot_copy.hpp"` | `<gufo/ext/experimental/snapshot_copy.hpp>` |
| `src/models/qwen38_flash_next/continuation_adapter.hpp` | `"src/cache/adapter.hpp"` | `<gufo/ext/adapter.hpp>` |
| `src/models/qwen38_flash_next/continuation_adapter.hpp` | `"src/core/hip/transfer_pool.hpp"` | `<gufo/ext/experimental/transfer_pool.hpp>` |
| `src/models/qwen38_flash_next/continuation_layout.hpp` | `"src/cache/adapter.hpp"` | `<gufo/ext/adapter.hpp>` |
| `src/models/qwen38_flash_next/continuation_layout.hpp` | `"src/cache/types.hpp"` | `<gufo/ext/types.hpp>` |
| `src/models/qwen38_flash_next/cpu_ops.cpp` | `"src/core/quant/ggml_dequant.hpp"` | `<gufo/ext/experimental/ggml_dequant.hpp>` |
| `src/models/qwen38_flash_next/engine.cpp` | `"src/core/gguf_reader.hpp"` | `<gufo/ext/gguf_reader.hpp>` |
| `src/models/qwen38_flash_next/engine.hpp` | `"src/core/sampling.hpp"` | `<gufo/ext/prompt_lookup.hpp>` |
| `src/models/qwen38_flash_next/engine.hpp` | `"src/core/session_mode.hpp"` | `<gufo/ext/sampling.hpp>` |
| `src/models/qwen38_flash_next/engine.hpp` | `"src/core/speculative/prompt_lookup.hpp"` | `<gufo/ext/session_mode.hpp>` |
| `src/models/qwen38_flash_next/kernels/rocm/batch.cpp` | `"src/core/hip/managed_alloc.hpp"` | `<gufo/ext/experimental/managed_alloc.hpp>` |
| `src/models/qwen38_flash_next/kernels/rocm/device_model.cpp` | `"src/core/hip/managed_alloc.hpp"` | `<gufo/ext/experimental/managed_alloc.hpp>` |
| `src/models/qwen38_flash_next/kernels/rocm/device_model.cpp` | `"src/core/hip/weight_upload.hpp"` | `<gufo/ext/experimental/weight_upload.hpp>` |
| `src/models/qwen38_flash_next/kernels/rocm/device_model.hpp` | `"src/core/gguf_reader.hpp"` | `<gufo/ext/gguf_reader.hpp>` |
| `src/models/qwen38_flash_next/kernels/rocm/executor.cpp` | `"src/core/hip/managed_alloc.hpp"` | `<gufo/ext/experimental/managed_alloc.hpp>` |
| `src/models/qwen38_flash_next/kernels/rocm/executor.cpp` | `"src/core/hip/snapshot_transfer.hpp"` | `<gufo/ext/experimental/snapshot_transfer.hpp>` |
| `src/models/qwen38_flash_next/kernels/rocm/executor.cpp` | `"src/core/platform/pages.hpp"` | `<gufo/ext/platform/pages.hpp>` |
| `src/models/qwen38_flash_next/kernels/rocm/executor.hpp` | `"src/core/hip/host_allocations.hpp"` | `<gufo/ext/experimental/host_allocations.hpp>` |
| `src/models/qwen38_flash_next/kernels/rocm/executor.hpp` | `"src/core/session_mode.hpp"` | `<gufo/ext/session_mode.hpp>` |
| `src/models/qwen38_flash_next/kernels/rocm/kernels.hpp` | `"src/core/sampling.hpp"` | `<gufo/ext/sampling.hpp>` |
| `src/models/qwen38_flash_next/kernels/rocm/mmq/mmid.hip.cpp` | `"src/core/hip/managed_alloc.hpp"` | `<gufo/ext/experimental/managed_alloc.hpp>` |
| `src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_ggml_stubs.hip.cpp` | `"src/core/hip/managed_alloc.hpp"` | `<gufo/ext/experimental/managed_alloc.hpp>` |
| `src/models/qwen38_flash_next/mtp_sampling.hpp` | `"src/core/sampling.hpp"` | `<gufo/ext/sampling.hpp>` |
| `src/models/qwen38_flash_next/ngram.cpp` | `"src/core/platform/fs.hpp"` | `<gufo/ext/experimental/ggml_dequant.hpp>` |
| `src/models/qwen38_flash_next/ngram.cpp` | `"src/core/platform/mapped_file.hpp"` | `<gufo/ext/platform/fs.hpp>` |
| `src/models/qwen38_flash_next/ngram.cpp` | `"src/core/quant/ggml_dequant.hpp"` | `<gufo/ext/platform/mapped_file.hpp>` |
| `src/models/qwen38_flash_next/ngram.hpp` | `"src/core/gguf_reader.hpp"` | `<gufo/ext/gguf_reader.hpp>` |
| `src/models/qwen38_flash_next/weights.hpp` | `"src/core/gguf_reader.hpp"` | `<gufo/ext/gguf_reader.hpp>` |

### DeepSeek (`src/models/deepseek_v4_flash`)

| File | Prior Include | Migrated Public Include |
|---|---|---|
| `src/models/deepseek_v4_flash/chat_template.cpp` | `"src/core/crypto/sha256.hpp"` | `<gufo/ext/sha256.hpp>` |
| `src/models/deepseek_v4_flash/chat_template.hpp` | `"src/core/gguf_reader.hpp"` | `<gufo/ext/gguf_reader.hpp>` |
| `src/models/deepseek_v4_flash/chat_template.hpp` | `"src/core/reasoning.hpp"` | `<gufo/ext/reasoning.hpp>` |
| `src/models/deepseek_v4_flash/dspark_sampler.hpp` | `"src/core/sampling.hpp"` | `<gufo/ext/sampling.hpp>` |
| `src/models/deepseek_v4_flash/engine.hpp` | `"src/core/session_mode.hpp"` | `<gufo/ext/session_mode.hpp>` |
| `src/models/deepseek_v4_flash/kernels/rocm/backend.hip.cpp` | `"src/core/logging/logging.hpp"` | `<gufo/ext/host.hpp>` |
| `src/models/deepseek_v4_flash/kernels/rocm/backend.hip.cpp` | `"src/core/hip/managed_alloc.hpp"` | `<gufo/ext/experimental/managed_alloc.hpp>` |
| `src/models/deepseek_v4_flash/kernels/rocm/backend.hip.cpp` | `"src/core/hip/weight_upload.hpp"` | `<gufo/ext/experimental/weight_upload.hpp>` |
| `src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_runtime.hip.hpp` | `"src/core/platform/mapped_file.hpp"` | `<gufo/ext/platform/mapped_file.hpp>` |
| `src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_runtime.hip.hpp` | `"src/core/platform/os_info.hpp"` | `<gufo/ext/platform/os_info.hpp>` |
| `src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_runtime.hip.hpp` | `"src/core/hip/snapshot_transfer.hpp"` | `<gufo/ext/experimental/snapshot_transfer.hpp>` |
| `src/models/deepseek_v4_flash/kernels/rocm/resident_api.h` | `"src/core/platform/mapped_file.hpp"` | `<gufo/ext/platform/mapped_file.hpp>` |
| `src/models/deepseek_v4_flash/runtime/model_data.cpp` | `"src/core/logging/logging.hpp"` | `<gufo/ext/host.hpp>` |
| `src/models/deepseek_v4_flash/runtime/model_data.cpp` | `"src/core/mapped_prefetch.hpp"` | `<gufo/ext/mapped_prefetch.hpp>` |
| `src/models/deepseek_v4_flash/runtime/model_data.cpp` | `"src/core/platform/publish.hpp"` | `<gufo/ext/platform/publish.hpp>` |
| `src/models/deepseek_v4_flash/runtime/model_data_internal.h` | `"src/core/platform/mapped_file.hpp"` | `<gufo/ext/platform/mapped_file.hpp>` |
| `src/models/deepseek_v4_flash/runtime/rocm_graph.cpp` | `"src/core/logging/logging.hpp"` | `<gufo/ext/host.hpp>` |
| `src/models/deepseek_v4_flash/runtime/rocm_graph.cpp` | `"src/core/platform/fs.hpp"` | `<gufo/ext/platform/fs.hpp>` |
| `src/models/deepseek_v4_flash/runtime/sampling.cpp` | `"src/core/sampling.hpp"` | `<gufo/ext/sampling.hpp>` |
| `src/models/deepseek_v4_flash/tool_calls.cpp` | `"src/core/json.hpp"` | `<gufo/ext/json.hpp>` |

---

## Remaining Intentional Coupling

1. **Own root-relative includes** (`src/models/<self>/**`): Preserved temporarily until each model has its own extension migration card (Cards 13, 15, 16, 17, 18, 20, 21, 22).
2. **Family includes** (`src/families/qwen/**`): Valid dependency on the acyclic Qwen family libraries extracted in Card 07.
3. **Exempt test-only ASR serving edge**: `src/cli/serve/asr_service.hpp` and `src/cli/serve/audio_asr_api.hpp` in `src/models/qwen3_asr/tests/audio_api_test.cpp`. This edge is explicitly permitted under test trees and scheduled for removal in Card 12 (`gufo_test_support`).

