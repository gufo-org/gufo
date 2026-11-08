# Pilot Inventory: EmbeddingGemma 2 Decoupling

Date: 2026-11-01
Scope: Inventory of references from `src/cli/` to `src/models/embedding_gemma2/` prior to Card 13 migration.

## Summary

Prior to Card 13, `src/cli/` held 8 direct `#include` statements targeting `src/models/embedding_gemma2/` headers across 2 serving files, plus symbol and model-specific configuration usage in 3 files (`src/cli/serve/embedding_service.hpp`, `src/cli/serve/embedding_service.cpp`, and `src/cli/serve/embeddings_api.cpp`), and CLI startup references in `src/cli/serve/serve.cpp`.

## Direct Header Includes (8 total)

### `src/cli/serve/embedding_service.hpp` (3 includes)
1. Line 13: `#include "src/models/embedding_gemma2/hip/encoder_runtime.hpp"`
   - Referenced symbol: `models::embedding_gemma2::hip::kMaxSequenceTokens` (used to define `EmbeddingService::kMaxInputTokens = 8192`)
2. Line 14: `#include "src/models/embedding_gemma2/image_processor.hpp"`
   - Referenced symbol: `models::embedding_gemma2::kDefaultImageSoftTokens` (used as default for `EmbeddingRequestOptions::image_soft_tokens = 280`)
3. Line 15: `#include "src/models/embedding_gemma2/video_processor.hpp"`
   - Referenced symbol: `models::embedding_gemma2::VideoOptions` (used as member `EmbeddingRequestOptions::video`)

### `src/cli/serve/embedding_service.cpp` (5 includes)
4. Line 20: `#include "src/models/embedding_gemma2/audio_decoder.hpp"`
   - Referenced symbol: `models::embedding_gemma2::DecodeAudio`
5. Line 21: `#include "src/models/embedding_gemma2/audio_processor.hpp"`
   - Referenced symbols: `models::embedding_gemma2::ValidateAudioProcessor`, `models::embedding_gemma2::ProcessAudio`, `kBeginAudioToken`, `kAudioToken`, `kEndAudioToken`, `kAudioTokenId`
6. Line 22: `#include "src/models/embedding_gemma2/image_processor.hpp"`
   - Referenced symbols: `models::embedding_gemma2::ValidateImageProcessor`, `models::embedding_gemma2::ProcessImage`, `kBeginImageToken`, `kImageToken`, `kEndImageToken`, `kImageTokenId`
7. Line 23: `#include "src/models/embedding_gemma2/loader.hpp"`
   - Referenced symbol: `models::embedding_gemma2::LooksLikeEmbeddingGemma2`
8. Line 24: `#include "src/models/embedding_gemma2/tokenizer.hpp"`
   - Referenced symbol: `models::embedding_gemma2::Tokenizer`

## Symbol and Configuration Leaks into Serving

### `src/cli/serve/embeddings_api.cpp`
- Lines 542-543: Referenced `models::embedding_gemma2::kMaxImageSoftTokens` and `models::embedding_gemma2::SupportedImageSoftTokens` via transitive include of `src/models/embedding_gemma2/image_processor.hpp`.

### `src/cli/serve/serve.cpp`
- Line 986: `ModelLoadLog load_log("embedding_gemma2", model);`
- Lines 989, 995, 1015: Direct instantiation of `server::EmbeddingService` and usage of `server::LoadSentenceTransformerPrompts`.

## CMake Target Coupling in root `CMakeLists.txt`

- Line 187: Hardcoded `add_subdirectory(src/models/embedding_gemma2)` in root build.
- Line 458: `target_link_libraries(gufo_http PUBLIC ... gufo_embedding_gemma2 ...)`
- Line 488: `target_link_libraries(gufo PRIVATE ... gufo_embedding_gemma2)`

## Decoupling Strategy (Card 13)

1. Define capability-neutral types (`EmbeddingBatch`, `EmbeddingInput`, `EmbeddingPart`, `EmbeddingStatus`, `EmbeddingVideoOptions`, `EmbeddingRequestOptions`, `EmbeddingRunner`) under public extension API `include/gufo/ext/embeddings.hpp`.
2. Move native model execution (`RunNative`, `EncoderRuntime`, `Tokenizer`, `ProcessImage`, `ProcessAudio`, `ProcessVideo`, `AudioDecoder`) entirely under `src/models/embedding_gemma2/` behind `ext::EmbeddingRunner`.
3. Provide pure artifact probe and artifact resolution within `src/models/embedding_gemma2/`.
4. Register provider `embeddinggemma-2` via generated constants `embedding_gemma2_constants.hpp` through `embedding_gemma2_provider_register`.
5. Remove all 8 `#include "src/models/embedding_gemma2/..."` statements from `src/cli/serve/` so `src/cli/` has 0 includes of `src/models/embedding_gemma2/`.
6. Enforce zero includes with ratchet in `tools/ci/check-extension-isolation.py`.
