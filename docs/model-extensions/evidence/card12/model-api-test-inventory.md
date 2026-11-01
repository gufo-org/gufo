# Model API Test Includes Inventory — Card 12

This inventory documents all `src/cli/` header includes and compiled translation units across model API test executables prior to Card 12 migration, the migrated state using `gufo_test_support`, and the explicit catalog of remaining test coupling in `tests/models/`.

## 1. Migrated Model API Tests

| Test Executable | Source Path | CMake Declaration | Pre-Migration `src/cli/` Includes | Compiled `src/cli/` Sources | Migrated Include & Link |
|---|---|---|---|---|---|
| `qwen3_asr_audio_api_test` | `src/models/qwen3_asr/tests/audio_api_test.cpp` | `src/models/qwen3_asr/CMakeLists.txt` | `src/cli/serve/asr_service.hpp`<br>`src/cli/serve/audio_asr_api.hpp` | `src/cli/serve/audio_asr_api.cpp`<br>`src/cli/serve/asr_service.cpp` | `#include "tests/support/model_api_test_support.hpp"`<br>Link: `gufo_test_support` |
| `embedding_gemma2_embeddings_api_test` | `tests/models/embedding_gemma2/embeddings_api_test.cpp` | `tests/models/embedding_gemma2/CMakeLists.txt` | `src/cli/serve/embeddings_api.hpp`<br>`src/cli/serve/embedding_service.hpp` | `src/cli/serve/embeddings_api.cpp`<br>`src/cli/serve/embedding_service.cpp` | `#include "tests/support/model_api_test_support.hpp"`<br>Link: `gufo_test_support` |
| `qwen3_tts_audio_api_test` | `tests/models/qwen3_tts/audio_api_test.cpp` | `CMakeLists.txt` | `src/cli/serve/audio_stream.hpp`<br>`src/cli/serve/audio_tts_api.hpp`<br>`src/cli/serve/tts_service.hpp` | `src/cli/serve/audio_tts_api.cpp`<br>`src/cli/serve/tts_service.cpp` | `#include "tests/support/model_api_test_support.hpp"`<br>Link: `gufo_test_support` |
| `qwen_image_21_test` | `tests/models/qwen_image_21/model_test.cpp` | `src/models/qwen_image_21/CMakeLists.txt` | `src/cli/serve/http_server.hpp`<br>`src/cli/serve/image_api.hpp` | (none; linked `gufo_http`) | `#include "tests/support/model_api_test_support.hpp"`<br>Link: `gufo_test_support` (replaces `gufo_http`) |

### Detailed Symbols Used by Migrated Tests

#### 1.1 Qwen3-ASR (`qwen3_asr_audio_api_test`)
- Header includes removed:
  - `src/cli/serve/asr_service.hpp`
  - `src/cli/serve/audio_asr_api.hpp`
- Types and functions used from `gufo::server`:
  - `AsrService`
  - `AsrServiceOptions`
  - `HandleAudioAsrApiRequest`
  - `HttpRequest`
  - `HttpResponse`
- Production sources removed from test target:
  - `src/cli/serve/audio_asr_api.cpp`
  - `src/cli/serve/asr_service.cpp`
- Test target link: `gufo_qwen3_asr`, `gufo_test_support`.

#### 1.2 EmbeddingGemma 2 (`embedding_gemma2_embeddings_api_test`)
- Header includes removed:
  - `src/cli/serve/embeddings_api.hpp`
  - `src/cli/serve/embedding_service.hpp`
- Types and functions used from `gufo::server`:
  - `EmbeddingBatch`
  - `EmbeddingInput`
  - `EmbeddingPart`
  - `EmbeddingRequestOptions`
  - `EmbeddingService`
  - `EmbeddingServiceOptions`
  - `EmbeddingStatus`
  - `HandleEmbeddingsApiRequest`
  - `HttpRequest`
  - `HttpResponse`
  - `IsEmbeddingsApiPath`
  - `LoadSentenceTransformerPrompts`
  - `TruncateKeepingImages`
- Production sources removed from test target:
  - `src/cli/serve/embeddings_api.cpp`
  - `src/cli/serve/embedding_service.cpp`
- Test target link: `gufo_embedding_gemma2`, `gufo_test_support`, `gufo_logging`, `gufo_core`.

#### 1.3 Qwen3-TTS (`qwen3_tts_audio_api_test`)
- Header includes removed:
  - `src/cli/serve/audio_stream.hpp`
  - `src/cli/serve/audio_tts_api.hpp`
  - `src/cli/serve/tts_service.hpp`
- Types and functions used from `gufo::server`:
  - `HandleAudioTtsApiRequest`
  - `HttpRequest`
  - `HttpResponse`
  - `TtsService`
  - `TtsServiceOptions`
- Production sources removed from test target:
  - `src/cli/serve/audio_tts_api.cpp`
  - `src/cli/serve/tts_service.cpp`
- Test target link: `gufo_qwen3_tts`, `gufo_test_support`.

#### 1.4 Qwen-Image-21 (`qwen_image_21_test`)
- Header includes removed:
  - `src/cli/serve/http_server.hpp`
  - `src/cli/serve/image_api.hpp`
- Types and functions used from `gufo::server`:
  - `ImageService`
  - `HttpRequest`
  - `HttpResponse`
  - `HandleImageApiRequest`
- Production sources removed: none (previously linked `gufo_http`).
- Test target link: `gufo_qwen_image_21`, `gufo_test_support` (direct link to `gufo_http` removed).

---

## 2. Retained Test Coupling (Deferred to Future Cards)

Per RFC scope and Card 12 brief, text-generation runner and continuation identity tests in `tests/models/` retain their existing includes of `src/cli/` until subsequent RFC cards address the shared text contract (Card 19) and generated serve module (Card 25):

| Test Source File | Target Name | `src/cli/` Include | Justification & Destination Card |
|---|---|---|---|
| `tests/models/deepseek_v4_flash/serving_test.cpp` | `deepseek_v4_flash_serving_test` | `src/cli/serve/inference_backend.hpp` | Text runner harness; Card 19 / Card 22 |
| `tests/models/qwen38_flash_next/session_test.cpp` | `qwen38_flash_next_session_test` | `src/cli/serve/inference_backend.hpp` | Text runner harness; Card 19 / Card 21 |
| `tests/models/qwen27b/serving_test.cpp` | `qwen27b_serving_test` | `src/cli/serve/inference_backend.hpp` | Text runner harness; Card 19 / Card 20 |
| `tests/models/deepseek_v4_flash/continuation_identity_test.cpp` | `deepseek_v4_flash_continuation_identity_test` | `src/cli/serve/continuation_identity.hpp` | Continuation identity verification; Card 19 |
| `tests/models/qwen27b/continuation_identity_test.cpp` | `qwen27b_continuation_identity_test` | `src/cli/serve/continuation_identity.hpp` | Continuation identity verification; Card 19 |
| `tests/models/qwen38_flash_next/continuation_identity_test.cpp` | `qwen38_flash_next_continuation_identity_test` | `src/cli/serve/continuation_identity.hpp` | Continuation identity verification; Card 19 |
| `tests/models/qwen/chat_template_golden_helpers.hpp` | (included by chat template tests) | `src/cli/serve/chat_template.hpp` | Chat template test fixture; Card 19 |

None of these files reside under `src/models/` or any other `BOUNDARY_LAYERS` path.
`src/models/` now contains **0** includes of `src/cli/` across both production and test files.
