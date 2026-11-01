#ifndef GUFO_TESTS_SUPPORT_MODEL_API_TEST_SUPPORT_HPP_
#define GUFO_TESTS_SUPPORT_MODEL_API_TEST_SUPPORT_HPP_

// In-tree test support fixtures for model API tests.
// Wraps HTTP and serving internals so model test sources do not include
// headers under src/cli/. Excluded from the public extension API and production
// builds.

#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/audio_asr_api.hpp"
#include "src/cli/serve/audio_stream.hpp"
#include "src/cli/serve/audio_tts_api.hpp"
#include "src/cli/serve/embedding_service.hpp"
#include "src/cli/serve/embeddings_api.hpp"
#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/image_api.hpp"
#include "src/cli/serve/tts_service.hpp"

namespace gufo::test {

// Test support marker function verifying link against gufo_test_support.
const char* ModelApiTestSupportVersion();

}  // namespace gufo::test

#endif  // GUFO_TESTS_SUPPORT_MODEL_API_TEST_SUPPORT_HPP_
