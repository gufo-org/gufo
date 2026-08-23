#ifndef STRIX_SERVER_INFERENCE_BACKEND_HPP_
#define STRIX_SERVER_INFERENCE_BACKEND_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen/qwen_chat_template.hpp"
#include "src/models/qwen/qwen_tokenizer.hpp"

namespace strix::hip {
class QwenGpuModel;
}

namespace strix::models::deepseek_v4_flash {
class Model;
}

namespace strix::server {

/// Thread-safe HTTP inference facade over shared immutable GPU model resources
/// and a bounded pool of request-owned executor sessions.
class InferenceBackend {
public:
  using CancellationCheck = std::function<bool()>;

  struct Result {
    std::string text;
    std::vector<tokenization::TokenId> tokens;
    std::size_t prompt_tokens = 0;
    std::size_t completion_tokens = 0;
    double ttft_ms = 0.0;
    double mean_inter_token_ms = 0.0;
    bool cancelled = false;
  };

  InferenceBackend();
  ~InferenceBackend();

  InferenceBackend(const InferenceBackend&) = delete;
  InferenceBackend& operator=(const InferenceBackend&) = delete;
  InferenceBackend(InferenceBackend&&) = delete;
  InferenceBackend& operator=(InferenceBackend&&) = delete;

  /// Loads weights from a GGUF file. Returns false and sets *error on failure.
  bool load(const std::string& model_path, std::string* error,
            std::uint32_t max_context = 4096, std::size_t session_count = 1);

#if defined(ENGINE_ENABLE_HIP)
  /// Installs a previously loaded model without duplicating mapped weights.
  bool load(std::shared_ptr<const hip::QwenGpuModel> model, std::string* error,
            std::uint32_t max_context = 4096, std::size_t session_count = 1);

  /// Installs a previously loaded DeepSeek model with request-owned sessions.
  bool load(std::shared_ptr<models::deepseek_v4_flash::Model> model,
            std::string* error, std::uint32_t max_context = 4096,
            std::size_t session_count = 1);
#endif

  /// Stable model identifier used in API responses.
  std::string model_id() const;

  /// Plain text completion (no chat framing).
  Result complete(std::string_view prompt, std::size_t max_tokens,
                  float temperature,
                  const CancellationCheck& is_cancelled = {});

  /// Framed chat conversation; returns the assistant reply.
  Result chat(const std::vector<tokenization::ChatMessage>& messages,
              std::size_t max_tokens, float temperature,
              const CancellationCheck& is_cancelled = {});

  /// Token count of raw text (no generation).
  std::size_t count_tokens(std::string_view text) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace strix::server

#endif  // STRIX_SERVER_INFERENCE_BACKEND_HPP_
