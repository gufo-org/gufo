#ifndef GUFO_MODELS_QWEN_GENERATOR_HPP_
#define GUFO_MODELS_QWEN_GENERATOR_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/sampling.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::models {

/// Generation parameters for controlling the auto-regressive decode loop.
struct GenerationOptions {
  std::size_t max_new_tokens = 128;
  sampling::SamplingConfig sampling;
};

/// End-to-end Qwen auto-regressive generation engine.
class QwenGenerator {
public:
  QwenGenerator(QwenModelWeights weights,
                std::unique_ptr<tokenization::QwenTokenizer> tokenizer,
                std::uint32_t max_context = 4096);

  /// Creates a generator directly from a loaded GGUF file.
  [[nodiscard]] static std::unique_ptr<QwenGenerator> CreateFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Executes prompt prefill and decodes tokens, invoking on_token for each
  /// generated token. Generation halts if on_token returns false or when an EOS
  /// token is emitted.
  [[nodiscard]] std::vector<tokenization::TokenId> Generate(
      std::span<const tokenization::TokenId> prompt_tokens,
      const GenerationOptions& options,
      const std::function<bool(tokenization::TokenId, std::string_view)>&
          on_token = nullptr);

  /// Helper to generate from raw text prompt.
  [[nodiscard]] std::string GenerateText(std::string_view prompt,
                                         const GenerationOptions& options = {});

  [[nodiscard]] const tokenization::QwenTokenizer& GetTokenizer()
      const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }

private:
  QwenModelWeights weights_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  QwenKvCache kv_cache_;
  QwenSsmCache ssm_cache_;
  QwenScratchArena arena_;
};

}  // namespace gufo::models

#endif  // GUFO_MODELS_QWEN_GENERATOR_HPP_
