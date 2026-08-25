#ifndef STRIX_SERVER_TTS_SERVICE_HPP_
#define STRIX_SERVER_TTS_SERVICE_HPP_

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen3_tts/config.hpp"
#include "src/models/qwen3_tts/synthesis.hpp"

namespace strix::server {

struct TtsServiceOptions {
  using Runner =
      std::function<bool(const models::qwen3_tts::SynthesisRequest&,
                         const models::qwen3_tts::CancellationCheck&,
                         models::qwen3_tts::SynthesisResult*, std::string*)>;

  std::filesystem::path model_root;
  std::size_t native_context_tokens{4096};
  bool validate_model{true};
  models::qwen3_tts::ModelVariant variant{
      models::qwen3_tts::ModelVariant::kCustomVoice};
  std::string model_id;
  std::vector<std::string> voices;
  Runner runner;
};

/// Synchronous, bounded native HIP Qwen3-TTS service. The official Python
/// implementation is used only by offline validation tools.
class TtsService {
public:
  explicit TtsService(TtsServiceOptions options);
  ~TtsService();

  TtsService(const TtsService&) = delete;
  TtsService& operator=(const TtsService&) = delete;
  TtsService(TtsService&&) = delete;
  TtsService& operator=(TtsService&&) = delete;

  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] std::string initialization_error() const;
  [[nodiscard]] std::string model_id() const;
  [[nodiscard]] std::string backend_name() const;
  [[nodiscard]] std::vector<std::string> voices() const;
  [[nodiscard]] models::qwen3_tts::ModelVariant variant() const noexcept;

  [[nodiscard]] bool Synthesize(
      const models::qwen3_tts::SynthesisRequest& request,
      const models::qwen3_tts::CancellationCheck& is_cancelled,
      models::qwen3_tts::SynthesisResult* result, std::string* error = nullptr);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace strix::server

#endif  // STRIX_SERVER_TTS_SERVICE_HPP_
