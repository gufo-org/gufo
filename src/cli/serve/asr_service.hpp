#ifndef GUFO_SERVER_ASR_SERVICE_HPP_
#define GUFO_SERVER_ASR_SERVICE_HPP_

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include "gufo/ext/asr.hpp"

namespace gufo::server {

struct AsrServiceOptions {
  using Runner = std::function<bool(const ext::TranscriptionRequest&,
                                    const ext::CancellationCheck&,
                                    ext::TranscriptionResult*, std::string*)>;

  std::filesystem::path model_root;
  std::size_t native_context_tokens{1024};
  bool validate_model{true};
  std::string model_id;
  Runner runner;
};

/// Synchronous, bounded native HIP Qwen3-ASR-1.7B service.
class AsrService {
public:
  explicit AsrService(AsrServiceOptions options);
  ~AsrService();

  AsrService(const AsrService&) = delete;
  AsrService& operator=(const AsrService&) = delete;
  AsrService(AsrService&&) = delete;
  AsrService& operator=(AsrService&&) = delete;

  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] std::string initialization_error() const;
  [[nodiscard]] std::string model_id() const;
  [[nodiscard]] std::string backend_name() const;

  [[nodiscard]] bool Transcribe(const ext::TranscriptionRequest& request,
                                const ext::CancellationCheck& is_cancelled,
                                ext::TranscriptionResult* result,
                                std::string* error = nullptr);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_ASR_SERVICE_HPP_
