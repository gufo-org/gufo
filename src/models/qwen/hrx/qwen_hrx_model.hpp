#ifndef GUFO_MODELS_QWEN_HRX_QWEN_HRX_MODEL_HPP_
#define GUFO_MODELS_QWEN_HRX_QWEN_HRX_MODEL_HPP_

#include <hrx/hrx_runtime.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/model_config.hpp"
#include "src/models/qwen/hrx/qwen_hrx_contract.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::hrx {

struct HrxBufferBinding {
  hrx_buffer_t buffer{nullptr};
  std::size_t offset{0};
  std::size_t length{0};
};

/// Owns HRX imports of the read-only GGUF shards while retaining the reader
/// that owns the mapped host memory. Tensor bindings remain bounds-checked.
class QwenHrxModel {
public:
  ~QwenHrxModel();

  QwenHrxModel(const QwenHrxModel&) = delete;
  QwenHrxModel& operator=(const QwenHrxModel&) = delete;
  QwenHrxModel(QwenHrxModel&&) = delete;
  QwenHrxModel& operator=(QwenHrxModel&&) = delete;

  [[nodiscard]] static std::unique_ptr<QwenHrxModel> CreateFromGguf(
      std::shared_ptr<const core::GgufReader> reader, hrx_device_t device,
      std::string* error_msg = nullptr);

  [[nodiscard]] const models::QwenModelWeights& GetWeights() const noexcept {
    return weights_;
  }
  [[nodiscard]] const tokenization::QwenTokenizer& GetTokenizer()
      const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }
  [[nodiscard]] const QwenHrxArtifactContract& GetArtifactContract()
      const noexcept {
    return contract_;
  }
  [[nodiscard]] std::optional<HrxBufferBinding> Bind(
      const models::QwenTensorRef& tensor) const noexcept;

private:
  struct ImportedRegion {
    const void* host_data{nullptr};
    std::size_t size{0};
    hrx_buffer_t buffer{nullptr};
  };

  QwenHrxModel(std::shared_ptr<const core::GgufReader> reader,
               models::QwenModelWeights weights,
               std::shared_ptr<const tokenization::QwenTokenizer> tokenizer,
               QwenHrxArtifactContract contract,
               std::vector<ImportedRegion> regions);

  std::shared_ptr<const core::GgufReader> reader_;
  models::QwenModelWeights weights_;
  QwenHrxArtifactContract contract_;
  std::shared_ptr<const tokenization::QwenTokenizer> tokenizer_;
  std::vector<ImportedRegion> regions_;
};

}  // namespace gufo::hrx

#endif  // GUFO_MODELS_QWEN_HRX_QWEN_HRX_MODEL_HPP_
