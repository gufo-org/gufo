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
#include "src/core/hrx/hrx_buffer_binding.hpp"
#include "src/core/model_config.hpp"
#include "src/models/qwen/hrx/qwen_hrx_contract.hpp"
#include "src/models/qwen/hrx/qwen_hrx_tensor_binding.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::hrx {

/// Immutable native-Q8_0 bindings for one production layer. Fields that do
/// not apply to the layer kind remain invalid.
struct QwenHrxLayerBindings {
  bool is_full_attention{false};
  HrxBufferBinding attn_norm;
  HrxBufferBinding ffn_norm;
  HrxBufferBinding ffn_gate;
  HrxBufferBinding ffn_up;
  /// Valid only when ffn_gate and ffn_up are adjacent in one HRX buffer.
  HrxBufferBinding ffn_gate_up;
  HrxBufferBinding ffn_down;

  HrxBufferBinding attn_q;
  HrxBufferBinding attn_k;
  HrxBufferBinding attn_v;
  HrxBufferBinding attn_output;
  HrxBufferBinding attn_q_norm;
  HrxBufferBinding attn_k_norm;

  HrxBufferBinding attn_qkv;
  HrxBufferBinding attn_gate;
  HrxBufferBinding ssm_a;
  HrxBufferBinding ssm_conv1d;
  HrxBufferBinding ssm_dt;
  HrxBufferBinding ssm_alpha;
  HrxBufferBinding ssm_beta;
  /// Valid only when ssm_alpha and ssm_beta are adjacent in one HRX buffer.
  HrxBufferBinding ssm_alpha_beta;
  HrxBufferBinding ssm_norm;
  HrxBufferBinding ssm_out;
};

struct QwenHrxWeightBindings {
  HrxBufferBinding token_embedding;
  HrxBufferBinding output_norm;
  HrxBufferBinding output;
  std::vector<QwenHrxLayerBindings> layers;
};

/// Owns read-only HRX buffers for the GGUF shards. Tensor bindings remain
/// bounds-checked and do not depend on the GGUF mapping after construction.
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

  [[nodiscard]] const tokenization::QwenTokenizer& GetTokenizer()
      const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return config_;
  }
  [[nodiscard]] bool UsesDeviceLocalWeights() const noexcept {
    return uses_device_local_weights_;
  }
  [[nodiscard]] const QwenHrxArtifactContract& GetArtifactContract()
      const noexcept {
    return contract_;
  }
  [[nodiscard]] const QwenHrxWeightBindings& GetNativeBindings()
      const noexcept {
    return native_bindings_;
  }
  /// Binds a contiguous GGUF tensor payload only when its encoded type and
  /// logical element count exactly match the artifact ABI. Exact type and
  /// elements establish the contiguous payload size; arbitrary tensor stride
  /// metadata is not represented by QwenTensorRef and is not accepted here.
  [[nodiscard]] std::optional<HrxBufferBinding> BindTensor(
      const models::QwenTensorRef& tensor, core::GgmlType expected_type,
      std::size_t expected_elements,
      std::string* error_msg = nullptr) const noexcept;

  /// Typed convenience for row-major BF16 artifact matrices.
  [[nodiscard]] std::optional<HrxBufferBinding> BindBf16Matrix(
      const models::QwenTensorRef& tensor, std::size_t rows,
      std::size_t columns, std::string* error_msg = nullptr) const noexcept;
  [[nodiscard]] std::optional<HrxBufferBinding> BindQ8_0Matrix(
      const models::QwenTensorRef& tensor, std::size_t rows,
      std::size_t columns, std::string* error_msg = nullptr) const noexcept;
  [[nodiscard]] std::optional<HrxBufferBinding> BindF32Vector(
      const models::QwenTensorRef& tensor, std::size_t elements,
      std::string* error_msg = nullptr) const noexcept;

private:
  struct ImportedRegion {
    const void* host_data{nullptr};
    std::size_t size{0};
    hrx_buffer_t buffer{nullptr};

    ImportedRegion() = default;
    ImportedRegion(const void* data, std::size_t byte_size,
                   hrx_buffer_t owned_buffer) noexcept;
    ~ImportedRegion();
    ImportedRegion(const ImportedRegion&) = delete;
    ImportedRegion& operator=(const ImportedRegion&) = delete;
    ImportedRegion(ImportedRegion&& other) noexcept;
    ImportedRegion& operator=(ImportedRegion&& other) noexcept;
  };

  QwenHrxModel(core::ModelConfig config,
               std::shared_ptr<const tokenization::QwenTokenizer> tokenizer,
               QwenHrxArtifactContract contract,
               std::vector<ImportedRegion> regions,
               bool uses_device_local_weights);

  /// Low-level storage lookup after the typed payload contract is validated.
  [[nodiscard]] std::optional<HrxBufferBinding> BindStorage(
      const models::QwenTensorRef& tensor) const noexcept;
  [[nodiscard]] bool BuildNativeQ8Bindings(
      const models::QwenModelWeights& weights, std::string* error_msg);

  core::ModelConfig config_;
  QwenHrxArtifactContract contract_;
  std::shared_ptr<const tokenization::QwenTokenizer> tokenizer_;
  std::vector<ImportedRegion> regions_;
  QwenHrxWeightBindings native_bindings_;
  bool uses_device_local_weights_{false};
};

}  // namespace gufo::hrx

#endif  // GUFO_MODELS_QWEN_HRX_QWEN_HRX_MODEL_HPP_
