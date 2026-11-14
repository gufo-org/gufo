#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next::exl3 {
class Pack;
}

namespace gufo::models::qwen38_flash_next::rocm {

/// One weight resident in device memory, still in its GGUF encoding, or an
/// EXL3 trellis linear: `data` then holds its column-block-major tiles
/// (exl3.hpp), cols/rows its input/output width and `exl3_k` its bits per
/// weight; `type` is kI32, which no GGUF kernel accepts.
struct DeviceTensor {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{0};
  std::uint32_t experts{1};
  const void* exl3_suh{nullptr};  ///< [experts][cols] fp16
  const void* exl3_svh{nullptr};  ///< [experts][rows] fp16
  int exl3_k{0};

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] bool IsExl3() const noexcept { return exl3_k != 0; }
  [[nodiscard]] const float* f32() const noexcept {
    return static_cast<const float*>(data);
  }
};

struct DeviceMixer {
  DeviceTensor norm;
  DeviceTensor down;
  DeviceTensor up;
  DeviceTensor inject;
};

struct DeviceLayer {
  bool linear{false};
  DeviceMixer hc_attn;
  DeviceMixer hc_ffn;

  DeviceTensor ssm_qkv, ssm_gate, ssm_conv1d, ssm_dt, ssm_a, ssm_norm, ssm_out;
  /// qkv and gate rows stacked ([hidden -> conv channels + value dim]) when
  /// both are Q8_0; then ssm_qkv/ssm_gate are empty.
  DeviceTensor ssm_in;
  /// alpha and beta rows stacked: [hidden -> 2 * v_heads] F16.
  DeviceTensor ssm_alpha_beta;
  DeviceTensor attn_q, attn_k, attn_v, attn_out, attn_q_norm, attn_k_norm,
      indexer_q, indexer_k, indexer_q_norm, indexer_k_norm;
  /// [q|gate ; k ; v] rows stacked when all are Q8_0; then attn_q/k/v are
  /// empty.
  DeviceTensor attn_qkv;
  DeviceTensor ple_key, ple_value, ple_norm_key, ple_norm_query, ple_norm_conv,
      ple_conv1d;
  /// Router rows followed by the shared-expert gate row:
  /// [hidden -> num_experts + 1] F16.
  DeviceTensor router;
  DeviceTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps, shexp_gate, shexp_up,
      shexp_down;
  DeviceTensor nextn_enorm, nextn_hnorm, nextn_fc_embedding, nextn_fc_hidden;
  DeviceMixer nextn_head;
};

/// How Upload stores a hyper-connection inject projection read in `disk`
/// format. The binder accepts F32, Q8_0 and BF16 rows, and each reaches the
/// device as F32 rows for the fused inject kernels, which compute a prompt
/// row the same way at any pass width. A separate 16-bit projection would
/// take the library GEMM, whose sums depend on the pass width, and on the
/// wide HC route it shares the half staging buffer with the mixed row.
enum class HcInjectUpload { kCopy, kExpandQ8_0, kWidenBF16 };

[[nodiscard]] constexpr HcInjectUpload HcInjectUploadFor(
    core::GgmlType disk) noexcept {
  switch (disk) {
    case core::GgmlType::kQ8_0:
      return HcInjectUpload::kExpandQ8_0;
    case core::GgmlType::kBF16:
      return HcInjectUpload::kWidenBF16;
    default:
      return HcInjectUpload::kCopy;
  }
}

/// Widens `count` BF16 values to F32 exactly: a BF16 value is the upper half
/// of an F32.
void WidenBF16(const std::uint16_t* src, float* dst,
               std::size_t count) noexcept;

/// The trunk (and optionally the MTP draft block) uploaded to the GPU. The
/// n-gram table is never uploaded: it is read from disk per token.
class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  /// Streams tensors from the same open files used to bind their metadata.
  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const ModelWeights& weights, const core::GgufReader& reader,
      const MtpWeights* mtp, const core::GgufReader* mtp_reader,
      std::string* error_msg = nullptr);

  /// Uploads an EXL3 checkpoint bound by ModelWeights::BindExl3 (and its
  /// MTP block): trellis payloads are read from the checkpoint files and
  /// repacked on the device, converted tensors are copied from host memory.
  [[nodiscard]] static std::unique_ptr<DeviceModel> UploadExl3(
      const ModelWeights& weights, const exl3::Pack& pack,
      const MtpWeights* mtp, std::string* error_msg = nullptr);

  const Config& config() const noexcept { return config_; }
  const DeviceTensor& token_embd() const noexcept { return token_embd_; }
  const DeviceTensor& output() const noexcept { return output_; }
  /// The leading output() rows the MTP draft head scores (see
  /// draft_vocab.hpp): a Q4_K copy of a Q8_0 head, else those rows in
  /// place; empty when the draft projects the full vocabulary.
  const DeviceTensor& draft_output() const noexcept { return draft_output_; }
  /// Special tokens above the prefix: the first rows of each session's
  /// draft table.
  std::span<const std::int32_t> draft_specials() const noexcept {
    return draft_specials_;
  }
  /// The head drafts copy rows from: output(), or its Q8_0 expansion when
  /// output() is an EXL3 linear (ExpandHeadQ8).
  const DeviceTensor& draft_source() const noexcept {
    return head_q8_.empty() ? output_ : head_q8_;
  }
  /// Bytes of one draft_source() row.
  std::size_t output_row_bytes() const noexcept {
    if (!head_q8_.empty())
      return static_cast<std::size_t>(head_q8_.cols) / 32 * 34;
    return output_.rows == 0 ? 0 : output_bytes_ / output_.rows;
  }
  /// Expands an EXL3 output head once into the Q8_0 matrix it equals, the
  /// source of the draft vocabulary (SetDraftVocab) and of draft tables.
  [[nodiscard]] bool ExpandHeadQ8(std::string* error_msg);
  /// Drafts score output() rows [0, prefix_rows) plus a per-session table
  /// that starts with `specials`. The rows of a Q8_0 head are requantized
  /// to Q4_K once, here; the table keeps the head's format. Draft proposals
  /// only steer speculation: the target still verifies every token, so greedy
  /// text is unchanged. Sampled text can differ: a smaller head changes
  /// acceptance, and so the later random draws.
  [[nodiscard]] bool SetDraftVocab(std::uint32_t prefix_rows,
                                   std::vector<std::int32_t> specials,
                                   std::string* error_msg);
  const DeviceMixer& hc_head() const noexcept { return hc_head_; }
  const std::vector<DeviceLayer>& layers() const noexcept { return layers_; }
  [[nodiscard]] bool has_mtp() const noexcept { return has_mtp_; }
  const DeviceLayer& mtp() const noexcept { return mtp_; }
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }
  /// Widest K among the BF16/F16 matrices (activation staging for hipBLAS).
  [[nodiscard]] std::size_t max_half_cols() const noexcept {
    return max_half_cols_;
  }
  /// Widest K among the Q8_0 matrices (decode activation quantization).
  [[nodiscard]] std::size_t max_q8_cols() const noexcept {
    return max_q8_cols_;
  }
  /// Widest input and output among the EXL3 linears (0 for GGUF models).
  [[nodiscard]] std::size_t max_exl3_in() const noexcept {
    return max_exl3_in_;
  }
  [[nodiscard]] std::size_t max_exl3_out() const noexcept {
    return max_exl3_out_;
  }

private:
  friend struct DeviceModelTestAccess;  // synthetic models in GPU tests
  DeviceModel() = default;

  Config config_;
  DeviceTensor token_embd_;
  DeviceTensor output_;
  std::size_t output_bytes_{0};
  DeviceTensor draft_output_;
  DeviceTensor head_q8_;
  std::vector<std::int32_t> draft_specials_;
  DeviceMixer hc_head_;
  std::vector<DeviceLayer> layers_;
  DeviceLayer mtp_;
  bool has_mtp_{false};
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
  std::size_t max_half_cols_{1};
  std::size_t max_q8_cols_{32};
  std::size_t max_exl3_in_{0};
  std::size_t max_exl3_out_{0};
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_
