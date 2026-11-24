#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/primitives.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::ext::qwen::flash_next {
class WeightSource;
}  // namespace gufo::ext::qwen::flash_next

namespace gufo::models::qwen38_flash_next::rocm {

/// One weight resident in device memory, still in its GGUF encoding, or in
/// the encoding its weight source describes by `format_state` (Tensor).
struct DeviceTensor {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{0};
  std::uint32_t experts{1};
  const void* format_state{nullptr};

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] const float* f32() const noexcept {
    return static_cast<const float*>(data);
  }
};

/// The stage-facing handle of a device weight, and back.
[[nodiscard]] inline ext::qwen::flash_next::Tensor ToTensor(
    const DeviceTensor& d) noexcept {
  return {d.data, d.type, d.cols, d.rows, d.experts, d.format_state};
}
[[nodiscard]] inline DeviceTensor FromTensor(
    const ext::qwen::flash_next::Tensor& t) noexcept {
  return {t.data(), t.type(),    t.cols(),
          t.rows(), t.experts(), t.format_state()};
}

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
  /// Streams tensors from `source`'s regions; tensors with format_state go
  /// through WeightSource::Upload.
  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const ModelWeights& weights, const MtpWeights* mtp,
      ext::qwen::flash_next::WeightSource& source,
      std::string* error_msg = nullptr);

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
  /// The head draft tables copy rows from: output(), or the GGUF copy a
  /// weight source provided (UseDraftSource).
  const DeviceTensor& draft_source() const noexcept {
    return draft_source_.empty() ? output_ : draft_source_;
  }
  /// Bytes of one draft_source() row.
  std::size_t output_row_bytes() const noexcept {
    if (!draft_source_.empty())
      return draft_source_bytes_ / draft_source_.rows;
    return output_.rows == 0 ? 0 : output_bytes_ / output_.rows;
  }
  /// Drafts copy rows of `head` (a GGUF matrix equal to output(), owned by
  /// the caller) instead of output(). Before SetDraftVocab.
  [[nodiscard]] bool UseDraftSource(const DeviceTensor& head,
                                    std::string* error_msg);
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

private:
  friend struct DeviceModelTestAccess;  // synthetic models in GPU tests
  DeviceModel() = default;
  /// Uploads the trunk and the optional MTP block from `shards` (the trunk's
  /// first `trunk_shards`, then the draft block's).
  static std::unique_ptr<DeviceModel> UploadShards(
      const ModelWeights& weights, const MtpWeights* mtp,
      std::vector<core::GgufMappedRegion> shards, std::uint32_t trunk_shards,
      ext::qwen::flash_next::WeightSource* source, std::string* error_msg);

  Config config_;
  DeviceTensor token_embd_;
  DeviceTensor output_;
  std::size_t output_bytes_{0};
  DeviceTensor draft_output_;
  DeviceTensor draft_source_;
  std::size_t draft_source_bytes_{0};
  std::vector<std::int32_t> draft_specials_;
  DeviceMixer hc_head_;
  std::vector<DeviceLayer> layers_;
  DeviceLayer mtp_;
  bool has_mtp_{false};
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
  std::size_t max_half_cols_{1};
  std::size_t max_q8_cols_{32};
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_
