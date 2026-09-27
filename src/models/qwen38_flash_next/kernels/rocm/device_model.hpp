#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/distributed/tp_partition.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

/// One weight resident in device memory, still in its GGUF encoding.
struct DeviceTensor {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{0};
  std::uint32_t experts{1};

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
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

/// The heads of a GDN or attention block one rank computes: all of them on
/// one host and in the draft block, the rank's share of every trunk layer
/// under TP.
struct MixerHeads {
  std::uint32_t ssm_k{0};     ///< GDN key heads
  std::uint32_t ssm_v{0};     ///< GDN value heads
  std::uint32_t ssm_dim{0};   ///< GDN head width
  std::uint32_t attn{0};      ///< attention query heads
  std::uint32_t attn_kv{0};   ///< attention KV heads
  std::uint32_t attn_dim{0};  ///< attention head width

  [[nodiscard]] static MixerHeads All(const Config& c) noexcept {
    return {c.ssm_num_k_heads, c.ssm_num_v_heads, c.ssm_head_dim,
            c.num_heads,       c.num_kv_heads,    c.head_dim};
  }
  [[nodiscard]] std::uint32_t SsmKeyDim() const noexcept {
    return ssm_k * ssm_dim;
  }
  [[nodiscard]] std::uint32_t SsmValueDim() const noexcept {
    return ssm_v * ssm_dim;
  }
  [[nodiscard]] std::uint32_t SsmConvChannels() const noexcept {
    return 2 * SsmKeyDim() + SsmValueDim();
  }
  [[nodiscard]] std::uint32_t AttentionQDim() const noexcept {
    return attn * attn_dim;
  }
  [[nodiscard]] std::uint32_t AttentionKvDim() const noexcept {
    return attn_kv * attn_dim;
  }
};

struct DeviceLayer {
  bool linear{false};
  /// The heads of this layer's GDN or attention block this rank computes.
  MixerHeads heads;
  /// Under TP the block's heads are split across ranks, so its output
  /// projection yields this rank's partial, which a second sum completes.
  bool mixer_split{false};
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
  /// Under TP the shared expert is split across ranks: this rank holds its
  /// share of the intermediate rows of gate/up and the matching columns of
  /// down, so its output is a partial the MoE all-reduce completes. False when
  /// the weights cannot be split on a block boundary; then every rank holds
  /// the whole shared expert and only rank zero keeps its output.
  bool shexp_split{false};
  DeviceTensor nextn_enorm, nextn_hnorm, nextn_fc_embedding, nextn_fc_hidden;
  DeviceMixer nextn_head;
};

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
      std::string* error_msg = nullptr,
      const distributed::TpPartition* partition = nullptr);

  const Config& config() const noexcept { return config_; }
  const DeviceTensor& token_embd() const noexcept { return token_embd_; }
  const DeviceTensor& output() const noexcept { return output_; }
  const DeviceMixer& hc_head() const noexcept { return hc_head_; }
  const std::vector<DeviceLayer>& layers() const noexcept { return layers_; }
  [[nodiscard]] bool has_mtp() const noexcept { return has_mtp_; }
  const DeviceLayer& mtp() const noexcept { return mtp_; }
  /// The heads of every trunk layer on this rank; the session state of the
  /// trunk's GDN and attention layers has this geometry.
  const MixerHeads& trunk_heads() const noexcept { return trunk_heads_; }
  [[nodiscard]] bool trunk_split() const noexcept { return trunk_split_; }
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }
  [[nodiscard]] std::uint32_t tp_rank() const noexcept { return tp_rank_; }
  [[nodiscard]] std::uint32_t tp_world_size() const noexcept {
    return tp_world_size_;
  }
  /// Widest K among the BF16/F16 matrices (activation staging for hipBLAS).
  [[nodiscard]] std::size_t max_half_cols() const noexcept {
    return max_half_cols_;
  }
  /// Widest K among the Q8_0 matrices (decode activation quantization).
  [[nodiscard]] std::size_t max_q8_cols() const noexcept {
    return max_q8_cols_;
  }

private:
  DeviceModel() = default;

  Config config_;
  DeviceTensor token_embd_;
  DeviceTensor output_;
  DeviceMixer hc_head_;
  std::vector<DeviceLayer> layers_;
  DeviceLayer mtp_;
  bool has_mtp_{false};
  MixerHeads trunk_heads_;
  bool trunk_split_{false};
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
  std::uint32_t tp_rank_{0};
  std::uint32_t tp_world_size_{1};
  std::size_t max_half_cols_{1};
  std::size_t max_q8_cols_{32};
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_
