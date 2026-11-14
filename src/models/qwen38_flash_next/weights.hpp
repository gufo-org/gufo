#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_WEIGHTS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen38_flash_next/config.hpp"

namespace gufo::models::qwen38_flash_next {

/// Optional host destinations for independent predictor qualification.
/// Every populated span contains the final complete row of its stage.
struct MtpTrace {
  std::span<float> normalized_hidden;
  std::span<float> fused;
  std::span<float> attention;
  std::span<float> hidden;
  std::span<float> head;
  std::span<float> ffn_input;
  std::span<float> ffn_output;
  [[nodiscard]] bool Valid(std::size_t H, std::size_t D) const noexcept {
    for (const auto row : {normalized_hidden, fused, attention, hidden})
      if (!row.empty() && row.size() != D)
        return false;
    for (const auto row : {head, ffn_input, ffn_output})
      if (!row.empty() && row.size() != H)
        return false;
    return true;
  }
};

/// `TensorRef::shard` of a tensor whose payload lives in host memory the
/// binder prepared (a converted EXL3-checkpoint tensor), not in a mapped
/// file region.
inline constexpr std::uint32_t kHostShard = 0xFFFFFFFFU;

/// One EXL3 trellis linear, or a stack of expert linears of one shape, in a
/// checkpoint. The device stores each matrix's tiles column-block-major;
/// the optional 128-block maps reorder (or select) input and output blocks
/// while uploading, which commutes with the block-diagonal Hadamards.
struct Exl3Ref {
  struct Part {
    std::uint32_t file{0};         ///< Checkpoint file of all three.
    std::uint64_t file_offset{0};  ///< Trellis payload offset in that file.
    std::uint64_t suh_offset{0};   ///< [src_in] fp16
    std::uint64_t svh_offset{0};   ///< [src_out] fp16
    /// Mapped scales; reading them faults the pages in, so uploads read
    /// the file ranges instead.
    const std::uint16_t* suh{nullptr};
    const std::uint16_t* svh{nullptr};
  };
  int k{0};                  ///< Bits per weight of every part.
  std::uint32_t src_in{0};   ///< Stored input width.
  std::uint32_t src_out{0};  ///< Stored output width.
  /// Device input block i reads stored block in_blocks[i]; empty keeps all.
  std::vector<std::uint32_t> in_blocks;
  /// Device output block j reads stored block out_blocks[j]; empty keeps all.
  std::vector<std::uint32_t> out_blocks;
  std::vector<Part> parts;  ///< One per expert; one for a dense matrix.

  [[nodiscard]] std::uint32_t In() const noexcept {
    return in_blocks.empty()
               ? src_in
               : static_cast<std::uint32_t>(in_blocks.size()) * 128;
  }
  [[nodiscard]] std::uint32_t Out() const noexcept {
    return out_blocks.empty()
               ? src_out
               : static_cast<std::uint32_t>(out_blocks.size()) * 128;
  }
  /// Stored trellis bytes of one part.
  [[nodiscard]] std::size_t SourceBytes() const noexcept {
    return static_cast<std::size_t>(src_in) * src_out * k / 8;
  }
  /// Device tile bytes of one part.
  [[nodiscard]] std::size_t DeviceBytes() const noexcept {
    return static_cast<std::size_t>(In()) * Out() * k / 8;
  }
};

/// Non-owning view of one GGUF tensor. `rows` x `cols` follows the GGUF
/// convention: cols (ne[0]) is the contiguous reduction dimension, rows
/// (ne[1]) the output dimension, and `experts` (ne[2]) stacks whole matrices.
/// An EXL3 linear sets `exl3` (cols = in, rows = out); its `type` is unused.
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};
  std::uint64_t file_offset{0};  ///< Byte offset inside the owning shard.
  std::uint32_t shard{0};        ///< Mapped region index in the reader.
  std::string_view name{};
  std::shared_ptr<const Exl3Ref> exl3{};

  [[nodiscard]] bool empty() const noexcept {
    return data == nullptr && exl3 == nullptr;
  }
  [[nodiscard]] std::uint64_t ElementCount() const noexcept {
    return cols * rows * experts;
  }
  /// Encoded bytes of one row (`cols` elements) in this format.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows * experts;
  }
  /// Base address of one stacked expert matrix.
  [[nodiscard]] const std::uint8_t* Expert(std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) + RowBytes() * rows * e;
  }
};

/// Two hyper-connection mixers per layer plus the output head mixer share
/// this shape; `inject` is absent on the head mixer.
struct HcMixer {
  TensorRef norm;    ///< [hc_dim] grouped RMSNorm gamma, (1 + w) folded.
  TensorRef down;    ///< [hc_dim -> low_rank]
  TensorRef up;      ///< [low_rank -> hc_dim]
  TensorRef inject;  ///< [hc_dim -> hc_count], empty on the head mixer.
};

struct LayerWeights {
  bool linear{false};
  HcMixer hc_attn;
  HcMixer hc_ffn;

  // Gated DeltaNet (linear layers).
  TensorRef ssm_qkv;     ///< [hidden -> 2*key_dim + value_dim]
  TensorRef ssm_gate;    ///< [hidden -> value_dim], the z output gate.
  TensorRef ssm_conv1d;  ///< [conv_kernel, channels]
  TensorRef ssm_alpha;   ///< [hidden -> v_heads]
  TensorRef ssm_beta;    ///< [hidden -> v_heads]
  TensorRef ssm_dt;      ///< [v_heads] softplus bias
  TensorRef ssm_a;       ///< [v_heads] = -exp(A_log)
  TensorRef ssm_norm;    ///< [ssm_head_dim]
  TensorRef ssm_out;     ///< [value_dim -> hidden]

  // Gated GQA + QSA indexer (attention layers).
  TensorRef attn_q;       ///< [hidden -> heads * 2 * head_dim], q|gate per head
  TensorRef attn_k;       ///< [hidden -> kv_heads * head_dim]
  TensorRef attn_v;       ///< [hidden -> kv_heads * head_dim]
  TensorRef attn_out;     ///< [heads * head_dim -> hidden]
  TensorRef attn_q_norm;  ///< [head_dim]
  TensorRef attn_k_norm;  ///< [head_dim]
  TensorRef indexer_q;    ///< [hidden -> indexer_heads * indexer_dim]
  TensorRef indexer_k;    ///< [hidden -> indexer_dim]
  TensorRef indexer_q_norm;
  TensorRef indexer_k_norm;

  // PLE n-gram injection (one linear layer).
  TensorRef ple_key;         ///< [ple_dim -> hc_dim]
  TensorRef ple_value;       ///< [ple_dim -> hidden]
  TensorRef ple_norm_key;    ///< [hc_dim]
  TensorRef ple_norm_query;  ///< [hc_dim]
  TensorRef ple_norm_conv;   ///< [hc_dim]
  TensorRef ple_conv1d;      ///< [conv_kernel, hc_dim]

  // Mixture of experts.
  TensorRef router;          ///< [hidden -> num_experts] F32
  TensorRef ffn_gate_exps;   ///< [hidden -> expert_ff] x experts
  TensorRef ffn_up_exps;     ///< [hidden -> expert_ff] x experts
  TensorRef ffn_down_exps;   ///< [expert_ff -> hidden] x experts
  TensorRef shexp_gate_inp;  ///< [hidden] -> scalar sigmoid gate
  TensorRef shexp_gate;      ///< [hidden -> shared_ff]
  TensorRef shexp_up;        ///< [hidden -> shared_ff]
  TensorRef shexp_down;      ///< [shared_ff -> hidden]

  // Speculative `nextn` block only.
  TensorRef nextn_enorm;  ///< [hidden]
  TensorRef nextn_hnorm;  ///< [hc_dim], one RMS denominator across all streams
  TensorRef nextn_eh_proj;  ///< GGUF rows: [fc_embedding | fc_hidden]
  /// EXL3 checkpoints store the two halves of nextn_eh_proj separately.
  TensorRef nextn_fc_embedding;
  TensorRef nextn_fc_hidden;
  HcMixer nextn_head;  ///< Output mixer of the draft block.
};

namespace exl3 {
class Pack;
}

/// Host payloads a checkpoint binder converted (folded norms, permuted
/// rows, Q8_0 projections). TensorRefs with shard kHostShard point here.
struct HostTensors {
  std::deque<std::vector<std::uint8_t>> buffers;
  std::deque<std::string> names;
};

/// The EXL3 checkpoint's trellis n-gram table, read from disk only.
struct Exl3NgramRef {
  std::uint32_t file{0};         ///< Checkpoint file of the table.
  std::uint64_t file_offset{0};  ///< First row's offset in that file.
  std::uint64_t rows{0};
  int k{0};                              ///< Bits per symbol.
  std::vector<std::uint16_t> head_bias;  ///< [ple_heads][ple_head_dim] fp16
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;  ///< [hidden -> vocab]
  TensorRef output;      ///< [hidden -> vocab]
  HcMixer hc_head;       ///< Final mixer, doubles as the output norm.
  TensorRef ple_table;   ///< [ple_head_dim, ple_rows], read from disk only.
  /// The EXL3 checkpoint's n-gram table; ple_table is then empty.
  std::shared_ptr<const Exl3NgramRef> ple_exl3;
  std::vector<LayerWeights> layers;

  /// Binds and validates the trunk artifact. Tensor payloads stay mapped and
  /// untouched; only headers are read.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Binds an EXL3 checkpoint. Trellis payloads stay mapped; the small
  /// tensors whose layout or encoding differs from the GGUF artifact are
  /// converted into `host`, which must outlive the weights.
  [[nodiscard]] static std::optional<ModelWeights> BindExl3(
      const exl3::Pack& pack, HostTensors& host,
      std::string* error_msg = nullptr);
};

/// The MTP draft block from the `shared-*` sidecar: one attention layer with
/// its own hyper-connection mixers and experts; token embedding and LM head
/// are borrowed from the trunk.
struct MtpWeights {
  Config config;
  LayerWeights block;

  [[nodiscard]] static std::optional<MtpWeights> Bind(
      const core::GgufReader& reader, const Config& trunk,
      std::string* error_msg = nullptr);

  /// The `mtp.*` block of an EXL3 checkpoint, or nullopt when it has none.
  [[nodiscard]] static std::optional<MtpWeights> BindExl3(
      const exl3::Pack& pack, const Config& trunk, HostTensors& host,
      std::string* error_msg = nullptr);
};

/// Hyper-connection inject and GDN alpha/beta tensors of `layer` stored as
/// Q8_0. The official trunk GGUF keeps them in F32; the device expands Q8_0
/// copies to F32 at load, because their W8A8 route breaks copied identifiers.
[[nodiscard]] std::size_t ExpandedQ8_0Tensors(
    const LayerWeights& layer) noexcept;

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_WEIGHTS_HPP_
