#pragma once

#include <array>
#include <cstring>
#include <functional>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/models/inkling2/config.hpp"
#include "src/models/inkling2/tensor_ref.hpp"

namespace gufo::models::inkling2 {

/// Loop controller of one core layer for one stream (q, k, v or ffn):
/// `h = down . [rms(x); rms(msg)]`, then gain and forget gates of rank
/// `gate_rank`. All F32.
struct LoopController {
  TensorRef down;    // ne (2 * n_embd, gate_rank)
  TensorRef gain;    // ne (gate_rank, n_embd)
  TensorRef forget;  // ne (gate_rank, n_embd)
};

/// Index of a stream in `LayerWeights::loop` and `msg_query`.
enum Stream : std::size_t {
  kStreamQ,
  kStreamK,
  kStreamV,
  kStreamFfn,
  kStreams
};

/// Every tensor of one physical layer. Tensors a layer kind does not have are
/// left empty: `attn_norm` only on sliding-window layers, the `*_inp_norm`,
/// `attn_*_conv_proj` and a 3-row `attn_res_query` only on global layers,
/// `loop`/`msg_query` only on core layers, and `attn_res_query` not on layer 0.
struct LayerWeights {
  bool is_global = false;
  bool is_core = false;

  // Attention. ne in ggml order; heads: 24 query, 8 key/value, dim 128.
  TensorRef attn_norm;        // (n_embd), sliding-window layers
  TensorRef attn_q_inp_norm;  // (n_embd), global layers
  TensorRef attn_k_inp_norm;
  TensorRef attn_v_inp_norm;
  TensorRef attn_q;          // (n_embd, 3072)
  TensorRef attn_k;          // (n_embd, 1024)
  TensorRef attn_v;          // (n_embd, 1024)
  TensorRef attn_gate;       // (n_embd, 3072) output gate
  TensorRef attn_output;     // (3072, n_embd)
  TensorRef attn_rel;        // (n_embd, 384) relative states, 16 x 24 heads
  TensorRef attn_sink_proj;  // (n_embd, 24)
  TensorRef attn_q_clip;     // (128, 24)
  TensorRef attn_rel_clip;   // (16, 24)
  TensorRef attn_rel_bias;   // (16, 1024)
  TensorRef attn_k_conv;     // (4, 1024) static taps
  TensorRef attn_v_conv;     // (4, 1024)
  TensorRef
      attn_k_conv_proj;  // (n_embd, 32) input-dependent taps, global layers
  TensorRef attn_v_conv_proj;
  TensorRef attn_post_conv;  // (4, n_embd) conv on the attention output
  TensorRef ffn_post_conv;   // (4, n_embd) conv on the FFN output
  TensorRef attn_res_query;  // (n_embd), or (n_embd, 3) on global layers

  // Block-attention-residual query of the FFN input.
  TensorRef ffn_res_query;  // (n_embd)

  // MoE: 128 experts, no shared expert.
  TensorRef ffn_norm;      // (n_embd)
  TensorRef ffn_gate_inp;  // (n_embd, 128) router, F32
  TensorRef exp_probs_b;   // (128) selection bias, F32
  TensorRef
      ffn_gate_up_exps;     // (n_embd, 2 * n_ff, 128), rows gate/up interleaved
  TensorRef ffn_down_exps;  // (n_ff, n_embd, 128)
  /// Per-expert F32 scale (128), applied after the expert matmul. Present on
  /// the NVFP4 tensors of the Unsloth files, absent otherwise.
  TensorRef ffn_gate_up_scale;
  TensorRef ffn_down_scale;

  // Loop (core layers 6..17).
  std::array<LoopController, kStreams> loop;
  /// Message query per stream, each (n_embd) F32. Points into
  /// `blk.N.loop_msg_query`.
  std::array<TensorRef, kStreams> msg_query;
};

/// All weights of a model, bound to a mapped GGUF. Valid while `reader` lives.
struct Inkling2Weights {
  Inkling2Config config;
  TensorRef token_embd;        // (n_embd, n_vocab)
  TensorRef token_embd_norm;   // (n_embd)
  TensorRef output;            // (n_embd, n_vocab)
  TensorRef output_norm;       // (n_embd)
  TensorRef output_res_query;  // (n_embd)
  std::array<LayerWeights, Inkling2Config::kPhysLayers> layers;
  /// F32 copies of small tensors that the file stores as BF16 (the loop
  /// controllers of the Unsloth Q8_K_XL files). The TensorRefs of those tensors
  /// point into these buffers instead of the mapped file; the engine uploads
  /// them next to the file's own tensors.
  std::vector<std::shared_ptr<const std::vector<float>>> widened;

  /// Parses the metadata, then finds and checks every tensor: name, shape,
  /// type (supported set; F32 where the format needs it; NVFP4 only on
  /// experts, with its per-expert scale). Any other tensor in the file is an
  /// error. On failure returns nullopt and sets `*why`.
  [[nodiscard]] static std::optional<Inkling2Weights> Bind(
      const core::GgufReader& reader, std::string* why);
};

/// Calls `fn(name, tensor)` for every bound tensor under a layout-independent
/// name: `blk.N.<gguf suffix>`, `blk.N.msg_query.<q|k|v|ffn>` and
/// `blk.N.loop.<stream>.<down|gain|forget>`.
void ForEachTensor(
    const Inkling2Weights& weights,
    const std::function<void(const std::string&, const TensorRef&)>& fn);

/// Multi-line human summary: layout, tensors and bytes per type, expert types.
[[nodiscard]] std::string Summarize(const Inkling2Weights& weights);

}  // namespace gufo::models::inkling2
