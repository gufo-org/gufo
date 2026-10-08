#include "src/models/gemma4/weights.hpp"

#include <cmath>
#include <cstring>
#include <initializer_list>
#include <string>
#include <utility>

namespace gufo::models::gemma4 {
namespace {

using core::GgmlType;

struct Format {
  std::uint32_t block;
  std::uint32_t bytes;
};

/// Storage geometry of the formats this runtime decodes. Unknown types get a
/// zero block, which every validation below rejects.
[[nodiscard]] Format FormatOf(GgmlType type) noexcept {
  switch (type) {
    case GgmlType::kF32:
      return {1, 4};
    case GgmlType::kF16:
    case GgmlType::kBF16:
      return {1, 2};
    case GgmlType::kQ8_0:
      return {32, 34};
    case GgmlType::kQ4_0:
      return {32, 18};
    case GgmlType::kQ5_1:
      return {32, 24};
    case GgmlType::kQ4_K:
      return {256, 144};
    case GgmlType::kQ5_K:
      return {256, 176};
    case GgmlType::kQ6_K:
      return {256, 210};
    default:
      return {0, 0};
  }
}

/// Projection formats with decode GEMV, batched verification and prefill
/// GEMM kernels. BF16 projections run as binary16 (converted at upload).
constexpr std::initializer_list<GgmlType> kProjection = {
    GgmlType::kQ8_0, GgmlType::kQ4_0, GgmlType::kQ4_K, GgmlType::kQ5_K,
    GgmlType::kQ6_K, GgmlType::kF16,  GgmlType::kBF16};
/// Embedding rows are decoded one token at a time; the tied LM head also
/// needs a GEMV, which every listed format has.
constexpr std::initializer_list<GgmlType> kEmbedding = {
    GgmlType::kBF16, GgmlType::kQ8_0, GgmlType::kQ4_0,
    GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K};
constexpr std::initializer_list<GgmlType> kVector = {GgmlType::kF32};
/// Routed expert formats with decode, verification and prefill kernels: the
/// fused gate/up projection reduces over the hidden width, the down
/// projection over the (not 256-aligned) expert width. BF16 experts run as
/// binary16, like BF16 projections.
constexpr std::initializer_list<GgmlType> kExpertGateUp = {
    GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K,
    GgmlType::kQ8_0, GgmlType::kF16,  GgmlType::kBF16};
constexpr std::initializer_list<GgmlType> kExpertDown = {
    GgmlType::kQ5_1, GgmlType::kQ8_0, GgmlType::kF16, GgmlType::kBF16};

struct Binder {
  const core::GgufReader& reader;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  /// Binds `name` with the exact shape and one of the accepted formats;
  /// `experts` > 1 expects a stack of that many [cols, rows] matrices.
  TensorRef Get(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                std::initializer_list<GgmlType> types, bool required = true,
                std::uint64_t experts = 1) {
    TensorRef t;
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      if (required) {
        Fail("missing tensor " + name);
      }
      return t;
    }
    const auto& d = info->dimensions;
    const std::uint64_t got_cols = d.size() > 0 ? d[0] : 1;
    const std::uint64_t got_rows = d.size() > 1 ? d[1] : 1;
    const std::uint64_t got_experts = d.size() > 2 ? d[2] : 1;
    if (got_cols != cols || got_rows != rows || got_experts != experts ||
        d.size() > 3) {
      Fail("tensor " + name + " has shape [" + std::to_string(got_cols) + ", " +
           std::to_string(got_rows) + ", " + std::to_string(got_experts) +
           "], expected [" + std::to_string(cols) + ", " +
           std::to_string(rows) + ", " + std::to_string(experts) + "]");
      return t;
    }
    bool type_ok = false;
    for (auto type : types) {
      type_ok = type_ok || info->type == type;
    }
    const Format format = FormatOf(info->type);
    if (!type_ok || format.block == 0 || cols % format.block != 0) {
      Fail("tensor " + name + " has unsupported format " +
           std::string(core::ToString(info->type)));
      return t;
    }
    t.data = info->data;
    t.type = info->type;
    t.cols = cols;
    t.rows = rows;
    t.experts = experts;
    t.name = info->name;
    // Locate the shard so disk readers can address the payload directly and
    // a payload running past its shard is caught here, not as a fault later.
    const auto regions = reader.GetMappedRegions();
    const auto address = reinterpret_cast<std::uintptr_t>(info->data);
    bool inside = false;
    for (std::uint32_t i = 0; i < regions.size(); ++i) {
      const auto base = reinterpret_cast<std::uintptr_t>(regions[i].data);
      if (address >= base && address < base + regions[i].size) {
        t.shard = i;
        t.file_offset = address - base;
        inside = t.SizeBytes() <= regions[i].size - t.file_offset;
        break;
      }
    }
    if (!inside) {
      Fail("tensor " + name + " is truncated");
      return TensorRef{};
    }
    return t;
  }

  /// Reads a one-element F32 tensor such as layer_output_scale.
  float Scalar(const std::string& name) {
    const TensorRef t = Get(name, 1, 1, kVector);
    if (!ok) {
      return 0.0F;
    }
    float value = 0.0F;
    std::memcpy(&value, t.data, sizeof(value));
    if (!std::isfinite(value)) {
      Fail("tensor " + name + " is not finite");
    }
    return value;
  }

  RopeFactors Factors(const Config& c) {
    RopeFactors f;
    const TensorRef t =
        Get("rope_freqs.weight", c.head_dim_global / 2, 1, kVector);
    if (!ok) {
      return f;
    }
    f.values.resize(c.head_dim_global / 2);
    std::memcpy(f.values.data(), t.data, f.values.size() * sizeof(float));
    for (float v : f.values) {
      if (!std::isfinite(v) || !(v > 0.0F)) {
        Fail("rope_freqs.weight must contain positive finite divisors");
        break;
      }
    }
    return f;
  }

  /// Binds one block. `with_kv` is false for Q-only draft layers.
  LayerWeights Layer(const Config& c, std::uint32_t l, bool with_kv) {
    LayerWeights w;
    const std::string p = "blk." + std::to_string(l) + ".";
    const std::uint64_t hidden = c.hidden_size;
    const std::uint64_t head_dim = c.HeadDim(l);
    w.attn_norm = Get(p + "attn_norm.weight", hidden, 1, kVector);
    w.attn_q = Get(p + "attn_q.weight", hidden, c.QDim(l), kProjection);
    w.attn_q_norm = Get(p + "attn_q_norm.weight", head_dim, 1, kVector);
    if (with_kv) {
      w.attn_k = Get(p + "attn_k.weight", hidden, c.KvDim(l), kProjection);
      w.attn_v =
          Get(p + "attn_v.weight", hidden, c.KvDim(l), kProjection, false);
      w.attn_k_norm = Get(p + "attn_k_norm.weight", head_dim, 1, kVector);
    } else if (reader.FindTensor(p + "attn_k.weight") != nullptr ||
               reader.FindTensor(p + "attn_v.weight") != nullptr) {
      Fail("shared-KV draft layer " + std::to_string(l) +
           " must not carry K/V projections");
    }
    w.attn_output =
        Get(p + "attn_output.weight", c.QDim(l), hidden, kProjection);
    w.post_attn_norm =
        Get(p + "post_attention_norm.weight", hidden, 1, kVector);
    w.ffn_norm = Get(p + "ffn_norm.weight", hidden, 1, kVector);
    w.ffn_gate = Get(p + "ffn_gate.weight", hidden, c.ffn_size, kProjection);
    w.ffn_up = Get(p + "ffn_up.weight", hidden, c.ffn_size, kProjection);
    w.ffn_down = Get(p + "ffn_down.weight", c.ffn_size, hidden, kProjection);
    w.post_ffn_norm = Get(p + "post_ffw_norm.weight", hidden, 1, kVector);
    if (reader.FindTensor(p + "layer_output_scale.weight") != nullptr) {
      w.output_scale = Scalar(p + "layer_output_scale.weight");
    }
    if (c.HasExperts()) {
      Experts(c, p, &w);
    } else if (reader.FindTensor(p + "ffn_gate_inp.weight") != nullptr) {
      Fail("layer " + std::to_string(l) + " carries experts in a dense model");
    }
    return w;
  }

  /// The router and the routed experts of an expert layer. Only the fused
  /// gate/up layout is supported.
  void Experts(const Config& c, const std::string& p, LayerWeights* w) {
    const std::uint64_t hidden = c.hidden_size;
    const std::uint64_t experts = c.num_experts;
    const std::uint64_t width = c.expert_ffn_size;
    w->router = Get(p + "ffn_gate_inp.weight", hidden, experts, kVector);
    w->router_scale = Get(p + "ffn_gate_inp.scale", hidden, 1, kVector);
    w->pre_ffn_norm_2 = Get(p + "pre_ffw_norm_2.weight", hidden, 1, kVector);
    w->post_ffn_norm_1 = Get(p + "post_ffw_norm_1.weight", hidden, 1, kVector);
    w->post_ffn_norm_2 = Get(p + "post_ffw_norm_2.weight", hidden, 1, kVector);
    if (ok && reader.FindTensor(p + "ffn_gate_up_exps.weight") == nullptr) {
      Fail("tensor " + p +
           "ffn_gate_up_exps.weight is missing; separate gate and up "
           "experts are not supported");
      return;
    }
    w->gate_up_exps = Get(p + "ffn_gate_up_exps.weight", hidden, 2 * width,
                          kExpertGateUp, true, experts);
    w->down_exps = Get(p + "ffn_down_exps.weight", width, hidden, kExpertDown,
                       true, experts);
    w->down_exps_scale = Get(p + "ffn_down_exps.scale", experts, 1, kVector);
  }
};

/// Vocabulary size from the embedding tensor, which every other vocabulary
/// sized tensor must match.
std::uint64_t VocabRows(const core::GgufReader& reader) {
  const auto* info = reader.FindTensor("token_embd.weight");
  if (info == nullptr || info->dimensions.size() != 2) {
    return 0;
  }
  return info->dimensions[1];
}

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  const Format f = FormatOf(type);
  return f.block == 0 ? 0 : static_cast<std::size_t>(cols / f.block) * f.bytes;
}

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& reader,
                                               std::string* error_msg) {
  auto config = Config::FromGguf(reader, error_msg);
  if (!config) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  ModelWeights w;
  w.config = std::move(*config);
  const Config& c = w.config;
  const std::uint64_t vocab = VocabRows(reader);
  if (vocab == 0 || vocab > 0x7fffffffULL) {
    b.Fail("token_embd.weight must be a [hidden, vocab] matrix");
    return std::nullopt;
  }
  w.vocab_size = static_cast<std::uint32_t>(vocab);
  w.token_embd = b.Get("token_embd.weight", c.hidden_size, vocab, kEmbedding);
  w.output = b.Get("output.weight", c.hidden_size, vocab, kEmbedding, false);
  if (w.output.empty()) {
    w.output = w.token_embd;
  }
  w.output_norm = b.Get("output_norm.weight", c.hidden_size, 1, kVector);
  w.rope_factors = b.Factors(c);
  for (std::uint32_t l = 0; b.ok && l < c.num_layers; ++l) {
    w.layers.push_back(b.Layer(c, l, true));
    // A missing V projection means V is the unnormalized K projection; that
    // is only defined when K and V share a width.
    if (b.ok && w.layers.back().attn_v.empty() && c.IsSliding(l)) {
      b.Fail("sliding layer " + std::to_string(l) + " lacks attn_v");
    }
  }
  if (!b.ok) {
    return std::nullopt;
  }
  return w;
}

std::optional<DraftWeights> DraftWeights::Bind(const core::GgufReader& reader,
                                               const ModelWeights& target,
                                               std::string* error_msg) {
  auto config = Config::DraftFromGguf(reader, target.config, error_msg);
  if (!config) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  DraftWeights w;
  w.config = std::move(*config);
  const Config& c = w.config;
  if (VocabRows(reader) != target.vocab_size) {
    b.Fail("gemma4-assistant vocabulary differs from the target");
    return std::nullopt;
  }
  const std::uint64_t target_hidden = c.target_hidden_size;
  w.pre_projection = b.Get("nextn.pre_projection.weight", 2 * target_hidden,
                           c.hidden_size, kProjection);
  w.post_projection = b.Get("nextn.post_projection.weight", c.hidden_size,
                            target_hidden, kProjection);
  w.token_embd =
      b.Get("token_embd.weight", c.hidden_size, target.vocab_size, kEmbedding);
  w.output_norm = b.Get("output_norm.weight", c.hidden_size, 1, kVector);
  w.rope_factors = b.Factors(c);
  for (std::uint32_t l = 0; b.ok && l < c.num_layers; ++l) {
    w.layers.push_back(b.Layer(c, l, false));
  }
  if (!b.ok) {
    return std::nullopt;
  }
  return w;
}

}  // namespace gufo::models::gemma4
