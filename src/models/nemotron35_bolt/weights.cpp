#include "src/models/nemotron35_bolt/weights.hpp"

#include <cmath>
#include <cstring>
#include <initializer_list>
#include <string>
#include <utility>

namespace gufo::models::nemotron35_bolt {
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
    case GgmlType::kIQ2_XXS:
      return {256, 66};
    case GgmlType::kNVFP4:
      return {64, 36};
    default:
      return {0, 0};
  }
}

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

  /// Binds `name` with the exact shape and one of the accepted formats. A
  /// `.weight` tensor picks up the scalar F32 `.scale` companion when the
  /// file carries one; llama.cpp multiplies the matmul output by it.
  TensorRef Get(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                std::uint64_t experts, std::initializer_list<GgmlType> types) {
    TensorRef t;
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      Fail("missing tensor " + name);
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
    // Locate the shard so disk readers can address the payload directly, and
    // so a payload running past its shard is caught here rather than as a
    // fault later.
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
    constexpr std::string_view kWeight = ".weight";
    if (name.size() > kWeight.size() &&
        name.compare(name.size() - kWeight.size(), kWeight.size(), kWeight) ==
            0) {
      const std::string scale_name =
          name.substr(0, name.size() - kWeight.size()) + ".scale";
      if (const auto* s = reader.FindTensor(scale_name)) {
        if (s->type != GgmlType::kF32 || s->ElementCount() != 1) {
          Fail("tensor " + scale_name + " must be one F32 value");
          return TensorRef{};
        }
        std::memcpy(&t.scale, s->data, sizeof(float));
        if (!std::isfinite(t.scale) || t.scale == 0.0F) {
          Fail("tensor " + scale_name + " is not a finite nonzero scale");
          return TensorRef{};
        }
      }
    }
    return t;
  }

  LayerWeights Layer(const Config& c, std::uint32_t il) {
    LayerWeights l;
    l.type = c.blocks[il];
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::uint64_t hidden = c.hidden_size;
    const auto dense = {GgmlType::kNVFP4, GgmlType::kQ8_0, GgmlType::kBF16,
                        GgmlType::kF16, GgmlType::kF32};
    const auto experts = {GgmlType::kIQ2_XXS, GgmlType::kNVFP4,
                          GgmlType::kQ8_0};

    l.norm = Get(p + "attn_norm.weight", hidden, 1, 1, {GgmlType::kF32});
    switch (l.type) {
      case BlockType::kMamba:
        l.ssm_in =
            Get(p + "ssm_in.weight", hidden, c.SsmInProjection(), 1, dense);
        l.ssm_conv1d = Get(p + "ssm_conv1d.weight", c.ssm_conv_kernel,
                           c.SsmConvChannels(), 1, {GgmlType::kF32});
        l.ssm_conv1d_b = Get(p + "ssm_conv1d.bias", c.SsmConvChannels(), 1, 1,
                             {GgmlType::kF32});
        l.ssm_dt_b =
            Get(p + "ssm_dt.bias", c.ssm_heads, 1, 1, {GgmlType::kF32});
        l.ssm_a = Get(p + "ssm_a", 1, c.ssm_heads, 1, {GgmlType::kF32});
        l.ssm_d = Get(p + "ssm_d", 1, c.ssm_heads, 1, {GgmlType::kF32});
        l.ssm_norm = Get(p + "ssm_norm.weight", c.ssm_inner_size / c.ssm_groups,
                         c.ssm_groups, 1, {GgmlType::kF32});
        l.ssm_out =
            Get(p + "ssm_out.weight", c.ssm_inner_size, hidden, 1, dense);
        break;
      case BlockType::kAttention:
        l.attn_q =
            Get(p + "attn_q.weight", hidden, c.AttentionQDim(), 1, dense);
        l.attn_k =
            Get(p + "attn_k.weight", hidden, c.AttentionKvDim(), 1, dense);
        l.attn_v =
            Get(p + "attn_v.weight", hidden, c.AttentionKvDim(), 1, dense);
        l.attn_out =
            Get(p + "attn_output.weight", c.AttentionQDim(), hidden, 1, dense);
        break;
      case BlockType::kMoe: {
        // Routed experts per block are the router's row count.
        const auto* router = reader.FindTensor(p + "ffn_gate_inp.weight");
        const std::uint64_t n =
            router != nullptr && router->dimensions.size() == 2
                ? router->dimensions[1]
                : 0;
        if (n < c.experts_used[il] || n > c.max_experts) {
          Fail("invalid routed expert count in block " + std::to_string(il));
          return l;
        }
        l.num_experts = static_cast<std::uint32_t>(n);
        const std::uint64_t ff = c.expert_ff[il];
        l.router =
            Get(p + "ffn_gate_inp.weight", hidden, n, 1, {GgmlType::kF32});
        l.router_bias = Get(p + "exp_probs_b.bias", n, 1, 1, {GgmlType::kF32});
        l.latent_down =
            Get(p + "ffn_latent_down.weight", hidden, c.latent_size, 1, dense);
        l.latent_up =
            Get(p + "ffn_latent_up.weight", c.latent_size, hidden, 1, dense);
        l.up_exps =
            Get(p + "ffn_up_exps.weight", c.latent_size, ff, n, experts);
        l.down_exps =
            Get(p + "ffn_down_exps.weight", ff, c.latent_size, n, experts);
        l.up_shexp = Get(p + "ffn_up_shexp.weight", hidden, c.shared_expert_ff,
                         1, experts);
        l.down_shexp = Get(p + "ffn_down_shexp.weight", c.shared_expert_ff,
                           hidden, 1, experts);
        break;
      }
    }
    return l;
  }
};

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  const Format f = FormatOf(type);
  if (f.block == 0 || cols % f.block != 0) {
    return 0;
  }
  return static_cast<std::size_t>(cols / f.block) * f.bytes;
}

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& reader,
                                               std::string* error_msg) {
  auto config = Config::FromGguf(reader, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  ModelWeights w;
  w.config = std::move(*config);
  const Config& c = w.config;
  const auto dense = {GgmlType::kQ8_0, GgmlType::kBF16, GgmlType::kF16,
                      GgmlType::kF32};

  w.token_embd =
      b.Get("token_embd.weight", c.hidden_size, c.vocab_size, 1, dense);
  w.output_norm =
      b.Get("output_norm.weight", c.hidden_size, 1, 1, {GgmlType::kF32});
  w.output = b.Get("output.weight", c.hidden_size, c.vocab_size, 1, dense);
  w.layers.reserve(c.num_layers);
  for (std::uint32_t il = 0; il < c.num_layers && b.ok; ++il) {
    w.layers.push_back(b.Layer(c, il));
  }
  if (!b.ok) {
    return std::nullopt;
  }
  return w;
}

std::optional<DrafterWeights> DrafterWeights::Bind(
    const core::GgufReader& reader, const Config& target,
    std::string* error_msg) {
  auto config = DrafterConfig::FromGguf(reader, target, error_msg);
  if (!config.has_value())
    return std::nullopt;
  Binder b{reader, error_msg};
  DrafterWeights w;
  w.config = std::move(*config);
  const DrafterConfig& c = w.config;
  const std::uint64_t hidden = c.hidden_size;
  const std::uint64_t lc = c.lc_hidden;
  const std::uint64_t slots = c.Slots();
  const std::uint64_t nodes = slots * c.lc_top_k;
  const auto f32 = {GgmlType::kF32};
  const auto bf16 = {GgmlType::kBF16};
  const auto fp4 = {GgmlType::kNVFP4};

  w.fc = b.Get("fc.weight", c.FeatureSize(), hidden, 1, fp4);
  w.enc_norm = b.Get("enc.output_norm.weight", hidden, 1, 1, f32);
  w.output_norm = b.Get("output_norm.weight", hidden, 1, 1, f32);
  for (std::uint32_t il = 0; il < c.num_layers && b.ok; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    DrafterLayerWeights& l = w.layers.emplace_back();
    l.attn_norm = b.Get(p + "attn_norm.weight", hidden, 1, 1, f32);
    l.attn_conv_base =
        b.Get(p + "attn_conv_base", hidden, c.conv_kernel, 2, f32);
    l.attn_conv_proj =
        b.Get(p + "attn_conv_proj.weight", hidden, c.ConvProjection(), 1, bf16);
    l.attn_q = b.Get(p + "attn_q.weight", hidden, c.QDim(), 1, bf16);
    l.attn_k = b.Get(p + "attn_k.weight", hidden, c.KvDim(), 1, bf16);
    l.attn_v = b.Get(p + "attn_v.weight", hidden, c.KvDim(), 1, bf16);
    l.attn_q_norm = b.Get(p + "attn_q_norm.weight", c.head_dim, 1, 1, f32);
    l.attn_k_norm = b.Get(p + "attn_k_norm.weight", c.head_dim, 1, 1, f32);
    l.attn_out = b.Get(p + "attn_output.weight", c.QDim(), hidden, 1, bf16);
    l.attn_sinks = b.Get(p + "attn_sinks.weight", c.num_heads, 1, 1, f32);
    l.ffn_norm = b.Get(p + "ffn_norm.weight", hidden, 1, 1, f32);
    l.ffn_conv_base = b.Get(p + "ffn_conv_base", hidden, c.conv_kernel, 2, f32);
    l.ffn_conv_proj =
        b.Get(p + "ffn_conv_proj.weight", hidden, c.ConvProjection(), 1, bf16);
    l.ffn_gate = b.Get(p + "ffn_gate.weight", hidden, c.ff_size, 1, fp4);
    l.ffn_up = b.Get(p + "ffn_up.weight", hidden, c.ff_size, 1, fp4);
    l.ffn_down = b.Get(p + "ffn_down.weight", c.ff_size, hidden, 1, fp4);
  }

  w.token_proj = b.Get("lilicorr.token_proj.weight", hidden, lc, 1, fp4);
  w.token_proj_b = b.Get("lilicorr.token_proj.bias", lc, 1, 1, f32);
  w.pass_proj = b.Get("lilicorr.pass_hidden_proj.weight", hidden, lc, 1, fp4);
  w.pass_proj_b = b.Get("lilicorr.pass_hidden_proj.bias", lc, 1, 1, f32);
  w.context_proj = b.Get("lilicorr.context_proj.weight", hidden, lc, 1, fp4);
  w.context_proj_b = b.Get("lilicorr.context_proj.bias", lc, 1, 1, f32);
  // llama.cpp stores the candidate-feature MLP as the block after the
  // LiLiCorr layers, which follow the drafter's own blocks.
  const std::string f =
      "blk." + std::to_string(c.num_layers + c.lc_layers) + ".";
  w.feature_norm = b.Get(f + "ffn_norm.weight", 5, 1, 1, f32);
  w.feature_norm_b = b.Get(f + "ffn_norm.bias", 5, 1, 1, f32);
  w.feature_up = b.Get(f + "ffn_up.weight", 5, lc, 1, f32);
  w.feature_up_b = b.Get(f + "ffn_up.bias", lc, 1, 1, f32);
  w.feature_down = b.Get(f + "ffn_down.weight", lc, lc, 1, f32);
  w.feature_down_b = b.Get(f + "ffn_down.bias", lc, 1, 1, f32);
  w.slot_embd = b.Get("lilicorr.slot_embedding.weight", lc, slots, 1, f32);
  w.rank_embd = b.Get("lilicorr.rank_embedding.weight", lc, c.lc_top_k, 1, f32);
  w.attention_bias =
      b.Get("lilicorr.attention_bias.bias", nodes, nodes, c.lc_heads, f32);
  for (std::uint32_t i = 0; i < c.lc_layers && b.ok; ++i) {
    const std::string p = "blk." + std::to_string(c.num_layers + i) + ".";
    const std::string q = "lilicorr.layers." + std::to_string(i) + ".";
    LiliCorrLayerWeights& l = w.lc_layers.emplace_back();
    l.attn_norm = b.Get(p + "attn_norm.weight", lc, 1, 1, f32);
    l.mlp_norm = b.Get(p + "ffn_norm.weight", lc, 1, 1, f32);
    l.in_proj = b.Get(q + "attn.in_proj.weight", lc, 3 * lc, 1, f32);
    l.in_proj_b = b.Get(q + "attn.in_proj.bias", 3 * lc, 1, 1, f32);
    l.out_proj = b.Get(p + "attn_output.weight", lc, lc, 1, fp4);
    l.out_proj_b = b.Get(p + "attn_output.bias", lc, 1, 1, f32);
    l.mlp_up = b.Get(p + "ffn_up.weight", lc, c.lc_mlp, 1, fp4);
    l.mlp_up_b = b.Get(p + "ffn_up.bias", c.lc_mlp, 1, 1, f32);
    l.mlp_down = b.Get(p + "ffn_down.weight", c.lc_mlp, lc, 1, fp4);
    l.mlp_down_b = b.Get(p + "ffn_down.bias", lc, 1, 1, f32);
  }
  w.lc_output_norm = b.Get("lilicorr.output_norm.weight", lc, 1, 1, f32);
  w.anchor_norm = b.Get("lilicorr.anchor_norm.weight", lc, 1, 1, f32);
  w.factor_in = b.Get("lilicorr.factor_input_proj.weight", 3 * lc, lc, 1, f32);
  w.factor_in_b = b.Get("lilicorr.factor_input_proj.bias", lc, 1, 1, f32);
  w.out_head = b.Get("lilicorr.out_head.weight", lc, c.lc_factor, 1, f32);
  w.out_head_b = b.Get("lilicorr.out_head.bias", c.lc_factor, 1, 1, f32);
  w.in_head = b.Get("lilicorr.in_head.weight", lc, c.lc_factor, 1, f32);
  w.in_head_b = b.Get("lilicorr.in_head.bias", c.lc_factor, 1, 1, f32);
  w.anchor_head =
      b.Get("lilicorr.anchor_out_head.weight", lc, c.lc_factor, 1, f32);
  w.anchor_head_b =
      b.Get("lilicorr.anchor_out_head.bias", c.lc_factor, 1, 1, f32);
  if (!b.ok)
    return std::nullopt;
  return w;
}

}  // namespace gufo::models::nemotron35_bolt
