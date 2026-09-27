#include "src/models/qwen38_flash_next/weights.hpp"

#include <initializer_list>
#include <string>
#include <utility>

namespace gufo::models::qwen38_flash_next {
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
    case GgmlType::kQ5_1:
      return {32, 24};
    case GgmlType::kIQ4_NL:
      return {32, 18};
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

  /// Binds `name` with the exact shape and one of the accepted formats.
  TensorRef Get(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                std::uint64_t experts, std::initializer_list<GgmlType> types,
                bool required = true) {
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
    return t;
  }

  HcMixer Mixer(const std::string& prefix, const Config& c, bool with_inject) {
    HcMixer m;
    const std::uint64_t hc_dim = c.HcDim();
    m.norm = Get(prefix + "_norm.weight", hc_dim, 1, 1, {GgmlType::kF32});
    m.down =
        Get(prefix + "_down.weight", hc_dim, c.hc_low_rank, 1,
            {GgmlType::kQ8_0, GgmlType::kBF16, GgmlType::kF16, GgmlType::kF32});
    m.up =
        Get(prefix + "_up.weight", c.hc_low_rank, hc_dim, 1,
            {GgmlType::kQ8_0, GgmlType::kBF16, GgmlType::kF16, GgmlType::kF32});
    if (with_inject) {
      m.inject = Get(prefix + "_inject.weight", hc_dim, c.hc_count, 1,
                     {GgmlType::kF32, GgmlType::kQ8_0, GgmlType::kBF16});
    }
    return m;
  }

  LayerWeights Layer(const Config& c, std::uint32_t il, bool linear, bool ple,
                     bool nextn) {
    LayerWeights l;
    l.linear = linear;
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::uint64_t hidden = c.hidden_size;
    const std::uint64_t hc_dim = c.HcDim();
    const auto dense = {GgmlType::kQ8_0, GgmlType::kBF16, GgmlType::kF16,
                        GgmlType::kF32};
    const auto experts = {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K,
                          GgmlType::kQ5_1, GgmlType::kQ8_0};

    l.hc_attn = Mixer(p + "hc_attn", c, true);
    l.hc_ffn = Mixer(p + "hc_ffn", c, true);

    if (linear) {
      l.ssm_qkv =
          Get(p + "attn_qkv.weight", hidden, c.SsmConvChannels(), 1, dense);
      l.ssm_gate =
          Get(p + "attn_gate.weight", hidden, c.SsmValueDim(), 1, dense);
      l.ssm_conv1d = Get(p + "ssm_conv1d.weight", c.ssm_conv_kernel,
                         c.SsmConvChannels(), 1, {GgmlType::kF32});
      l.ssm_alpha = Get(p + "ssm_alpha.weight", hidden, c.ssm_num_v_heads, 1,
                        {GgmlType::kF32, GgmlType::kBF16, GgmlType::kQ8_0});
      l.ssm_beta = Get(p + "ssm_beta.weight", hidden, c.ssm_num_v_heads, 1,
                       {GgmlType::kF32, GgmlType::kBF16, GgmlType::kQ8_0});
      l.ssm_dt =
          Get(p + "ssm_dt.bias", c.ssm_num_v_heads, 1, 1, {GgmlType::kF32});
      l.ssm_a = Get(p + "ssm_a", c.ssm_num_v_heads, 1, 1, {GgmlType::kF32});
      l.ssm_norm =
          Get(p + "ssm_norm.weight", c.ssm_head_dim, 1, 1, {GgmlType::kF32});
      l.ssm_out = Get(p + "ssm_out.weight", c.SsmValueDim(), hidden, 1, dense);
    } else {
      l.attn_q =
          Get(p + "attn_q.weight", hidden, 2 * c.AttentionQDim(), 1, dense);
      l.attn_k = Get(p + "attn_k.weight", hidden, c.AttentionKvDim(), 1, dense);
      l.attn_v = Get(p + "attn_v.weight", hidden, c.AttentionKvDim(), 1, dense);
      l.attn_out =
          Get(p + "attn_output.weight", c.AttentionQDim(), hidden, 1, dense);
      l.attn_q_norm =
          Get(p + "attn_q_norm.weight", c.head_dim, 1, 1, {GgmlType::kF32});
      l.attn_k_norm =
          Get(p + "attn_k_norm.weight", c.head_dim, 1, 1, {GgmlType::kF32});
      l.indexer_q = Get(p + "indexer.q_proj.weight", hidden,
                        c.indexer_heads * c.indexer_head_dim, 1, dense);
      l.indexer_k = Get(p + "indexer.k_proj.weight", hidden, c.indexer_head_dim,
                        1, dense);
      l.indexer_q_norm = Get(p + "indexer.q_norm.weight", c.indexer_head_dim, 1,
                             1, {GgmlType::kF32});
      l.indexer_k_norm = Get(p + "indexer.k_norm.weight", c.indexer_head_dim, 1,
                             1, {GgmlType::kF32});
    }

    if (ple) {
      const std::uint64_t ple_dim = c.PleEmbeddingDim();
      l.ple_key = Get(p + "ple_key.weight", ple_dim, hc_dim, 1, dense);
      l.ple_value = Get(p + "ple_value.weight", ple_dim, hidden, 1, dense);
      l.ple_norm_key =
          Get(p + "ple_norm_key.weight", hc_dim, 1, 1, {GgmlType::kF32});
      l.ple_norm_query =
          Get(p + "ple_norm_query.weight", hc_dim, 1, 1, {GgmlType::kF32});
      l.ple_norm_conv =
          Get(p + "ple_norm_conv.weight", hc_dim, 1, 1, {GgmlType::kF32});
      l.ple_conv1d = Get(p + "ple_conv1d.weight", c.ple_conv_kernel, hc_dim, 1,
                         {GgmlType::kF32});
    }

    l.router = Get(p + "ffn_gate_inp.weight", hidden, c.num_experts, 1,
                   {GgmlType::kF32});
    l.ffn_gate_exps = Get(p + "ffn_gate_exps.weight", hidden, c.expert_ff,
                          c.num_experts, experts);
    l.ffn_up_exps = Get(p + "ffn_up_exps.weight", hidden, c.expert_ff,
                        c.num_experts, experts);
    l.ffn_down_exps = Get(p + "ffn_down_exps.weight", c.expert_ff, hidden,
                          c.num_experts, experts);
    l.shexp_gate_inp =
        Get(p + "ffn_gate_inp_shexp.weight", hidden, 1, 1, {GgmlType::kF32});
    l.shexp_gate =
        Get(p + "ffn_gate_shexp.weight", hidden, c.shared_expert_ff, 1, dense);
    l.shexp_up =
        Get(p + "ffn_up_shexp.weight", hidden, c.shared_expert_ff, 1, dense);
    l.shexp_down =
        Get(p + "ffn_down_shexp.weight", c.shared_expert_ff, hidden, 1, dense);

    if (nextn) {
      l.nextn_enorm =
          Get(p + "nextn.enorm.weight", hidden, 1, 1, {GgmlType::kF32});
      l.nextn_hnorm =
          Get(p + "nextn.hnorm.weight", hc_dim, 1, 1, {GgmlType::kF32});
      l.nextn_eh_proj =
          Get(p + "nextn.eh_proj.weight", 2 * hidden, hidden, 1, dense);
      l.nextn_head = Mixer(p + "nextn.hc_head", c, false);
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
  auto config = Config::FromGguf(reader, true, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  ModelWeights w;
  w.config = *config;
  const Config& c = w.config;
  const auto dense = {GgmlType::kQ8_0, GgmlType::kQ6_K, GgmlType::kBF16,
                      GgmlType::kF16, GgmlType::kF32};

  {
    // The vocabulary is the embedding row count; no metadata key carries it.
    const auto* info = reader.FindTensor("token_embd.weight");
    if (info == nullptr || info->dimensions.size() != 2) {
      b.Fail("missing token_embd.weight");
      return std::nullopt;
    }
    w.config.vocab_size = static_cast<std::uint32_t>(info->dimensions[1]);
  }
  w.token_embd =
      b.Get("token_embd.weight", c.hidden_size, c.vocab_size, 1, dense);
  w.output =
      b.Get("output.weight", c.hidden_size, c.vocab_size, 1, dense, false);
  if (w.output.empty()) {
    w.output = w.token_embd;
  }
  w.hc_head = b.Mixer("output_hc", c, false);
  if (c.ple_layer >= 0) {
    // The converter may pad the table beyond the hashed range, so take the
    // row count from the file and only check that it covers every head.
    const auto* info = reader.FindTensor("per_layer_token_embd.weight");
    const std::uint64_t rows = info != nullptr && info->dimensions.size() == 2
                                   ? info->dimensions[1]
                                   : 0;
    if (rows < c.ple_rows) {
      b.Fail("per_layer_token_embd.weight is missing or too short");
    } else {
      w.ple_table =
          b.Get("per_layer_token_embd.weight", c.ple_head_dim, rows, 1,
                {GgmlType::kIQ4_NL, GgmlType::kBF16, GgmlType::kQ8_0});
    }
  }
  w.layers.reserve(c.num_layers);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    w.layers.push_back(
        b.Layer(c, il, c.IsLinearLayer(il), c.IsPleLayer(il), false));
  }
  if (!b.ok) {
    return std::nullopt;
  }
  return w;
}

std::optional<MtpWeights> MtpWeights::Bind(const core::GgufReader& reader,
                                           const Config& trunk,
                                           std::string* error_msg) {
  auto config = Config::FromGguf(reader, false, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  if (!config->MtpMatches(trunk)) {
    b.Fail(
        "MTP sidecar architecture or inference constants differ from the "
        "trunk");
    return std::nullopt;
  }
  MtpWeights m;
  m.config = *config;
  m.config.vocab_size = trunk.vocab_size;
  // The draft block is a full-attention layer at index num_layers.
  m.block = b.Layer(m.config, m.config.num_layers, false, false, true);
  if (!b.ok) {
    return std::nullopt;
  }
  return m;
}

}  // namespace gufo::models::qwen38_flash_next
