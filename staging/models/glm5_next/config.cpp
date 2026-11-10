#include "staging/models/glm5_next/config.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>

namespace gufo::models::glm5_next {
namespace {

constexpr std::string_view kArchitecture = "glm5-next";

// The single artifact size whose kernels and cache layout were designed and
// measured. Membership means the geometry is supported; divisibility alone is
// not enough to run.
struct GeometryProfile {
  std::string_view name;
  std::uint32_t num_layers;    // trunk layers, nextn excluded
  std::uint32_t nextn_layers;  // self-speculative draft blocks
  std::uint32_t hidden_size;
  std::uint32_t vocab_size;
  std::uint32_t num_heads;
  std::uint32_t q_lora_rank;
  std::uint32_t kv_lora_rank;
  std::uint32_t mla_head_dim;
  std::uint32_t mla_value_dim;
  std::uint32_t kda_heads;
  std::uint32_t kda_head_dim;
  std::uint32_t ssm_conv_kernel;
  std::uint32_t kda_stride;  // DSA at (il + 1) % kda_stride == 0 in the trunk
  std::uint32_t indexer_heads;
  std::uint32_t indexer_key_len;
  std::uint32_t indexer_top_k;
  std::uint32_t indexer_kpool;
  std::uint32_t hc_count;
  std::uint32_t num_experts;
  std::uint32_t num_experts_used;
  std::uint32_t expert_ff;
  std::uint32_t shared_expert_ff;
  std::uint32_t leading_dense_blocks;
  std::uint32_t dense_ff;
};

constexpr GeometryProfile kProfiles[] = {
    {
        .name = "GLM-5.3-Flash",
        .num_layers = 45,
        .nextn_layers = 1,
        .hidden_size = 4096,
        .vocab_size = 154880,
        .num_heads = 64,
        .q_lora_rank = 1536,
        .kv_lora_rank = 512,
        .mla_head_dim = 256,
        .mla_value_dim = 256,
        .kda_heads = 64,
        .kda_head_dim = 128,
        .ssm_conv_kernel = 4,
        .kda_stride = 4,
        .indexer_heads = 32,
        .indexer_key_len = 128,
        .indexer_top_k = 2048,
        .indexer_kpool = 4,
        .hc_count = 4,
        .num_experts = 288,
        .num_experts_used = 8,
        .expert_ff = 2048,
        .shared_expert_ff = 2048,
        .leading_dense_blocks = 3,
        .dense_ff = 12288,
    },
};

struct Reader {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  std::uint32_t U32(std::string_view key, bool required = true,
                    std::uint32_t fallback = 0) {
    const auto value = gguf.GetMetadataUint32(key);
    if (value.has_value()) {
      return *value;
    }
    if (required || gguf.FindMetadata(key) != nullptr) {
      Fail("missing or invalid GGUF integer " + std::string(key));
    }
    return fallback;
  }

  float F32(std::string_view key, float fallback, bool required = true) {
    const auto value = gguf.GetMetadataFloat32(key);
    if (!value || !std::isfinite(*value)) {
      if (required || gguf.FindMetadata(key) != nullptr) {
        Fail("missing or invalid GGUF float " + std::string(key));
      }
      return fallback;
    }
    return *value;
  }

  bool Boolean(std::string_view key, bool required, bool fallback) {
    const auto value = gguf.GetMetadataBool(key);
    if (value.has_value()) {
      return *value;
    }
    if (required || gguf.FindMetadata(key) != nullptr) {
      Fail("missing or invalid GGUF bool " + std::string(key));
    }
    return fallback;
  }

  /// Float arrays of any element width; GGUF writers pick the narrowest type.
  std::vector<float> F32Array(std::string_view key, bool required = true) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<float> out;
    if (meta == nullptr) {
      if (required) {
        Fail("missing GGUF array " + std::string(key));
      }
      return out;
    }
    if (const auto* d = std::get_if<std::vector<double>>(&meta->value)) {
      for (const double v : *d) {
        if (!std::isfinite(v)) {
          Fail("non-finite GGUF array value " + std::string(key));
          return {};
        }
        out.push_back(static_cast<float>(v));
      }
    } else {
      Fail("GGUF array must contain floats: " + std::string(key));
    }
    return out;
  }

  std::vector<std::uint64_t> U64Array(std::string_view key,
                                      bool required = true) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<std::uint64_t> out;
    if (meta == nullptr) {
      if (required) {
        Fail("missing GGUF array " + std::string(key));
      }
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out = *u;
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (const auto v : *s) {
        if (v < 0) {
          Fail("negative GGUF array value " + std::string(key));
          return {};
        }
        out.push_back(static_cast<std::uint64_t>(v));
      }
    } else {
      Fail("GGUF array must contain integers: " + std::string(key));
    }
    return out;
  }
};

/// Expands a scalar-or-array clamp key to one entry per block, matching the
/// reference loader (a scalar fills every block, an array must cover all).
void ExpandClamps(Reader& r, std::string_view key, std::uint32_t num_layers_all,
                  std::vector<float>* out) {
  const auto* meta = r.gguf.FindMetadata(key);
  if (meta == nullptr) {
    r.Fail("missing GGUF key " + std::string(key));
    return;
  }
  std::vector<float> values;
  if (meta->type == core::GgufValueType::kFloat32) {
    const auto v = r.gguf.GetMetadataFloat32(key);
    if (!v || !std::isfinite(*v)) {
      r.Fail("invalid float " + std::string(key));
      return;
    }
    values.assign(num_layers_all, *v);
  } else {
    values = r.F32Array(key);
  }
  if (!r.ok) {
    return;
  }
  if (values.size() == 1) {
    values.assign(num_layers_all, values[0]);
  }
  if (values.size() != num_layers_all) {
    r.Fail("clamp array size mismatch: " + std::string(key));
    return;
  }
  *out = std::move(values);
}

}  // namespace

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       bool require_trunk,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture") != kArchitecture) {
    r.Fail("GGUF architecture is not glm5-next");
    return std::nullopt;
  }

  Config c;
  const std::string_view p = "glm5-next.";
  auto key = [&](std::string_view suffix) {
    return std::string(p) + std::string(suffix);
  };

  c.num_layers_all = r.U32(key("block_count"));
  c.nextn_layers = r.U32(key("nextn_predict_layers"), false, 0);
  if (c.nextn_layers >= c.num_layers_all) {
    r.Fail("nextn_predict_layers must be below block_count");
    return std::nullopt;
  }
  c.num_layers = c.num_layers_all - c.nextn_layers;
  c.hidden_size = r.U32(key("embedding_length"));
  c.vocab_size = r.U32(key("vocab_size"));
  c.context_length = r.U32(key("context_length"));
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"), 1e-5F);
  c.ln_eps = r.F32(key("attention.layer_norm_epsilon"), 1e-6F, false);

  c.num_heads = r.U32(key("attention.head_count"));
  c.q_lora_rank = r.U32(key("attention.q_lora_rank"));
  c.kv_lora_rank = r.U32(key("attention.kv_lora_rank"));
  if (r.U32(key("rope.dimension_count"), false, 0) != 0) {
    r.Fail("glm5-next is NoPE (rope.dimension_count must be 0)");
  }
  {
    const auto kv = r.U64Array(key("attention.head_count_kv"));
    if (kv.size() != c.num_layers_all) {
      r.Fail("attention.head_count_kv must cover every block");
    } else {
      for (const auto v : kv) {
        if (v != 0 && v != 1) {
          r.Fail("glm5-next MLA layers have exactly one latent KV head");
          break;
        }
        c.head_count_kv.push_back(static_cast<std::uint32_t>(v));
      }
    }
  }

  c.ssm_conv_kernel = r.U32(key("ssm.conv_kernel"));
  c.kda_head_dim = r.U32(key("kda.head_dim"));
  c.kda_gate_lower_bound = r.F32(key("kda.gate_lower_bound"), 0.0F);

  c.indexer_heads = r.U32(key("attention.indexer.head_count"));
  c.indexer_key_len = r.U32(key("attention.indexer.key_length"));
  c.indexer_top_k = r.U32(key("attention.indexer.top_k"));
  c.indexer_kpool = r.U32(key("attention.indexer.kpool"));
  c.index_share_mtp =
      r.Boolean(key("attention.indexer.index_share_mtp"), false, false);
  // The reference keeps its hparams default true when the key is absent
  // (llama reads it with required=false), so pool tails are selected unless
  // the artifact explicitly disables it.
  c.kpool_select_tail =
      r.Boolean(key("attention.indexer.kpool_select_tail"), false, true);
  if (c.indexer_kpool <= 1 || c.indexer_top_k % c.indexer_kpool != 0) {
    r.Fail("indexer kpool must exceed 1 and divide top_k");
  }
  if (gguf.FindMetadata(key("attention.indexer.types")) != nullptr) {
    // The measured artifact omits the array (all trunk DSA layers run a full
    // indexer). Mixed sharing is not yet supported by the kernels.
    r.Fail("attention.indexer.types arrays are not supported");
  }

  c.hc_count = r.U32(key("hyper_connection.count"));
  c.hc_sinkhorn_iters = r.U32(key("hyper_connection.sinkhorn_iterations"));
  c.hc_eps = r.F32(key("hyper_connection.epsilon"), 1e-6F);
  if (c.hc_count != 4) {
    r.Fail("glm5-next mHC requires hyper_connection.count == 4");
  }

  c.num_experts = r.U32(key("expert_count"));
  c.num_experts_used = r.U32(key("expert_used_count"));
  c.expert_ff = r.U32(key("expert_feed_forward_length"));
  c.shared_expert_ff =
      r.U32(key("expert_shared_feed_forward_length"), false, c.expert_ff);
  if (r.U32(key("expert_shared_count"), false, 1) != 1) {
    r.Fail("glm5-next expects exactly one shared expert");
  }
  if (r.U32(key("expert_group_count"), false, 1) != 1 ||
      r.U32(key("expert_group_used_count"), false, 1) != 1) {
    r.Fail("glm5-next expert routing must be ungrouped");
  }
  c.expert_weights_scale = r.F32(key("expert_weights_scale"), 1.0F, false);
  c.expert_weights_norm = r.Boolean(key("expert_weights_norm"), false, false);
  c.expert_gating_func = r.U32(key("expert_gating_func"));

  c.leading_dense_blocks = r.U32(key("leading_dense_block_count"), false, 0);
  c.dense_ff = r.U32(key("feed_forward_length"));
  ExpandClamps(r, key("swiglu_clamp_exp"), c.num_layers_all,
               &c.swiglu_clamp_exp);
  ExpandClamps(r, key("swiglu_clamp_shexp"), c.num_layers_all,
               &c.swiglu_clamp_shexp);

  if (!r.ok) {
    return std::nullopt;
  }

  // Structural limits of this runtime, not of the format.
  if (c.context_length == 0 || c.vocab_size == 0 || !std::isfinite(c.rms_eps) ||
      c.rms_eps <= 0 || !std::isfinite(c.ln_eps) || c.ln_eps <= 0 ||
      c.hidden_size == 0 || c.num_heads == 0 || c.q_lora_rank == 0 ||
      c.kv_lora_rank == 0 || c.ssm_conv_kernel == 0 || c.kda_head_dim == 0 ||
      !std::isfinite(c.kda_gate_lower_bound) ||
      c.kda_gate_lower_bound >= 0.0F || c.num_experts == 0 ||
      c.num_experts_used == 0 || c.num_experts_used > c.num_experts ||
      c.expert_ff == 0 || c.shared_expert_ff == 0 ||
      !std::isfinite(c.expert_weights_scale) || c.expert_gating_func != 2 ||
      !std::isfinite(c.hc_eps) || c.hc_eps <= 0 || c.hc_sinkhorn_iters == 0 ||
      c.dense_ff == 0 || c.leading_dense_blocks >= c.num_layers) {
    r.Fail("glm5-next metadata describes an unsupported shape");
    return std::nullopt;
  }

  const GeometryProfile* profile = nullptr;
  for (const GeometryProfile& p : kProfiles) {
    if (c.num_layers == p.num_layers && c.nextn_layers == p.nextn_layers &&
        c.hidden_size == p.hidden_size && c.vocab_size == p.vocab_size &&
        c.num_heads == p.num_heads && c.q_lora_rank == p.q_lora_rank &&
        c.kv_lora_rank == p.kv_lora_rank &&
        c.ssm_conv_kernel == p.ssm_conv_kernel &&
        c.kda_head_dim == p.kda_head_dim &&
        c.indexer_heads == p.indexer_heads &&
        c.indexer_key_len == p.indexer_key_len &&
        c.indexer_top_k == p.indexer_top_k &&
        c.indexer_kpool == p.indexer_kpool && c.hc_count == p.hc_count &&
        c.num_experts == p.num_experts &&
        c.num_experts_used == p.num_experts_used &&
        c.expert_ff == p.expert_ff &&
        c.shared_expert_ff == p.shared_expert_ff &&
        c.leading_dense_blocks == p.leading_dense_blocks &&
        c.dense_ff == p.dense_ff) {
      profile = &p;
      break;
    }
  }
  if (profile == nullptr) {
    r.Fail("unsupported GLM-5.3 kernel geometry");
    return std::nullopt;
  }
  c.profile_name = profile->name;
  c.kda_heads = profile->kda_heads;
  c.mla_head_dim = profile->mla_head_dim;
  c.mla_value_dim = profile->mla_value_dim;
  c.kda_stride = profile->kda_stride;
  if (require_trunk) {
    // DSA blocks sit at (il + 1) % kda_stride == 0 in the trunk and on the
    // nextn block; every other trunk block is KDA.
    for (std::uint32_t il = 0; il < c.num_layers_all; ++il) {
      const bool expected =
          ((il + 1) % c.kda_stride == 0) || il >= c.num_layers;
      if (c.IsDsaLayer(il) != expected) {
        r.Fail(
            "glm5-next DSA layer placement differs from the supported "
            "pattern");
        return std::nullopt;
      }
    }
  }
  return c;
}

}  // namespace gufo::models::glm5_next
