#include "src/models/gemma4/config.hpp"

#include <cmath>
#include <string_view>
#include <utility>
#include <variant>

namespace gufo::models::gemma4 {
namespace {

constexpr std::string_view kTargetArchitecture = "gemma4";
constexpr std::string_view kDraftArchitecture = "gemma4-assistant";

struct Reader {
  const core::GgufReader& gguf;
  std::string prefix;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  [[nodiscard]] std::string Key(std::string_view suffix) const {
    return prefix + std::string(suffix);
  }

  std::uint32_t U32(std::string_view suffix, bool required = true,
                    std::uint32_t fallback = 0) {
    const std::string key = Key(suffix);
    const auto value = gguf.GetMetadataUint32(key);
    if (value.has_value()) {
      return *value;
    }
    if (required || gguf.FindMetadata(key) != nullptr) {
      Fail("missing or invalid GGUF integer " + key);
    }
    return fallback;
  }

  float F32(std::string_view suffix, bool required = true,
            float fallback = 0.0F) {
    const std::string key = Key(suffix);
    const auto value = gguf.GetMetadataFloat32(key);
    if (value.has_value() && std::isfinite(*value)) {
      return *value;
    }
    if (required || gguf.FindMetadata(key) != nullptr) {
      Fail("missing or invalid GGUF float " + key);
    }
    return fallback;
  }

  /// Integer or boolean arrays of any width; GGUF writers pick the type.
  std::vector<std::uint64_t> U64Array(std::string_view suffix) {
    const std::string key = Key(suffix);
    const auto* meta = gguf.FindMetadata(key);
    std::vector<std::uint64_t> out;
    if (meta == nullptr) {
      Fail("missing GGUF array " + key);
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out = *u;
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (auto v : *s) {
        if (v < 0) {
          Fail("negative GGUF array value " + key);
          return {};
        }
        out.push_back(static_cast<std::uint64_t>(v));
      }
    } else {
      Fail("GGUF array must contain integers: " + key);
    }
    return out;
  }

  /// Per-layer values stored either as one scalar or as one entry per layer.
  std::vector<std::uint32_t> PerLayer(std::string_view suffix,
                                      std::uint32_t layers) {
    const std::string key = Key(suffix);
    const auto* meta = gguf.FindMetadata(key);
    if (meta == nullptr) {
      Fail("missing GGUF value " + key);
      return {};
    }
    if (const auto scalar = gguf.GetMetadataUint32(key)) {
      return std::vector<std::uint32_t>(layers, *scalar);
    }
    const auto values = U64Array(suffix);
    if (!ok) {
      return {};
    }
    if (values.size() != layers) {
      Fail(key + " must describe every layer");
      return {};
    }
    std::vector<std::uint32_t> out;
    for (auto v : values) {
      if (v == 0 || v > 0xffffffffULL) {
        Fail(key + " has an invalid entry");
        return {};
      }
      out.push_back(static_cast<std::uint32_t>(v));
    }
    return out;
  }
};

/// Fields shared by the target and the drafter; `r.prefix` selects the
/// architecture namespace.
Config ReadCommon(Reader& r) {
  Config c;
  c.num_layers = r.U32("block_count");
  c.hidden_size = r.U32("embedding_length");
  c.context_length = r.U32("context_length");
  c.rms_eps = r.F32("attention.layer_norm_rms_epsilon");
  c.num_heads = r.U32("attention.head_count");
  if (!r.ok) {
    return c;
  }
  if (c.num_layers == 0 || c.hidden_size == 0 || c.num_heads == 0 ||
      c.context_length == 0 || !(c.rms_eps > 0.0F)) {
    r.Fail("gemma4 dimensions must be positive");
    return c;
  }

  const auto ffn = r.PerLayer("feed_forward_length", c.num_layers);
  if (!r.ok) {
    return c;
  }
  c.ffn_size = ffn[0];
  for (auto v : ffn) {
    if (v != c.ffn_size) {
      r.Fail("per-layer feed-forward widths are not supported");
      return c;
    }
  }
  c.kv_heads = r.PerLayer("attention.head_count_kv", c.num_layers);

  const auto pattern = r.U64Array("attention.sliding_window_pattern");
  if (r.ok && pattern.size() != c.num_layers) {
    r.Fail("attention.sliding_window_pattern must describe every layer");
  }
  if (!r.ok) {
    return c;
  }
  for (auto v : pattern) {
    c.sliding.push_back(v != 0 ? 1 : 0);
  }

  c.head_dim_global = r.U32("attention.key_length");
  c.head_dim_sliding = r.U32("attention.key_length_swa");
  if (r.U32("attention.value_length") != c.head_dim_global ||
      r.U32("attention.value_length_swa") != c.head_dim_sliding) {
    r.Fail("gemma4 requires equal key and value head dimensions");
  }
  c.rope_dim_global = r.U32("rope.dimension_count");
  c.rope_dim_sliding = r.U32("rope.dimension_count_swa");
  c.rope_theta_global = r.F32("rope.freq_base");
  c.rope_theta_sliding = r.F32("rope.freq_base_swa");
  c.sliding_window = r.U32("attention.sliding_window");
  if (!r.ok) {
    return c;
  }

  for (const std::uint32_t dim : {c.head_dim_global, c.head_dim_sliding}) {
    // Kernels reduce heads in 64-wide lane groups and GGUF K-quants need
    // 256-aligned reductions for the output projection.
    if (dim == 0 || dim % 64 != 0) {
      r.Fail("gemma4 head dimensions must be positive multiples of 64");
      return c;
    }
  }
  if (c.rope_dim_global != c.head_dim_global ||
      c.rope_dim_sliding != c.head_dim_sliding) {
    r.Fail("gemma4 rope must span the full head dimension");
    return c;
  }
  if (!(c.rope_theta_global > 0.0F) || !(c.rope_theta_sliding > 0.0F)) {
    r.Fail("gemma4 rope bases must be positive");
    return c;
  }
  if (c.sliding_window == 0) {
    r.Fail("gemma4 sliding window must be positive");
    return c;
  }
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    if (c.num_heads % c.kv_heads[l] != 0) {
      r.Fail("gemma4 query heads must be a multiple of KV heads in layer " +
             std::to_string(l));
      return c;
    }
  }
  if (r.U32("embedding_length_per_layer_input", false, 0) != 0) {
    r.Fail("gemma4 per-layer embeddings are not supported");
  }
  return c;
}

}  // namespace

std::uint32_t Config::GlobalLayerCount() const noexcept {
  std::uint32_t count = 0;
  for (auto s : sliding) {
    count += s == 0 ? 1U : 0U;
  }
  return count;
}

std::uint32_t Config::SharedKvSource(std::uint32_t draft_layer,
                                     const Config& target) const noexcept {
  const bool want_sliding = IsSliding(draft_layer);
  for (std::uint32_t l = target.num_layers; l-- > 0;) {
    if (target.IsSliding(l) == want_sliding) {
      return l;
    }
  }
  return target.num_layers;
}

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       std::string* error_msg) {
  Reader r{gguf, std::string(kTargetArchitecture) + ".", error_msg};
  if (gguf.GetMetadataString("general.architecture") != kTargetArchitecture) {
    r.Fail("GGUF architecture is not gemma4");
    return std::nullopt;
  }
  Config c = ReadCommon(r);
  if (!r.ok) {
    return std::nullopt;
  }
  c.final_logit_softcap = r.F32("final_logit_softcapping", false, 0.0F);
  if (r.U32("attention.shared_kv_layers", false, 0) != 0) {
    r.Fail("gemma4 targets with shared KV layers are not supported");
  }
  c.num_experts = r.U32("expert_count", false, 0);
  if (r.ok && c.num_experts != 0) {
    c.experts_used = r.U32("expert_used_count");
    const auto widths = r.PerLayer("expert_feed_forward_length", c.num_layers);
    if (r.ok) {
      c.expert_ffn_size = widths[0];
      for (auto v : widths) {
        if (v != c.expert_ffn_size) {
          r.Fail("per-layer expert widths are not supported");
          break;
        }
      }
    }
    if (r.ok &&
        (c.experts_used == 0 || c.experts_used > c.num_experts ||
         c.experts_used > kMaxExpertsUsed || c.num_experts > kMaxExperts)) {
      r.Fail("gemma4 expert routing must use 1.." +
             std::to_string(kMaxExpertsUsed) + " of at most " +
             std::to_string(kMaxExperts) + " experts");
    }
  } else if (r.ok && gguf.FindTensor("blk.0.ffn_gate_inp.weight") != nullptr) {
    r.Fail("gemma4 expert tensors without gemma4.expert_count");
  }
  if (r.ok && c.final_logit_softcap < 0.0F) {
    r.Fail("gemma4 final logit softcap must not be negative");
  }
  if (r.ok &&
      (c.GlobalLayerCount() == 0 || c.GlobalLayerCount() == c.num_layers)) {
    r.Fail("gemma4 targets need both sliding and global layers");
  }
  if (!r.ok) {
    return std::nullopt;
  }
  return c;
}

std::optional<Config> Config::DraftFromGguf(const core::GgufReader& gguf,
                                            const Config& target,
                                            std::string* error_msg) {
  Reader r{gguf, std::string(kDraftArchitecture) + ".", error_msg};
  if (gguf.GetMetadataString("general.architecture") != kDraftArchitecture) {
    r.Fail("GGUF architecture is not gemma4-assistant");
    return std::nullopt;
  }
  Config c = ReadCommon(r);
  if (!r.ok) {
    return std::nullopt;
  }
  c.target_hidden_size = r.U32("embedding_length_out");
  c.shared_kv_layers = r.U32("attention.shared_kv_layers");
  const std::uint32_t nextn =
      r.U32("nextn_predict_layers", false, c.num_layers);
  if (!r.ok) {
    return std::nullopt;
  }
  if (c.target_hidden_size != target.hidden_size) {
    r.Fail("gemma4-assistant embedding_length_out must match the target");
  } else if (c.shared_kv_layers != c.num_layers || nextn != c.num_layers) {
    r.Fail("gemma4-assistant drafts must share KV in every layer");
  } else if (c.head_dim_global != target.head_dim_global ||
             c.head_dim_sliding != target.head_dim_sliding ||
             c.rope_theta_global != target.rope_theta_global ||
             c.rope_theta_sliding != target.rope_theta_sliding ||
             c.sliding_window != target.sliding_window ||
             c.rms_eps != target.rms_eps) {
    r.Fail("gemma4-assistant attention constants differ from the target");
  } else {
    for (std::uint32_t l = 0; l < c.num_layers; ++l) {
      const std::uint32_t source = c.SharedKvSource(l, target);
      if (source >= target.num_layers ||
          c.kv_heads[l] != target.kv_heads[source]) {
        r.Fail("gemma4-assistant layer " + std::to_string(l) +
               " cannot attend its target KV source");
        break;
      }
    }
  }
  if (!r.ok) {
    return std::nullopt;
  }
  return c;
}

}  // namespace gufo::models::gemma4
