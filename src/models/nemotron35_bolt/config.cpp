#include "src/models/nemotron35_bolt/config.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <utility>
#include <variant>

namespace gufo::models::nemotron35_bolt {
namespace {

constexpr std::string_view kArchitecture = "nemotron_h_moe";
constexpr std::uint32_t kSigmoidGating = 2;  // llama.cpp gating function id

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
    if ((value && !std::isfinite(*value)) ||
        (!value && (required || gguf.FindMetadata(key) != nullptr))) {
      Fail("missing or invalid GGUF float " + std::string(key));
    }
    return value.value_or(fallback);
  }

  bool Bool(std::string_view key, bool fallback) {
    const auto value = gguf.GetMetadataBool(key);
    if (!value && gguf.FindMetadata(key) != nullptr) {
      Fail("invalid GGUF bool " + std::string(key));
    }
    return value.value_or(fallback);
  }

  /// One value per block. Writers store a scalar when every block agrees and
  /// an integer array of any width otherwise.
  std::vector<std::uint32_t> PerLayer(std::string_view key,
                                      std::uint32_t layers) {
    std::vector<std::uint32_t> out;
    const auto* meta = gguf.FindMetadata(key);
    if (meta == nullptr) {
      Fail("missing GGUF value " + std::string(key));
      return out;
    }
    if (const auto scalar = gguf.GetMetadataUint32(key)) {
      out.assign(layers, *scalar);
      return out;
    }
    auto push = [&](std::uint64_t v) {
      if (v > UINT32_MAX) {
        Fail("GGUF array value out of range: " + std::string(key));
        return;
      }
      out.push_back(static_cast<std::uint32_t>(v));
    };
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      for (auto v : *u)
        push(v);
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (auto v : *s) {
        if (v < 0) {
          Fail("negative GGUF array value " + std::string(key));
          break;
        }
        push(static_cast<std::uint64_t>(v));
      }
    } else {
      Fail("GGUF value must be an integer or integer array: " +
           std::string(key));
    }
    if (ok && out.size() != layers) {
      Fail("GGUF array must describe every block: " + std::string(key));
    }
    return out;
  }
};

}  // namespace

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture") != kArchitecture) {
    r.Fail("GGUF architecture is not nemotron_h_moe");
    return std::nullopt;
  }

  Config c;
  auto key = [](std::string_view suffix) {
    return std::string(kArchitecture) + "." + std::string(suffix);
  };

  c.num_layers = r.U32(key("block_count"));
  if (r.U32(key("nextn_predict_layers"), false, 0) != 0) {
    r.Fail("nemotron_h_moe MTP blocks are not supported");
  }
  c.hidden_size = r.U32(key("embedding_length"));
  c.vocab_size = r.U32(key("vocab_size"));
  c.context_length = r.U32(key("context_length"));
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"), 1e-5F);

  c.head_dim = r.U32(key("attention.key_length"));
  if (r.U32(key("attention.value_length"), false, c.head_dim) != c.head_dim)
    r.Fail("attention value length must match the key length");

  c.ssm_conv_kernel = r.U32(key("ssm.conv_kernel"));
  c.ssm_state_size = r.U32(key("ssm.state_size"));
  c.ssm_groups = r.U32(key("ssm.group_count"));
  c.ssm_inner_size = r.U32(key("ssm.inner_size"));
  c.ssm_heads = r.U32(key("ssm.time_step_rank"));

  c.max_experts = r.U32(key("expert_count"));
  c.latent_size = r.U32(key("moe_latent_size"));
  c.shared_expert_ff = r.U32(key("expert_shared_feed_forward_length"));
  c.expert_weights_scale = r.F32(key("expert_weights_scale"), 1.0F, false);
  c.expert_weights_norm = r.Bool(key("expert_weights_norm"), false);
  if (r.U32(key("expert_shared_count")) != 1)
    r.Fail("exactly one shared expert is supported");
  if (r.U32(key("expert_gating_func")) != kSigmoidGating)
    r.Fail("only sigmoid expert gating is supported");
  if (r.U32(key("expert_group_count"), false, 1) != 1)
    r.Fail("grouped expert routing is not supported");
  if (!r.ok) {
    return std::nullopt;
  }

  const auto heads = r.PerLayer(key("attention.head_count"), c.num_layers);
  const auto kv_heads =
      r.PerLayer(key("attention.head_count_kv"), c.num_layers);
  const auto ff = r.PerLayer(key("feed_forward_length"), c.num_layers);
  const auto expert_ff =
      r.PerLayer(key("expert_feed_forward_length"), c.num_layers);
  const auto used = r.PerLayer(key("expert_used_count"), c.num_layers);
  if (!r.ok) {
    return std::nullopt;
  }

  // As in llama.cpp, a block is recurrent when it declares neither KV heads
  // nor a feed-forward width, attention when it has only KV heads, and MoE
  // when it has a feed-forward width.
  c.blocks.resize(c.num_layers);
  c.experts_used.assign(c.num_layers, 0);
  c.expert_ff.assign(c.num_layers, 0);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (kv_heads[il] == 0 && ff[il] == 0) {
      c.blocks[il] = BlockType::kMamba;
    } else if (ff[il] == 0) {
      c.blocks[il] = BlockType::kAttention;
      if (c.num_heads == 0) {
        c.num_heads = heads[il];
        c.num_kv_heads = kv_heads[il];
      } else if (heads[il] != c.num_heads || kv_heads[il] != c.num_kv_heads) {
        r.Fail("attention blocks must share one head geometry");
      }
    } else {
      c.blocks[il] = BlockType::kMoe;
      if (kv_heads[il] != 0)
        r.Fail("MoE blocks cannot declare attention heads");
      c.experts_used[il] = used[il];
      c.expert_ff[il] = expert_ff[il] != 0 ? expert_ff[il] : ff[il];
    }
  }
  if (!r.ok) {
    return std::nullopt;
  }

  // Structural limits of this runtime, not of the format.
  bool shape_ok = c.num_layers != 0 && c.context_length != 0 &&
                  c.context_length <= INT32_MAX && c.rms_eps > 0 &&
                  c.hidden_size != 0 && c.vocab_size != 0 && c.num_heads != 0 &&
                  c.num_kv_heads != 0 && c.num_heads % c.num_kv_heads == 0 &&
                  c.head_dim != 0 && c.ssm_conv_kernel >= 2 &&
                  c.ssm_state_size != 0 && c.ssm_groups != 0 &&
                  c.ssm_heads != 0 && c.ssm_heads % c.ssm_groups == 0 &&
                  c.ssm_inner_size % c.ssm_heads == 0 &&
                  c.ssm_inner_size % c.ssm_groups == 0 && c.max_experts != 0 &&
                  c.latent_size != 0 && c.shared_expert_ff != 0 &&
                  c.expert_weights_scale > 0;
  for (std::uint32_t il = 0; shape_ok && il < c.num_layers; ++il) {
    if (c.blocks[il] == BlockType::kMoe) {
      shape_ok = c.experts_used[il] != 0 &&
                 c.experts_used[il] <= c.max_experts && c.expert_ff[il] != 0;
    }
  }
  if (!shape_ok) {
    r.Fail("nemotron_h_moe metadata describes an unsupported shape");
    return std::nullopt;
  }
  // These dimensions will select specialized kernels. Reject other profiles
  // before allocating; per-block expert counts and widths stay free.
  if (c.num_layers != 88 || c.hidden_size != 4096 || c.num_heads != 32 ||
      c.num_kv_heads != 2 || c.head_dim != 128 || c.ssm_conv_kernel != 4 ||
      c.ssm_state_size != 128 || c.ssm_groups != 8 ||
      c.ssm_inner_size != 8192 || c.ssm_heads != 128 || c.latent_size != 1024 ||
      c.shared_expert_ff != 5376) {
    r.Fail("unsupported Nemotron 3.5 Bolt geometry");
    return std::nullopt;
  }
  return c;
}

std::optional<DrafterConfig> DrafterConfig::FromGguf(
    const core::GgufReader& gguf, const Config& target,
    std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture") != "dflash") {
    r.Fail("GGUF architecture is not dflash");
    return std::nullopt;
  }
  DrafterConfig c;
  c.num_layers = r.U32("dflash.block_count");
  c.hidden_size = r.U32("dflash.embedding_length");
  c.ff_size = r.U32("dflash.feed_forward_length");
  c.num_heads = r.U32("dflash.attention.head_count");
  c.num_kv_heads = r.U32("dflash.attention.head_count_kv");
  c.head_dim = r.U32("dflash.attention.key_length");
  if (r.U32("dflash.attention.value_length", false, c.head_dim) != c.head_dim)
    r.Fail("drafter value length must match the key length");
  c.rms_eps = r.F32("dflash.attention.layer_norm_rms_epsilon", 1e-6F);
  c.rope_theta = r.F32("dflash.rope.freq_base", 10000.0F);
  if (r.U32("dflash.rope.dimension_count", false, c.head_dim) != c.head_dim)
    r.Fail("drafter RoPE must rotate the whole head");
  c.block_size = r.U32("dflash.block_size");
  c.sliding_window = r.U32("dflash.attention.sliding_window");
  c.conv_kernel = r.U32("dflash.conv_kernel_size");
  c.conv_group = r.U32("dflash.conv_group_size");
  if (r.U32("dflash.selector_rank", false, 0) != 0 ||
      r.U32("dflash.selector_top_k", false, 0) != 0)
    r.Fail("DFlash2 selector drafters are not supported");
  if (r.Bool("dflash.attention.causal", true))
    r.Fail("the drafter must attend non-causally within its block");
  for (const char* key :
       {"dflash.embedding_scale", "dflash.attention.scale",
        "dflash.attention.value_scale", "dflash.logit_scale",
        "dflash.final_logit_softcapping", "dflash.hyper_connection_count"}) {
    if (gguf.FindMetadata(key) != nullptr)
      r.Fail(std::string("unsupported drafter metadata ") + key);
  }
  if (const auto act = gguf.GetMetadataString("dflash.hidden_act");
      act && *act != "silu")
    r.Fail("the drafter FFN must use SiLU");
  if (gguf.FindMetadata("dflash.rope.dimension_sections") != nullptr)
    r.Fail("M-RoPE drafters are not supported");
  const auto mask = gguf.GetMetadataUint32("tokenizer.ggml.mask_token_id");
  if (!mask)
    r.Fail("the drafter has no mask token");
  c.mask_token = mask.value_or(0);
  c.lc_hidden = r.U32("lilicorr.hidden_size");
  c.lc_top_k = r.U32("lilicorr.candidate_topk");
  c.lc_heads = r.U32("lilicorr.num_heads");
  c.lc_layers = r.U32("lilicorr.num_layers");
  c.lc_factor = r.U32("lilicorr.factor_dim");
  c.lc_mlp = r.U32("lilicorr.mlp_hidden_size");
  c.lc_vector_eps = r.F32("lilicorr.vector_eps", 1e-4F);
  c.lc_logit_scale = r.F32("lilicorr.logit_scale", 8.0F);
  if (!r.ok)
    return std::nullopt;
  const auto window =
      r.PerLayer("dflash.attention.sliding_window_pattern", c.num_layers);
  const auto* layers = gguf.FindMetadata("dflash.target_layers");
  if (layers != nullptr) {
    if (const auto* s = std::get_if<std::vector<std::int64_t>>(&layers->value))
      for (auto v : *s)
        c.target_layers.push_back(v < 0 || v >= target.num_layers
                                      ? UINT32_MAX
                                      : static_cast<std::uint32_t>(v));
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&layers->value))
      for (auto v : *u)
        c.target_layers.push_back(v >= target.num_layers
                                      ? UINT32_MAX
                                      : static_cast<std::uint32_t>(v));
  }
  if (!r.ok)
    return std::nullopt;
  bool ok = !c.target_layers.empty() &&
            std::is_sorted(c.target_layers.begin(), c.target_layers.end());
  for (auto layer : c.target_layers)
    ok = ok && layer != UINT32_MAX;
  for (auto w : window)
    ok = ok && w == 1;
  if (!ok) {
    r.Fail("drafter target layers or sliding-window pattern are invalid");
    return std::nullopt;
  }
  ok = c.hidden_size == target.hidden_size && c.num_layers != 0 &&
       c.ff_size != 0 && c.num_heads != 0 && c.num_kv_heads != 0 &&
       c.num_heads % c.num_kv_heads == 0 && c.head_dim % 2 == 0 &&
       c.rms_eps > 0 && c.rope_theta > 0 && c.block_size >= 2 &&
       c.sliding_window != 0 && c.conv_kernel != 0 && c.conv_group != 0 &&
       c.hidden_size % c.conv_group == 0 && c.mask_token < target.vocab_size &&
       c.lc_hidden != 0 && c.lc_heads != 0 && c.lc_hidden % c.lc_heads == 0 &&
       c.lc_top_k != 0 && c.lc_layers != 0 && c.lc_factor != 0 &&
       c.lc_mlp != 0 && c.lc_vector_eps > 0 &&
       static_cast<std::uint64_t>(c.lc_top_k) * (c.lc_top_k + 1) <=
           c.hidden_size;
  if (!ok) {
    r.Fail("dflash metadata describes an unsupported drafter");
    return std::nullopt;
  }
  // The kernels below specialize on NVIDIA's drafter for Bolt.
  if (c.num_layers != 5 || c.num_heads != 32 || c.num_kv_heads != 2 ||
      c.head_dim != 128 || c.block_size != 8 || c.conv_kernel != 2 ||
      c.conv_group != 16 || c.lc_hidden != 1024 || c.lc_top_k != 8 ||
      c.lc_heads != 8 || c.lc_layers != 2 || c.lc_factor != 1024) {
    r.Fail("unsupported LiLiCorr drafter geometry");
    return std::nullopt;
  }
  return c;
}

}  // namespace gufo::models::nemotron35_bolt
