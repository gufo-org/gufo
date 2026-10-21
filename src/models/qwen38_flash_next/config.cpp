#include "src/models/qwen38_flash_next/config.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>
#include <utility>
#include <variant>

namespace gufo::models::qwen38_flash_next {
namespace {

constexpr std::string_view kArchitecture = "qwen4exp";

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

  float F32(std::string_view key, float fallback) {
    const auto value = gguf.GetMetadataFloat32(key);
    if (!value || !std::isfinite(*value))
      Fail("missing or invalid GGUF float " + std::string(key));
    return value.value_or(fallback);
  }

  /// Integer arrays of any width; GGUF writers pick the narrowest type.
  std::vector<std::uint64_t> U64Array(std::string_view key,
                                      bool required = true) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<std::uint64_t> out;
    if (meta == nullptr) {
      if (required)
        Fail("missing GGUF array " + std::string(key));
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out = *u;
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (auto v : *s) {
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

}  // namespace

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       bool require_trunk,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture") != kArchitecture) {
    r.Fail("GGUF architecture is not qwen4exp");
    return std::nullopt;
  }

  Config c;
  const std::string_view p = "qwen4exp.";
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
  c.context_length = r.U32(key("context_length"));
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"), 1e-6F);

  c.hc_count = r.U32(key("hyper_connection.count"));
  c.hc_low_rank = r.U32(key("hyper_connection.low_rank"));

  c.full_attention_interval = r.U32(key("full_attention_interval"));
  c.num_heads = r.U32(key("attention.head_count"));
  c.num_kv_heads = r.U32(key("attention.head_count_kv"));
  c.head_dim = r.U32(key("attention.key_length"));
  if (r.U32(key("attention.value_length"), false, c.head_dim) != c.head_dim)
    r.Fail("attention value length must match the supported key length");
  c.rotary_dim = r.U32(key("rope.dimension_count"));
  c.rope_theta = r.F32(key("rope.freq_base"), 10000000.0F);
  {
    const auto sections = r.U64Array(key("rope.dimension_sections"));
    if (sections.size() != 4 || sections[3] != 0) {
      r.Fail("rope.dimension_sections must contain three axes and a zero pad");
    } else {
      std::uint64_t sum = 0;
      for (std::size_t i = 0; i < 3; ++i) {
        if (sections[i] == 0 || sections[i] > c.rotary_dim / 2) {
          r.Fail("invalid mRoPE axis dimension");
          break;
        }
        sum += sections[i];
        c.rope_sections[i] = static_cast<std::uint32_t>(sections[i]);
      }
      if (sum != c.rotary_dim / 2)
        r.Fail("mRoPE sections must sum to half the rotary dimension");
    }
  }

  c.indexer_heads = r.U32(key("attention.indexer.head_count"));
  c.indexer_head_dim = r.U32(key("attention.indexer.key_length"));
  c.indexer_top_k = r.U32(key("attention.indexer.top_k"));
  {
    // One ratio per layer; zero on linear layers. Every attention layer of
    // this model shares one value, which is all the runtime implements.
    const auto ratios = r.U64Array(key("attention.compress_ratios"));
    if (ratios.size() != c.num_layers_all || c.full_attention_interval == 0) {
      r.Fail("attention.compress_ratios must describe every layer");
    } else if (!require_trunk && c.nextn_layers == 1 &&
               std::all_of(ratios.begin(), ratios.end(),
                           [](std::uint64_t ratio) { return ratio == 0; })) {
      // Draft-only Orca exports contain no trunk tensors and write a zero
      // placeholder for every ratio. As with the zero nextn entry below,
      // the exported indexer belongs to the ratio-four Flash-Next predictor.
      // MtpMatches still requires agreement with the validated target.
      c.compress_ratio = 4;
    } else {
      for (std::size_t i = 0; i < ratios.size(); ++i) {
        if (i >= c.num_layers) {
          // Older GGUF converters append zero for nextn layers even though
          // their exported attention includes QSA indexer weights. The pinned
          // Flash-Next architecture uses the same ratio-four indexer in MTP.
          if (ratios[i] != 0 && ratios[i] != 4)
            r.Fail("unsupported MTP attention compression ratio");
          continue;
        }
        const bool sparse = !c.IsLinearLayer(i);
        if (!sparse) {
          if (ratios[i] != 0)
            r.Fail("linear layers cannot use attention compression");
        } else if (ratios[i] == 0 || ratios[i] > 4096) {
          r.Fail("attention compression must fit the 4096-row index ring");
        } else if (c.compress_ratio == 0) {
          c.compress_ratio = static_cast<std::uint32_t>(ratios[i]);
        } else if (c.compress_ratio != ratios[i]) {
          r.Fail("attention.compress_ratios differ between attention layers");
        }
      }
    }
  }

  c.ssm_conv_kernel = r.U32(key("ssm.conv_kernel"));
  c.ssm_head_dim = r.U32(key("ssm.state_size"));
  c.ssm_num_k_heads = r.U32(key("ssm.group_count"));
  c.ssm_num_v_heads = r.U32(key("ssm.time_step_rank"));
  c.ssm_inner_size = r.U32(key("ssm.inner_size"));

  c.num_experts = r.U32(key("expert_count"));
  c.num_experts_used = r.U32(key("expert_used_count"));
  c.expert_ff = r.U32(key("expert_feed_forward_length"));
  c.shared_expert_ff = r.U32(key("expert_shared_feed_forward_length"));

  {
    const auto layers = r.U64Array(key("ple.layers"), false);
    if (layers.size() > 1) {
      r.Fail("only one PLE layer is supported");
    } else if (layers.size() == 1) {
      if (layers[0] >= c.num_layers) {
        r.Fail("PLE layer lies outside the trunk");
        return std::nullopt;
      }
      c.ple_layer = static_cast<std::int32_t>(layers[0]);
      c.ple_ngram_size = r.U32(key("ple.ngram_size"));
      c.ple_heads_per_ngram = r.U32(key("ple.heads_per_ngram"));
      c.ple_conv_kernel = r.U32(key("ple.conv_kernel"));
      c.ple_eos_token = r.U32(key("ple.eos_token_id"));
      c.ple_head_dim = r.U32(key("embedding_length_per_layer_input"));
      if (c.ple_ngram_size < 2 || c.ple_ngram_size > kMaxPleNgram ||
          c.ple_heads_per_ngram == 0 ||
          c.ple_heads_per_ngram > kMaxPleHeads / (c.ple_ngram_size - 1) ||
          c.ple_head_dim == 0 || c.ple_conv_kernel == 0) {
        r.Fail("unsupported PLE n-gram geometry");
      } else {
        c.ple_heads = (c.ple_ngram_size - 1) * c.ple_heads_per_ngram;
        const auto multipliers = r.U64Array(key("ple.layer_multipliers"));
        const auto offsets = r.U64Array(key("ple.head_offsets"));
        const auto vocab = r.U64Array(key("ple.head_vocab_sizes"));
        if (multipliers.size() != c.ple_ngram_size ||
            offsets.size() != c.ple_heads || vocab.size() != c.ple_heads) {
          r.Fail("PLE hash tables must match the head count");
        } else {
          std::copy_n(multipliers.begin(), c.ple_ngram_size,
                      c.ple_multipliers.begin());
          for (std::uint32_t h = 0; h < c.ple_heads; ++h) {
            if (vocab[h] == 0 || vocab[h] > UINT32_MAX ||
                offsets[h] > UINT32_MAX - vocab[h]) {
              r.Fail("PLE head range does not fit a 32-bit row index");
              break;
            }
            c.ple_head_offsets[h] = static_cast<std::uint32_t>(offsets[h]);
            c.ple_head_vocab[h] = static_cast<std::uint32_t>(vocab[h]);
            c.ple_rows = std::max(c.ple_rows, offsets[h] + vocab[h]);
          }
        }
      }
    }
  }

  if (!r.ok) {
    return std::nullopt;
  }

  // Structural limits of this runtime, not of the format.
  if (c.context_length == 0 || c.context_length > INT32_MAX ||
      !std::isfinite(c.rms_eps) || c.rms_eps <= 0 ||
      !std::isfinite(c.rope_theta) || c.rope_theta <= 0 || c.hidden_size == 0 ||
      c.hc_count < 2 || c.hc_low_rank == 0 || c.full_attention_interval == 0 ||
      c.num_heads == 0 || c.num_kv_heads == 0 ||
      c.num_heads % c.num_kv_heads != 0 || c.head_dim == 0 ||
      c.rotary_dim == 0 || c.rotary_dim % 2 != 0 || c.rotary_dim > c.head_dim ||
      c.indexer_heads == 0 || c.indexer_head_dim == 0 ||
      c.rotary_dim > c.indexer_head_dim || c.compress_ratio == 0 ||
      c.indexer_top_k == 0 || c.indexer_top_k % c.compress_ratio != 0 ||
      c.ssm_conv_kernel == 0 || c.ssm_head_dim == 0 || c.ssm_num_k_heads == 0 ||
      c.ssm_num_v_heads == 0 || c.ssm_num_v_heads % c.ssm_num_k_heads != 0 ||
      c.ssm_inner_size != c.SsmValueDim() || c.num_experts == 0 ||
      c.num_experts_used == 0 || c.num_experts_used > c.num_experts ||
      c.expert_ff == 0 || c.shared_expert_ff == 0) {
    r.Fail("qwen4exp metadata describes an unsupported shape");
    return std::nullopt;
  }
  // These dimensions select specialized attention, indexer, DeltaNet and
  // expert kernels. Reject other profiles before allocating or launching;
  // satisfying divisibility alone does not make a shape executable.
  if (c.num_layers != 48 || c.nextn_layers > 1 || c.hidden_size != 2560 ||
      c.hc_count != 4 || c.hc_low_rank != 320 ||
      c.full_attention_interval != 4 || c.num_heads != 24 ||
      c.num_kv_heads != 2 || c.head_dim != 256 || c.rotary_dim != 64 ||
      c.rope_sections != std::array<std::uint32_t, 4>{11, 11, 10, 0} ||
      c.indexer_heads != 4 || c.indexer_head_dim != 128 ||
      c.indexer_top_k != 2048 || c.compress_ratio != 4 ||
      c.ssm_conv_kernel != 4 || c.ssm_head_dim != 128 ||
      c.ssm_num_k_heads != 16 || c.ssm_num_v_heads != 48 ||
      c.num_experts != 512 || c.num_experts_used != 10 || c.expert_ff != 640 ||
      c.shared_expert_ff != 640 ||
      (c.ple_layer >= 0 &&
       (c.ple_ngram_size != 3 || c.ple_heads_per_ngram != 8 ||
        c.ple_head_dim != 160 || c.ple_conv_kernel != 4))) {
    r.Fail("unsupported Qwen3.8-Flash-Next kernel geometry");
    return std::nullopt;
  }
  if (require_trunk) {
    if (c.num_layers == 0 || c.num_layers % c.full_attention_interval != 0) {
      r.Fail("qwen4exp trunk layer count is not a multiple of the interval");
      return std::nullopt;
    }
    if (c.ple_layer >= 0 &&
        !c.IsLinearLayer(static_cast<std::uint32_t>(c.ple_layer))) {
      r.Fail("PLE layer must be a linear attention layer");
      return std::nullopt;
    }
  }
  return c;
}

}  // namespace gufo::models::qwen38_flash_next
