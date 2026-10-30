#include "src/models/lfm3/config.hpp"

#include <cmath>
#include <stdexcept>
namespace gufo::models::lfm3 {
std::optional<Config> Config::FromGguf(const core::GgufReader& r,
                                       std::string* error) {
  try {
    if (r.GetMetadataString("general.architecture") != "lfm3")
      throw std::runtime_error("GGUF architecture must be lfm3");
    auto u = [&](const char* suffix) {
      auto v = r.GetMetadataUint32(std::string("lfm3.") + suffix);
      if (!v || *v == 0)
        throw std::runtime_error(std::string("invalid lfm3.") + suffix);
      return *v;
    };
    auto f = [&](const char* suffix) {
      auto v = r.GetMetadataFloat32(std::string("lfm3.") + suffix);
      if (!v || !std::isfinite(*v) || *v <= 0)
        throw std::runtime_error(std::string("invalid lfm3.") + suffix);
      return *v;
    };
    Config c;
    c.decision = !r.GetMetadataBool("lfm3.attention.causal").value_or(true);
    if (c.decision &&
        r.GetMetadataString("lfm3.decision.type") != "lfm3-d1-omni")
      throw std::runtime_error("unsupported LFM3 decision model");
    const auto n = u("block_count");
    c.hidden = u("embedding_length");
    c.intermediate = u("feed_forward_length");
    c.context = u("context_length");
    c.heads = u("attention.head_count");
    c.head_dim = u("attention.key_length");
    c.window = u("attention.sliding_window");
    c.conv_width = u("shortconv.l_cache");
    c.eps = f("attention.layer_norm_rms_epsilon");
    c.theta = f("rope.freq_base");
    c.theta_swa = f("rope.freq_base_swa");
    c.vocab = r.GetMetadataStringArray("tokenizer.ggml.tokens").size();
    if (n > 256 || c.hidden > 16384 || c.intermediate > 131072 ||
        c.heads > 128 || c.head_dim != 128 || c.conv_width < 2 ||
        c.conv_width > 32 || c.vocab == 0 ||
        u("attention.value_length") != c.head_dim ||
        u("rope.dimension_count") != c.head_dim)
      throw std::runtime_error("unsupported LFM3 dimensions");
    const auto* kv = r.FindMetadata("lfm3.attention.head_count_kv");
    const auto* sw = r.FindMetadata("lfm3.attention.sliding_window_pattern");
    if (!kv || !sw)
      throw std::runtime_error("missing LFM3 layer layout");
    auto at = [](const core::GgufMetadataValue& m,
                 std::size_t i) -> std::uint64_t {
      if (auto* a = std::get_if<std::vector<std::uint64_t>>(&m.value);
          a && i < a->size())
        return (*a)[i];
      if (auto* a = std::get_if<std::vector<std::int64_t>>(&m.value);
          a && i < a->size() && (*a)[i] >= 0)
        return (*a)[i];
      throw std::runtime_error("invalid LFM3 per-layer array");
    };
    auto length = [](const core::GgufMetadataValue& m) -> std::size_t {
      if (auto* a = std::get_if<std::vector<std::uint64_t>>(&m.value))
        return a->size();
      if (auto* a = std::get_if<std::vector<std::int64_t>>(&m.value))
        return a->size();
      return 0;
    };
    if (length(*kv) != n || length(*sw) != n)
      throw std::runtime_error("LFM3 arrays must describe every layer");
    for (std::uint32_t i = 0; i < n; ++i) {
      auto h = at(*kv, i), s = at(*sw, i);
      if (h > c.heads || (h && c.heads % h) || s > 1 || (h == 0 && s) ||
          (c.decision && h && (c.heads / h > 8 || 64 % (c.heads / h))))
        throw std::runtime_error("invalid LFM3 layer heads/window");
      c.kv_heads.push_back(h);
      c.layers.push_back(!h  ? LayerType::kConvolution
                         : s ? LayerType::kSlidingAttention
                             : LayerType::kFullAttention);
    }
    return c;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return std::nullopt;
  }
}
}  // namespace gufo::models::lfm3
