#ifndef GUFO_MODELS_LFM3_CONFIG_HPP_
#define GUFO_MODELS_LFM3_CONFIG_HPP_
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
namespace gufo::models::lfm3 {
enum class LayerType { kConvolution, kFullAttention, kSlidingAttention };
struct Config {
  std::uint32_t hidden{}, intermediate{}, vocab{}, context{}, heads{},
      head_dim{};
  std::uint32_t window{}, conv_width{};
  float eps{}, theta{}, theta_swa{};
  bool decision{};
  std::vector<std::uint32_t> kv_heads;
  std::vector<LayerType> layers;
  static std::optional<Config> FromGguf(const core::GgufReader& reader,
                                        std::string* error = nullptr);
};
}  // namespace gufo::models::lfm3
#endif
