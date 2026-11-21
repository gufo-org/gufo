#ifndef GUFO_MODELS_YUE2_CONFIG_HPP_
#define GUFO_MODELS_YUE2_CONFIG_HPP_

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "src/models/yue2/protocol.hpp"

namespace gufo::models::yue2 {

/// The released backbone shape; anything else is rejected at load.
struct BackboneConfig {
  std::size_t hidden{2048};
  std::size_t layers{28};
  std::size_t heads{16};
  std::size_t kv_heads{8};
  std::size_t head_dim{128};
  std::size_t intermediate{6144};
  std::size_t vocabulary{kVocabularySize};
  std::size_t context{kContext};
  std::size_t latent_dim{kLatentDim};
  std::size_t max_latent_frames{24576};
  float rms_epsilon{1e-6F};
  float rope_theta{1000000.0F};
  float timestep_shift{1.0F};
};

struct GenerationDefaults {
  Sampling abc{DefaultAbcSampling()};
  Sampling semantic{DefaultSemanticSampling()};
  int ode_steps{32};
};

/// The Oobleck decoder layout of YuE2-Vae and YuE2-Vae-legacy.
struct VaeConfig {
  std::string release_variant;
  int sample_rate{48000};
  int channels{2};
  int latent_dim{64};
  int base_channels{64};
  int multipliers[7]{1, 1, 2, 4, 8, 16, 32};
  int strides[6]{2, 2, 4, 4, 5, 6};
  int core_frames{1024};
  int halo_frames{16};
};

struct ModelPaths {
  std::filesystem::path model;
  std::filesystem::path vae;
};

/// `config.json` `model_type == "yue2"`, `qwen.tiktoken` and a
/// `yue2_generation_config.json` of version "yue2-native-v1".
[[nodiscard]] bool LooksLikeYue2(const std::filesystem::path& model_root);

/// The backbone directory of `root`: itself or a saved pipeline's model.
[[nodiscard]] std::optional<std::filesystem::path> FindModelDirectory(
    const std::filesystem::path& root);

/// Resolves the model directory and its decoder: an explicit `vae_root`,
/// else a saved pipeline (`<root>/pipeline.json` naming both), else a
/// sibling `YuE2-Vae` directory.
[[nodiscard]] std::optional<ModelPaths> ResolveModelPaths(
    const std::filesystem::path& root, const std::filesystem::path& vae_root,
    std::string* error);

[[nodiscard]] bool LoadBackboneConfig(const std::filesystem::path& model_root,
                                      BackboneConfig* config,
                                      GenerationDefaults* defaults,
                                      std::string* error);
[[nodiscard]] bool LoadVaeConfig(const std::filesystem::path& vae_root,
                                 VaeConfig* config, std::string* error);

}  // namespace gufo::models::yue2

#endif  // GUFO_MODELS_YUE2_CONFIG_HPP_
