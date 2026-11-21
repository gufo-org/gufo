#include "staging/models/glm5_next/vision/config.hpp"

#include <cmath>
#include <cstdint>
#include <string_view>
#include <variant>

namespace gufo::models::glm5_next::vision {
namespace {

constexpr std::string_view kArchitecture = "clip";
constexpr std::string_view kProjector = "glm5v";

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

  std::uint64_t U64(std::string_view key, bool required = true,
                    std::uint64_t fallback = 0) {
    const std::uint64_t value = U32(key, required,
                                    static_cast<std::uint32_t>(fallback));
    return value;
  }

  float F32(std::string_view key, bool required = true, float fallback = 0.0F) {
    const auto value = gguf.GetMetadataFloat32(key);
    if (value.has_value() && std::isfinite(*value)) {
      return *value;
    }
    if (required || gguf.FindMetadata(key) != nullptr) {
      Fail("missing or invalid GGUF float " + std::string(key));
    }
    return fallback;
  }

  std::vector<float> F32Array(std::string_view key, std::size_t expected) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<float> out;
    if (meta == nullptr) {
      Fail("missing GGUF array " + std::string(key));
      return out;
    }
    const auto* values = std::get_if<std::vector<double>>(&meta->value);
    if (values == nullptr || values->size() != expected) {
      Fail("GGUF array must hold " + std::to_string(expected) +
           " floats: " + std::string(key));
      return out;
    }
    for (const double value : *values) {
      if (!std::isfinite(value)) {
        Fail("non-finite GGUF array value " + std::string(key));
        return {};
      }
      out.push_back(static_cast<float>(value));
    }
    return out;
  }
};

}  // namespace

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  const auto architecture = gguf.GetMetadataString("general.architecture");
  if (architecture != kArchitecture) {
    r.Fail("mmproj architecture is not clip");
    return std::nullopt;
  }
  const auto projector = gguf.GetMetadataString("clip.projector_type");
  if (projector != kProjector) {
    r.Fail("clip projector is not glm5v");
    return std::nullopt;
  }

  Config c;
  c.projection_dim = r.U32("clip.vision.projection_dim");
  c.image_size = r.U32("clip.vision.image_size");
  c.patch_size = r.U32("clip.vision.patch_size");
  c.embedding_length = r.U32("clip.vision.embedding_length");
  c.feed_forward_length = r.U32("clip.vision.feed_forward_length");
  c.block_count = r.U32("clip.vision.block_count");
  c.head_count = r.U32("clip.vision.attention.head_count");
  c.layer_norm_epsilon =
      r.F32("clip.vision.attention.layer_norm_epsilon", true, 1e-5F);
  c.merge_size = r.U32("clip.vision.spatial_merge_size", false, 2);
  c.swiglu_clamp = r.F32("clip.vision.swiglu_clamp", false, 10.0F);
  c.min_pixels = r.U64("clip.vision.image_min_pixels", true, 0);
  c.max_pixels = r.U64("clip.vision.image_max_pixels", true, 0);
  c.image_mean = r.F32Array("clip.vision.image_mean", 3);
  c.image_std = r.F32Array("clip.vision.image_std", 3);
  if (!r.ok) {
    return std::nullopt;
  }

  if (c.block_count == 0 || c.embedding_length == 0 || c.head_count == 0 ||
      c.patch_size == 0 || c.image_size == 0 || c.projection_dim == 0 ||
      c.feed_forward_length == 0 || c.embedding_length % c.head_count != 0 ||
      (c.embedding_length / c.head_count) % 4 != 0 ||
      !std::isfinite(c.layer_norm_epsilon) || c.layer_norm_epsilon <= 0 ||
      !std::isfinite(c.swiglu_clamp) || c.swiglu_clamp <= 0 ||
      c.min_pixels == 0 || c.max_pixels < c.min_pixels) {
    r.Fail("clip.vision metadata describes an unsupported shape");
    return std::nullopt;
  }

  // These dimensions select the specialized patch-embed, RoPE and merger
  // kernels. Reject other profiles before allocating or launching.
  if (c.projection_dim != 4096 || c.image_size != 448 || c.patch_size != 14 ||
      c.embedding_length != 1024 || c.feed_forward_length != 4096 ||
      c.block_count != 24 || c.head_count != 16 || c.merge_size != 2) {
    r.Fail("unsupported GLM-5 vision kernel geometry");
    return std::nullopt;
  }
  return c;
}

}  // namespace gufo::models::glm5_next::vision
