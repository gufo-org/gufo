#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_PROJECTION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_PROJECTION_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::qwen38_flash_next {

/// A runtime residual-stream projection recovered from a llama.cpp
/// control-vector GGUF: at each steered layer boundary the residual loses its
/// component along a unit direction, `h <- h - s * (h . v) * v`, applied to
/// every hyper-connection stream after the FFN/MoE combine write and before
/// the next mixer's grouped norm consumes it. The GGUF weights are never
/// modified; an unloaded vector leaves the engine on today's code path.
struct ProjectionVector {
  /// Unit directions, `[num_layers][hidden]`; layers without a `direction.N`
  /// tensor keep a zero row and a zero scale.
  std::vector<float> directions;
  /// Per-layer scale `s`: the norm the loader normalized away. Zero leaves
  /// that layer's boundary unsteered.
  std::vector<float> scales;
  std::uint32_t num_layers = 0;
  std::uint32_t hidden = 0;
  /// Inclusive layer range steered at the FFN/MoE combine boundary. Layer 0
  /// is never steered (the file carries directions for layers 1..L-1 only).
  std::uint32_t first_layer = 1;
  std::uint32_t last_layer = 0;

  [[nodiscard]] bool Empty() const noexcept { return scales.empty(); }
};

/// Parses a llama.cpp control-vector GGUF (`direction.N` F32 tensors, layers
/// 1..L-1, `direction.l` steering block `l`). Project-mode loader semantics,
/// shared by Strata and the llama.cpp reference: each direction is normalized
/// to unit length and its norm becomes that layer's scale, so a unit-norm
/// file reproduces `P = I - v v^T` exactly. A `controlvector.model_hint` key
/// that mismatches `expected_hint` is reported through `warning`; a missing
/// key (the common case) stays silent.
[[nodiscard]] bool LoadControlVector(
    const std::string& path, std::uint32_t num_layers, std::uint32_t hidden,
    std::string_view expected_hint, ProjectionVector* out,
    std::string* error_msg, std::string* warning = nullptr);

}  // namespace gufo::models::qwen38_flash_next

#endif
