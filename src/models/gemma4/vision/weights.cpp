#include "src/models/gemma4/vision/weights.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace gufo::models::gemma4::vision {

Weights Weights::Resolve(const core::GgufReader& reader,
                         std::uint32_t output_width) {
  const auto fail = [](const std::string& what) {
    throw std::invalid_argument("Gemma 4 vision sidecar: " + what);
  };
  if (reader.GetMetadataString("general.architecture") != "clip" ||
      reader.GetMetadataString("general.type") != "mmproj" ||
      reader.GetMetadataBool("clip.has_vision_encoder") != true ||
      reader.GetMetadataString("clip.vision.projector_type") != "gemma4v") {
    fail("not a gemma4v projector");
  }
  const auto expect = [&](std::string_view key, std::uint64_t value) {
    if (reader.GetMetadataUint64(key) != value) {
      fail("unexpected " + std::string(key));
    }
  };
  expect("clip.vision.projection_dim", output_width);
  expect("clip.vision.patch_size", 16);
  expect("clip.vision.embedding_length", kHidden);
  expect("clip.vision.feed_forward_length", kFfn);
  expect("clip.vision.block_count", kLayers);
  expect("clip.vision.attention.head_count", kHeads);
  if (reader.GetMetadataFloat32("clip.vision.attention.layer_norm_epsilon") !=
      kEps) {
    fail("unexpected layer_norm_epsilon");
  }
  for (const auto key : {"clip.vision.image_mean", "clip.vision.image_std"}) {
    const auto* value = reader.FindMetadata(key);
    const auto* array =
        value ? std::get_if<std::vector<double>>(&value->value) : nullptr;
    const double expected = std::string_view(key).ends_with("mean") ? 0 : 1;
    if (!array || *array != std::vector<double>(3, expected)) {
      fail("unsupported image normalization");
    }
  }
  // The pooling kernel is fixed at 3 unless the sidecar says otherwise.
  if (const auto merge =
          reader.GetMetadataUint64("clip.vision.projector.scale_factor");
      merge && *merge != 3) {
    fail("unsupported pooling kernel");
  }

  const auto require = [&](const std::string& name,
                           std::vector<std::uint64_t> shape,
                           core::GgmlType type) {
    const auto* tensor = reader.FindTensor(name);
    if (tensor == nullptr || tensor->dimensions != shape ||
        tensor->type != type || tensor->data == nullptr ||
        tensor->size_bytes !=
            tensor->ElementCount() * (type == core::GgmlType::kF32 ? 4 : 2)) {
      fail("invalid tensor " + name);
    }
    return tensor;
  };
  constexpr auto kF32 = core::GgmlType::kF32;
  constexpr auto kBF16 = core::GgmlType::kBF16;
  Weights w;
  w.output_width = output_width;
  w.patch = require("v.patch_embd.weight", {16, 16, 3, kHidden}, kF32);
  w.position =
      require("v.position_embd.weight", {kHidden, kPositions, 2}, kF32);
  w.std_bias = require("v.std_bias", {kHidden}, kF32);
  w.std_scale = require("v.std_scale", {kHidden}, kF32);
  w.projection =
      require("mm.input_projection.weight", {kHidden, output_width}, kBF16);
  for (std::uint32_t l = 0; l < kLayers; ++l) {
    const std::string p = "v.blk." + std::to_string(l) + ".";
    auto& L = w.layers[l];
    L.ln1 = require(p + "ln1.weight", {kHidden}, kF32);
    L.q = require(p + "attn_q.weight", {kHidden, kHidden}, kBF16);
    L.k = require(p + "attn_k.weight", {kHidden, kHidden}, kBF16);
    L.v = require(p + "attn_v.weight", {kHidden, kHidden}, kBF16);
    L.q_norm = require(p + "attn_q_norm.weight", {kHeadDim}, kF32);
    L.k_norm = require(p + "attn_k_norm.weight", {kHeadDim}, kF32);
    L.out = require(p + "attn_out.weight", {kHidden, kHidden}, kBF16);
    L.attn_post_norm = require(p + "attn_post_norm.weight", {kHidden}, kF32);
    L.ln2 = require(p + "ln2.weight", {kHidden}, kF32);
    L.gate = require(p + "ffn_gate.weight", {kHidden, kFfn}, kBF16);
    L.up = require(p + "ffn_up.weight", {kHidden, kFfn}, kBF16);
    L.down = require(p + "ffn_down.weight", {kFfn, kHidden}, kBF16);
    L.ffn_post_norm = require(p + "ffn_post_norm.weight", {kHidden}, kF32);
  }
  // Clamp scalars (Gemma4ClippableLinear) or biases would change the graph.
  constexpr std::uint64_t kTensors = 5 + 13 * kLayers;
  if (reader.GetTensorCount() != kTensors) {
    fail("unexpected tensors (clipped linears or biases are unsupported)");
  }
  return w;
}

}  // namespace gufo::models::gemma4::vision
