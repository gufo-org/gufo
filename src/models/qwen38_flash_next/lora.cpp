#include "src/models/qwen38_flash_next/lora.hpp"

#include <utility>

namespace gufo::models::qwen38_flash_next {
namespace {

constexpr std::string_view kExpectedTargets =
    "blk.N.ssm_out.weight, blk.N.attn_output.weight or "
    "blk.N.ffn_down_shexp.weight of a trunk layer";

/// The binder stores these trunk projections as single matrices
/// (experts == 1), so each maps onto one adapter target.
std::vector<qwen::lora::LoraMatrix> Matrices(
    std::span<const TensorRef> targets) {
  std::vector<qwen::lora::LoraMatrix> matrices;
  for (const auto& t : targets) {
    if (t.experts == 1) {
      matrices.push_back({t.name, t.data, t.type, t.rows, t.cols});
    }
  }
  return matrices;
}

}  // namespace

std::vector<TensorRef> LoraTargets(const ModelWeights& weights) {
  std::vector<TensorRef> targets;
  for (const auto& layer : weights.layers) {
    for (const TensorRef* t :
         {&layer.ssm_out, &layer.attn_out, &layer.shexp_down}) {
      if (!t->empty()) {
        targets.push_back(*t);
      }
    }
  }
  return targets;
}

std::unique_ptr<LoraAdapter> OpenLoraAdapter(const std::filesystem::path& path,
                                             float scale,
                                             std::string_view architecture,
                                             std::span<const TensorRef> targets,
                                             std::string* error) {
  return LoraAdapter::Open(path, scale, architecture, Matrices(targets),
                           kExpectedTargets, error);
}

std::unique_ptr<LoraAdapter> OpenLoraAdapter(
    std::unique_ptr<core::GgufReader> reader, std::string label, float scale,
    std::string_view architecture, std::span<const TensorRef> targets,
    std::string* error) {
  return LoraAdapter::Open(std::move(reader), std::move(label), scale,
                           architecture, Matrices(targets), kExpectedTargets,
                           error);
}

}  // namespace gufo::models::qwen38_flash_next
