#include "src/models/qwen/lora.hpp"

#include <cstdint>

namespace gufo::models::qwen {
namespace {

std::string Shape(const std::vector<std::uint64_t>& dimensions) {
  std::string text = "[";
  for (std::size_t i = 0; i < dimensions.size(); ++i) {
    text += (i == 0 ? "" : ", ") + std::to_string(dimensions[i]);
  }
  return text + "]";
}

}  // namespace

std::optional<std::vector<lora::LoraMatrix>> LoraTargets(
    const core::GgufReader& reader, const QwenModelWeights& weights,
    std::string* error) {
  const auto& c = weights.config;
  struct Role {
    const QwenTensorRef QwenLayerWeights::* tensor;
    std::uint64_t cols;
    std::vector<std::string_view> names;  ///< As weights.cpp binds them.
  };
  const Role roles[] = {
      {&QwenLayerWeights::ssm_out, c.ssm_inner_size, {"ssm_out.weight"}},
      {&QwenLayerWeights::attn_output,
       c.AttentionSize(),
       {"attn_output.weight", "attn_out.weight", "wo.weight"}},
      {&QwenLayerWeights::ffn_down, c.intermediate_size, {"ffn_down.weight"}},
  };
  std::vector<lora::LoraMatrix> targets;
  for (std::size_t i = 0; i < weights.layers.size(); ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    for (const auto& role : roles) {
      const QwenTensorRef& bound = weights.layers[i].*role.tensor;
      if (bound.empty()) {
        continue;
      }
      const core::GgufTensorInfo* info = nullptr;
      for (const auto name : role.names) {
        const auto* candidate = reader.FindTensor(prefix + std::string(name));
        if (candidate != nullptr && candidate->data == bound.data) {
          info = candidate;
          break;
        }
      }
      if (info == nullptr) {
        if (error != nullptr) {
          *error = "LoRA targets: no GGUF tensor of blk." + std::to_string(i) +
                   " holds the bound " + std::string(role.names.front());
        }
        return std::nullopt;
      }
      const auto& d = info->dimensions;
      if (d.size() != 2 || d[0] != role.cols || d[1] != c.hidden_size) {
        if (error != nullptr) {
          *error = "LoRA target " + std::string(info->name) + " has shape " +
                   Shape(d) + ", expected [" + std::to_string(role.cols) +
                   ", " + std::to_string(c.hidden_size) + "]";
        }
        return std::nullopt;
      }
      targets.push_back({info->name, info->data, info->type, d[1], d[0]});
    }
  }
  return targets;
}

std::unique_ptr<lora::LoraAdapter> OpenLoraAdapter(
    const core::GgufReader& reader, const QwenModelWeights& weights,
    const lora::LoraSelection& selection, std::string* error) {
  const auto targets = LoraTargets(reader, weights, error);
  if (!targets) {
    return nullptr;
  }
  auto adapter = lora::LoraAdapter::Open(
      selection.path, selection.scale,
      reader.GetMetadataString("general.architecture").value_or(""), *targets,
      kLoraExpectedTargets, error);
  if (adapter && !LoraPatchesOwnTheirBytes(reader, *adapter, error)) {
    return nullptr;
  }
  return adapter;
}

bool LoraPatchesOwnTheirBytes(const core::GgufReader& reader,
                              const lora::LoraAdapter& adapter,
                              std::string* error) {
  for (const auto& [name, patch] : adapter.patches()) {
    const auto begin = reinterpret_cast<std::uintptr_t>(patch.target.data);
    const auto end = begin + patch.target.SizeBytes();
    for (const auto& other : reader.GetTensors()) {
      const auto other_begin = reinterpret_cast<std::uintptr_t>(other.data);
      if (other.name != name && other.data != nullptr && other_begin < end &&
          begin < other_begin + other.size_bytes) {
        if (error != nullptr) {
          *error = "LoRA target " + name + " shares its bytes with " +
                   std::string(other.name) + " in the model";
        }
        return false;
      }
    }
  }
  return true;
}

bool ValidateCpuLoraOptions(const ext::ParsedOptions& options,
                            std::string* error) {
  lora::LoraSelection selection;
  if (!lora::ResolveLoraOptions(options, &selection, error)) {
    return false;
  }
  if (!selection.path.empty()) {
    if (error != nullptr) {
      *error =
          "--lora requires the ROCm backend; the Qwen3.8 27B CPU generator "
          "cannot merge an adapter";
    }
    return false;
  }
  return true;
}

}  // namespace gufo::models::qwen
