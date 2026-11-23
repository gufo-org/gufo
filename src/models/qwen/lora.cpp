#include "src/models/qwen/lora.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>

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

std::optional<std::vector<LoraTarget>> LoraTargetsByLayer(
    const core::GgufReader& reader, const QwenModelWeights& weights,
    std::string* error) {
  const auto& c = weights.config;
  struct Role {
    const QwenTensorRef QwenLayerWeights::* tensor;
    std::uint64_t cols;
    std::vector<std::string_view> names;  ///< As weights.cpp binds them.
    bool ffn_down;
  };
  const Role roles[] = {
      {&QwenLayerWeights::ssm_out, c.ssm_inner_size, {"ssm_out.weight"}, false},
      {&QwenLayerWeights::attn_output,
       c.AttentionSize(),
       {"attn_output.weight", "attn_out.weight", "wo.weight"},
       false},
      {&QwenLayerWeights::ffn_down,
       c.intermediate_size,
       {"ffn_down.weight"},
       true},
  };
  std::vector<LoraTarget> targets;
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
      targets.push_back({{info->name, info->data, info->type, d[1], d[0]},
                         static_cast<std::uint32_t>(i),
                         role.ffn_down});
    }
  }
  return targets;
}

std::optional<std::vector<lora::LoraMatrix>> LoraTargets(
    const core::GgufReader& reader, const QwenModelWeights& weights,
    std::string* error) {
  const auto targets = LoraTargetsByLayer(reader, weights, error);
  if (!targets) {
    return std::nullopt;
  }
  std::vector<lora::LoraMatrix> matrices;
  for (const auto& target : *targets) {
    matrices.push_back(target.matrix);
  }
  return matrices;
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
      kLoraExpectedTargets, error, lora::LoraTargetFormats::kAny);
  if (adapter && LoraMergesInPlace(*adapter) &&
      !LoraPatchesOwnTheirBytes(reader, *adapter, error)) {
    return nullptr;
  }
  return adapter;
}

bool LoraMergesInPlace(const lora::LoraAdapter& adapter) {
  for (const auto& [name, patch] : adapter.patches()) {
    if (patch.target.type != core::GgmlType::kQ8_0) {
      return false;
    }
  }
  return true;
}

std::string LoraAdapterIdentity(const lora::LoraAdapter& adapter) {
  return LoraMergesInPlace(adapter)
             ? adapter.CacheIdentity()
             : adapter.CacheIdentity(kLoraSidePathAlgorithm);
}

std::optional<LoraSidePath> PlanLoraSidePath(const core::GgufReader& reader,
                                             const lora::LoraAdapter& adapter,
                                             const QwenModelWeights& weights,
                                             std::string* error) {
  const auto fail = [&](const std::string& message) {
    if (error != nullptr) {
      *error = message;
    }
    return std::nullopt;
  };
  // Bind by tensor name: the side path allows projections that share bytes,
  // so a data pointer can name more than one layer.
  const auto targets = LoraTargetsByLayer(reader, weights, error);
  if (!targets) {
    return std::nullopt;
  }
  std::map<std::string_view, const LoraTarget*, std::less<>> by_name;
  for (const auto& target : *targets) {
    by_name.emplace(target.matrix.name, &target);
  }
  const auto padded = [](std::size_t n) { return (n + 15) / 16 * 16; };
  LoraSidePath path;
  std::size_t floats = 0;
  for (const auto& [name, patch] : adapter.patches()) {
    const auto found = by_name.find(name);
    if (found == by_name.end()) {
      return fail("LoRA target " + name + " is not a bound trunk projection");
    }
    LoraSidePathSlot slot;
    slot.layer = found->second->layer;
    slot.ffn_down = found->second->ffn_down;
    if (patch.rank > kLoraSidePathMaxRank) {
      return fail("LoRA target " + name + " has rank " +
                  std::to_string(patch.rank) + ", the side path accepts " +
                  std::to_string(kLoraSidePathMaxRank) + " at most");
    }
    if (patch.target.cols % 4 != 0) {
      return fail("LoRA target " + name + " has " +
                  std::to_string(patch.target.cols) +
                  " columns, the side path needs a multiple of 4");
    }
    slot.in = static_cast<std::uint32_t>(patch.target.cols);
    slot.out = static_cast<std::uint32_t>(patch.target.rows);
    slot.rank = patch.rank;
    slot.multiplier = adapter.Multiplier(patch);
    // Scale and alpha are each finite; their product may not be.
    if (!std::isfinite(slot.multiplier)) {
      return fail("LoRA target " + name +
                  " has a non-finite multiplier (scale * alpha / rank)");
    }
    slot.a = floats;
    slot.b = floats + padded(patch.a.size());
    floats = slot.b + padded(patch.b.size());
    path.slots.push_back(slot);
  }
  path.factors.assign(floats, 0.0F);
  std::size_t i = 0;
  for (const auto& [name, patch] : adapter.patches()) {
    const auto& slot = path.slots[i++];
    std::copy(patch.a.begin(), patch.a.end(),
              path.factors.begin() + static_cast<std::ptrdiff_t>(slot.a));
    std::copy(patch.b.begin(), patch.b.end(),
              path.factors.begin() + static_cast<std::ptrdiff_t>(slot.b));
  }
  return path;
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
