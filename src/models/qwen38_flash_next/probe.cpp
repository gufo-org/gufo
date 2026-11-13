#include "src/models/qwen38_flash_next/probe.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include "src/models/qwen38_flash_next/gyro/tq_types.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

/// The trunk geometry both providers share: false unless the architecture and
/// layer counts describe a Qwen3.8-Flash-Next artifact.
bool FlashNextTrunk(const core::GgufReader& reader,
                    std::uint32_t& trunk_layers) {
  const auto architecture =
      reader.GetMetadataString("general.architecture").value_or("");
  if (architecture != "qwen4exp") {
    return false;
  }
  const auto block_count = reader.GetMetadataUint32("qwen4exp.block_count");
  const auto nextn_layers =
      reader.GetMetadataUint32("qwen4exp.nextn_predict_layers").value_or(0);
  if (!block_count || nextn_layers >= *block_count) {
    return false;
  }
  trunk_layers = *block_count - nextn_layers;
  return true;
}

/// Calls `visit` for each routed-expert weight `ModelWeights::Bind` loads: the
/// gate, up and down projections of every trunk layer. Appended MTP layers and
/// unrelated tensors are not loaded as part of the trunk.
void ForExpertTensors(const core::GgufReader& reader,
                      std::uint32_t trunk_layers,
                      const std::function<void(const core::GgufTensorInfo&)>&
                          visit) {
  for (const auto& tensor : reader.GetTensors()) {
    auto name = tensor.name;
    if (!name.starts_with("blk.")) {
      continue;
    }
    name.remove_prefix(4);
    std::uint32_t layer = 0;
    const auto [end, error] =
        std::from_chars(name.data(), name.data() + name.size(), layer);
    if (error != std::errc{} || layer >= trunk_layers ||
        (name.starts_with('0') && end != name.data() + 1)) {
      continue;
    }
    name.remove_prefix(end - name.data());
    if (name != ".ffn_gate_exps.weight" && name != ".ffn_up_exps.weight" &&
        name != ".ffn_down_exps.weight") {
      continue;
    }
    visit(tensor);
  }
}

/// Opens a model file, or null when it is not a readable GGUF. Unsized tensors
/// are core's way of saying an extension owns that layout, so each provider
/// reads the file the way its own contract allows.
std::unique_ptr<core::GgufReader> OpenModel(std::string_view model_root,
                                            bool allow_unsized_tensors) {
  try {
    std::filesystem::path p(model_root);
    if (!std::filesystem::is_regular_file(p)) {
      return nullptr;
    }
    std::string err;
    return core::GgufReader::OpenFile(
        p.string(), &err, {.allow_unsized_tensors = allow_unsized_tensors});
  } catch (...) {
    return nullptr;
  }
}

}  // namespace

bool ProbeFlashNextGguf(const core::GgufReader& reader) {
  std::uint32_t trunk_layers = 0;
  if (!FlashNextTrunk(reader, trunk_layers)) {
    return false;
  }
  static constexpr std::array<core::GgmlType, 5> kExpertFormats = {
      core::GgmlType::kQ4_K, core::GgmlType::kQ5_K, core::GgmlType::kQ6_K,
      core::GgmlType::kQ5_1, core::GgmlType::kQ8_0};
  bool canonical = true;
  ForExpertTensors(reader, trunk_layers, [&](const core::GgufTensorInfo& t) {
    canonical = canonical && std::find(kExpertFormats.begin(),
                                       kExpertFormats.end(),
                                       t.type) != kExpertFormats.end();
  });
  return canonical;
}

bool ProbeFlashNextGyroGguf(const core::GgufReader& reader) {
  std::uint32_t trunk_layers = 0;
  if (!FlashNextTrunk(reader, trunk_layers)) {
    return false;
  }
  std::size_t experts = 0;
  std::size_t trellis = 0;
  ForExpertTensors(reader, trunk_layers, [&](const core::GgufTensorInfo& t) {
    ++experts;
    if (gyro::IsTQ2T(t.type)) {
      ++trellis;
    }
  });
  // A Gyro artifact replaces every trunk expert tensor with trellis rows; a
  // partly trellised file belongs to neither provider.
  return experts != 0 && trellis == experts;
}

std::optional<sampling::TextSamplingDefaults> FlashNextSamplingDefaults() {
  return sampling::PresetDefaults(sampling::TextModelPreset::kQwen38);
}

std::optional<sampling::TextSamplingDefaults> FlashNextSamplingDefaults(
    const core::GgufReader&) {
  return FlashNextSamplingDefaults();
}

bool ProbeFlashNext(std::string_view model_root) {
  const auto reader = OpenModel(model_root, false);
  return reader != nullptr && ProbeFlashNextGguf(*reader);
}

bool ProbeFlashNextGyro(std::string_view model_root) {
  const auto reader = OpenModel(model_root, true);
  return reader != nullptr && ProbeFlashNextGyroGguf(*reader);
}

}  // namespace gufo::models::qwen38_flash_next
