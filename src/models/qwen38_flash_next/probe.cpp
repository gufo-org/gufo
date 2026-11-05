#include "src/models/qwen38_flash_next/probe.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <string>

namespace gufo::models::qwen38_flash_next {

bool ProbeFlashNextGguf(const core::GgufReader& reader) {
  const auto architecture =
      reader.GetMetadataString("general.architecture").value_or("");
  if (architecture != "qwen4exp") {
    return false;
  }
  // Mirror the routed-expert allowlist of the weights loader: files whose
  // experts use other formats (for example the Unsloth IQ quants) cannot
  // load here, and declining them lets a provider with support for those
  // formats claim the checkpoint instead of failing deep in the load.
  static constexpr std::array<core::GgmlType, 5> kExpertFormats = {
      core::GgmlType::kQ4_K, core::GgmlType::kQ5_K, core::GgmlType::kQ6_K,
      core::GgmlType::kQ5_1, core::GgmlType::kQ8_0};
  for (const auto& tensor : reader.GetTensors()) {
    const auto name = tensor.name;
    if (name.find("ffn_gate_exps.") == std::string_view::npos &&
        name.find("ffn_up_exps.") == std::string_view::npos &&
        name.find("ffn_down_exps.") == std::string_view::npos) {
      continue;
    }
    if (std::find(kExpertFormats.begin(), kExpertFormats.end(), tensor.type) ==
        kExpertFormats.end()) {
      return false;
    }
  }
  return true;
}

std::optional<sampling::TextSamplingDefaults> FlashNextSamplingDefaults() {
  return sampling::PresetDefaults(sampling::TextModelPreset::kQwen38);
}

std::optional<sampling::TextSamplingDefaults> FlashNextSamplingDefaults(
    const core::GgufReader&) {
  return FlashNextSamplingDefaults();
}

bool ProbeFlashNext(std::string_view model_root) {
  try {
    std::filesystem::path p(model_root);
    if (!std::filesystem::exists(p)) {
      return false;
    }
    if (std::filesystem::is_regular_file(p)) {
      std::string err;
      auto reader = core::GgufReader::OpenFile(p.string(), &err);
      if (!reader)
        return false;
      return ProbeFlashNextGguf(*reader);
    }
    return false;
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::qwen38_flash_next
