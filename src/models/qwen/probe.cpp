#include "src/models/qwen/probe.hpp"

#include <filesystem>
#include <string>

#include "src/families/qwen/config/qwen_config.hpp"

namespace gufo::models::qwen {

bool ProbeQwenGguf(const core::GgufReader& reader) {
  const auto architecture =
      reader.GetMetadataString("general.architecture").value_or("");
  if (architecture == "deepseek4" || architecture == "qwen4exp") {
    return false;
  }
  if (architecture != "qwen" && architecture != "qwen2" &&
      architecture != "qwen35" && architecture != "qwen38") {
    return false;
  }
  const std::string prefix = std::string(architecture) + ".";
  const auto embedding_length =
      reader.GetMetadataUint64("qwen35.embedding_length")
          .value_or(reader.GetMetadataUint64(prefix + "embedding_length")
                        .value_or(0));
  return embedding_length == 5120;
}

std::optional<sampling::TextSamplingDefaults> QwenSamplingDefaults(
    const core::ModelConfig& config) {
  return sampling::PresetDefaults(sampling::TextPreset(config));
}

std::optional<sampling::TextSamplingDefaults> QwenSamplingDefaults(
    const core::GgufReader& reader) {
  const auto config = ExtractQwenConfig(reader);
  if (!config.has_value()) {
    return std::nullopt;
  }
  return QwenSamplingDefaults(*config);
}

bool ProbeQwen35_4BGguf(const core::GgufReader& reader) {
  if (reader.GetMetadataString("general.architecture").value_or("") !=
      "qwen35") {
    return false;
  }
  if (reader.GetMetadataUint64("qwen35.embedding_length").value_or(0) != 2560) {
    return false;
  }
  // An embedded MTP layer belongs to a larger Qwen artifact; the shared
  // trunk kernels for this shape have no validated draft path there.
  return reader.GetMetadataUint32("qwen35.nextn_predict_layers").value_or(0) ==
         0;
}

bool ProbeQwen35_4B(std::string_view model_root) {
  try {
    std::filesystem::path p(model_root);
    if (!std::filesystem::is_regular_file(p)) {
      return false;
    }
    std::string err;
    auto reader = core::GgufReader::OpenFile(p.string(), &err);
    if (!reader) {
      return false;
    }
    return ProbeQwen35_4BGguf(*reader);
  } catch (...) {
    return false;
  }
}

bool ProbeQwen(std::string_view model_root) {
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
      return ProbeQwenGguf(*reader);
    }
    return false;
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::qwen
