#include "src/models/qwen/probe.hpp"

#include <filesystem>
#include <string>

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
