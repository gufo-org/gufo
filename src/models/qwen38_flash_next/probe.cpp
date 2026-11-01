#include "src/models/qwen38_flash_next/probe.hpp"

#include <filesystem>
#include <string>

namespace gufo::models::qwen38_flash_next {

bool ProbeFlashNextGguf(const core::GgufReader& reader) {
  const auto architecture =
      reader.GetMetadataString("general.architecture").value_or("");
  return architecture == "qwen4exp";
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
